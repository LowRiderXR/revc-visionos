#include "common.h"

#include "main.h"
#include "General.h"
#include "Entity.h"
#include "RenderBuffer.h"
#include "TxdStore.h"
#include "Camera.h"
#include "Sprite.h"
#include "Timer.h"
#include "World.h"
#include "Weather.h"
#include "Collision.h"
#include "Timecycle.h"
#include "Coronas.h"
#include "PointLights.h"
#include "Shadows.h"
#include "Clock.h"
#include "Bridge.h"

#ifdef LIBRW_VISIONOS
extern "C" int vc_render_mode(void);   // gl3device: 1 = stereo VR, 0 = cinema
#ifdef LIBRW_VISIONOS
extern "C" int vc_get_eye_view_mv(int eye, float m[16]);   // one-pass: per-eye views the GPU twins use
extern "C" int vc_get_eye_proj(float m[16]);
extern "C" void vc_im3d_pull(float delta);   // librw gl3immed: per-eye view-space depth pull for the next im3d draws

// S1 Koronas (multiview-plan.md, Sprite-Familie S1, Entscheidung B): in stereo the corona
// disc is no longer a CPU-projected 2D sprite (projected ONCE with the centre camera and
// therefore drawn at the same place in both eyes = optical infinity, "headlights sit
// offset"), but a WORLD-SPACE billboard drawn through im3d. The GPU projects its four
// vertices with the same per-eye matrices as the world geometry -> correct disparity, no
// CPU knowledge of the eyes needed. Size mapping is exact: the game computes the sprite
// half-size in pixels as outh*size*fog with outh = focal/zorig, so the WORLD half-size at
// the true position is simply wpx/outh (= size*fog; the game's FOV-zoom factor cancels).
// The depth pull of the 2D path ("z -= nearDist", sun at 0.95*far) is NOT done by moving
// the quad along the centre-camera ray -- first device run (2026-09-30) showed why: a
// point pulled 1.5 m toward the head centre at 2 m distance gets the disparity of a
// 0.5 m object, floats in front of the lamp and swings on head turns. Instead the pull
// happens in the im3d vertex shader per eye (u_im3dPull: homothety about the eye in
// view space keeps each eye's screen position, changes only depth).
// Kept from the 2D path: the 20/z radian roll, the near fade (< 2.3 m), additive blend,
// ZTEST per LOScheck.
// VC_WORLD_CORONA=0 restores the 2D sprite path (A/B).
static bool vcWorldCoronaOn(void)
{
	static int e = -1;
	if(e < 0){ const char *s = getenv("VC_WORLD_CORONA"); e = (s && s[0] == '0') ? 0 : 1; }
	return e != 0 && vc_render_mode() == 1;
}
// Wet-road reflections (RenderReflections) as world quads; separate A/B switch so they can be
// isolated from the coronas: VC_WORLD_REFLECT=0 = stock 2D sprites (needs the corona path on).
static bool vcWorldReflectOn(void)
{
	static int e = -1;
	if(e < 0){ const char *s = getenv("VC_WORLD_REFLECT"); e = (s && s[0] == '0') ? 0 : 1; }
	return e != 0 && vcWorldCoronaOn();
}
// Known original behaviour (checked against the macOS reference build 2026-09-30): the
// coronas of the VERTICAL traffic lights are half hidden by their housing, the horizontal
// ones are not. Same in the flat game -- not a stereo defect, not fixed here.

// Probe VC_CORONA_DIAG=1: once per second, for the nearest corona drawn this frame, the
// per-eye screen x of the quad centre (what the GPU will draw), the disparity that the 2D
// sprite lacked, and the pixel-size identity (world half-size back-projected must equal
// the game's pixel half-size, tolerance <= 1 px).
struct VcCoronaDiag { float z, zTrue, wpx, worldHalf, x2d, pullSent; CVector pos; bool valid, ztest; };
static VcCoronaDiag vcCoronaNearest;
static int vcCoronaDrawn;
static float vcCoronaAxLen = 0.0f, vcCoronaAyLen = 0.0f, vcCoronaAxAyDot = 0.0f;   // raw axis check (before normalisation)

static void
vcRenderCoronaWorldQuad(const CVector &coors, float zorig, float zdraw, float outh,
	float wpx, float hpx, uint8 r, uint8 g, uint8 b, int16 intens, float rotation, uint8 a,
	bool nearFade, float x2d)
{
	if(zorig <= 0.001f || outh <= 0.0f || zdraw <= 0.001f) return;
	// near fade of RenderOneXLUSprite_Rotate_Aspect
	if(nearFade && zdraw < 2.3f){
		if(zdraw < 1.3f) return;
		int f = (zdraw - 1.3f)/(2.3f-1.3f) * 255;
		r = f*r >> 8; g = f*g >> 8; b = f*b >> 8; intens = f*intens >> 8;
	}
	// the quad sits at the TRUE position; the depth pull is a per-eye shader operation
	const CVector &centre = coors;
	// billboard axes = the world directions that map to view +x / +y, i.e. exactly the
	// screen axes CalcScreenCoors uses. m_viewMatrix is a world->view transform whose
	// COLUMNS are the view-space images of the world axes, stored in CMatrix slot order
	// Right (col 0, multiplies in.x), Forward (col 1, in.y), Up (col 2, in.z). The world
	// direction mapping to view +x is therefore row 0 = (Right.x, Forward.x, Up.x).
	// Device run 2 (2026-09-30) had Forward/Up swapped here -> non-orthonormal axes ->
	// the disc became a tilted streak that turned with the head. Normalised + re-
	// orthogonalised below so a scaled or slightly skewed view matrix cannot do that again.
	const CMatrix &V = TheCamera.m_viewMatrix;
	CVector vx(V.GetRight().x, V.GetForward().x, V.GetUp().x);   // view +x in world (screen right)
	CVector vy(V.GetRight().y, V.GetForward().y, V.GetUp().y);   // view +y in world (screen down)
	// Device run 6 (2026-09-30, screenshot with the head ROLLED): a screen-aligned billboard
	// rolls with the head -- the wide TYPE_STREAK headlight flares then lie tilted against
	// the world while the HUD stays level with them. The flat game never rolls its camera,
	// so "screen-horizontal" there equals "world-horizontal". Reproduce that: build the
	// billboard from the WORLD up axis (GTA: z), i.e. an upright billboard whose horizontal
	// axis is the true horizon regardless of head roll. Fallback to the screen axes only when
	// looking (almost) straight up/down, where the horizon is undefined.
	CVector fwd = coors - TheCamera.GetPosition();
	float fl = fwd.Magnitude(); if(fl < 1e-4f) return;
	fwd *= 1.0f / fl;
	CVector ax = CrossProduct(fwd, CVector(0.0f, 0.0f, 1.0f));   // world-horizontal, perpendicular to the line of sight
	CVector ay;
	if(ax.Magnitude() > 0.05f){
		ax.Normalise();
		ay = CrossProduct(ax, fwd);                                // in the vertical plane through the line of sight
		ay.Normalise();
		// keep the texture orientation of the 2D sprite: x along screen right, y along screen down
		if(DotProduct(ax, vx) < 0.0f) ax = -ax;
		if(DotProduct(ay, vy) < 0.0f) ay = -ay;
	}else{
		ax = vx; ay = vy;
		float lx = ax.Magnitude(), ly = ay.Magnitude();
		if(lx < 1e-4f || ly < 1e-4f) return;
		ax *= 1.0f / lx; ay -= ax * DotProduct(ax, ay);
		float l = ay.Magnitude(); if(l < 1e-4f) return;
		ay *= 1.0f / l;
	}
	vcCoronaAxLen = ax.Magnitude(); vcCoronaAyLen = ay.Magnitude(); vcCoronaAxAyDot = DotProduct(ax, ay);
	// pixel half-size -> world half-size at the true depth: wpx/outh (outh = focal/zorig)
	float w = wpx / outh, h = hpx / outh;
	float c = Cos(rotation), sn = Sin(rotation);
	// same corner order/uv as the 2D sprite: 0 (-w,-h) 1 (-w,+h) 2 (+w,+h) 3 (+w,-h)
	float sx[4] = { w*(-c-sn), w*(-c+sn), w*(+c+sn), w*(+c-sn) };
	float sy[4] = { h*(-c+sn), h*(+c+sn), h*(+c-sn), h*(-c-sn) };
	float us[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
	float vs[4] = { 0.0f, 1.0f, 1.0f, 0.0f };
	static RwIm3DVertex verts[4];
	static RwImVertexIndex idx[6] = { 0, 1, 2, 0, 2, 3 };
	for(int i = 0; i < 4; i++){
		CVector pv = centre + ax * sx[i] + ay * sy[i];
		RwIm3DVertexSetPos(&verts[i], pv.x, pv.y, pv.z);
		RwIm3DVertexSetRGBA(&verts[i], r*intens>>8, g*intens>>8, b*intens>>8, a);
		RwIm3DVertexSetU(&verts[i], us[i]);
		RwIm3DVertexSetV(&verts[i], vs[i]);
	}
	// per-eye depth pull: view z becomes zdraw instead of zorig, screen position unchanged
	// (the game's nearDist pull, and 0.95*far for the sun).
	float pull = zdraw / zorig - 1.0f;
	if(pull < -0.95f) pull = -0.95f;   // never collapse onto the eye
	vc_im3d_pull(pull);
	if(RwIm3DTransform(verts, 4, nil, rwIM3D_VERTEXXYZ|rwIM3D_VERTEXRGBA|rwIM3D_VERTEXUV)){
		RwIm3DRenderIndexedPrimitive(rwPRIMTYPETRILIST, idx, 6);
		RwIm3DEnd();
	}
	vc_im3d_pull(0.0f);
	vcCoronaDrawn++;
	if(!vcCoronaNearest.valid || zdraw < vcCoronaNearest.z){
		vcCoronaNearest.valid = true; vcCoronaNearest.z = zdraw; vcCoronaNearest.zTrue = zorig; vcCoronaNearest.wpx = wpx;
		vcCoronaNearest.worldHalf = w; vcCoronaNearest.pos = centre; vcCoronaNearest.x2d = x2d;
		vcCoronaNearest.pullSent = pull;
		void *zt = nil; RwRenderStateGet(rwRENDERSTATEZTESTENABLE, &zt); vcCoronaNearest.ztest = zt != nil;
	}
}

static void
vcCoronaDiagFlush(void)
{
	static int diagOn = -1;
	if(diagOn < 0){ const char *e = getenv("VC_CORONA_DIAG"); diagOn = (e && e[0] == '1') ? 1 : 0; }
	if(!diagOn){ vcCoronaNearest.valid = false; vcCoronaDrawn = 0; return; }
	static float lastT = 0.0f;
	float nowT = CTimer::GetTimeInMilliseconds() * 0.001f;
	if(nowT - lastT >= 1.0f){
		lastT = nowT;
		if(!vcCoronaNearest.valid){
			printf("[vc-corona] drawn=%d (none this frame)\n", vcCoronaDrawn);
		}else{
			float v0[16], v1[16], p[16];
			if(vc_get_eye_view_mv(0, v0) && vc_get_eye_view_mv(1, v1) && vc_get_eye_proj(p)){
				const CVector &q = vcCoronaNearest.pos;
				auto screenX = [&](const float *v) -> float {
					float lx = v[0]*q.x + v[4]*q.y + v[8]*q.z  + v[12];
					float ly = v[1]*q.x + v[5]*q.y + v[9]*q.z  + v[13];
					float lz = v[2]*q.x + v[6]*q.y + v[10]*q.z + v[14];
					float cx = p[0]*lx + p[4]*ly + p[8]*lz + p[12];
					float cw = p[3]*lx + p[7]*ly + p[11]*lz + p[15];
					return (cw > 0.0001f) ? (cx/cw * 0.5f + 0.5f) * SCREEN_WIDTH : -1.0f;
				};
				float x0 = screenX(v0), x1 = screenX(v1);
				float degPerPx = (p[0] > 0.0001f) ? (2.0f * Atan(1.0f / p[0]) * 180.0f / PI) / SCREEN_WIDTH : 0.0f;
				// size: world half-size back through the slice focal (0.5*p5*H) at the TRUE
				// depth; equals the game's pixel size when the game FOV is at its default
				// (the ratio is the game's FOV-zoom factor, which the VR view does not apply)
				float focal = 0.5f * p[5] * SCREEN_HEIGHT;
				float wpxBack = vcCoronaNearest.worldHalf * focal / vcCoronaNearest.zTrue;
				printf("[vc-corona] drawn=%d nearest dist=%.1fm (drawn depth %.1fm, pull x%.2f) | quad centre per eye x0=%.0f x1=%.0f -> disparity %.0f px = %.2f deg (2D sprite would sit at x=%.0f in both) | size: game %.1f px, quad %.1f px | axes |ax|=%.3f |ay|=%.3f ax.ay=%.3f (want 1/1/0) | pullSent=%.3f ztest=%d\n",
				       vcCoronaDrawn, vcCoronaNearest.zTrue, vcCoronaNearest.z, vcCoronaNearest.z / vcCoronaNearest.zTrue, x0, x1, x0 - x1, (x0 - x1) * degPerPx,
				       vcCoronaNearest.x2d, vcCoronaNearest.wpx, wpxBack, vcCoronaAxLen, vcCoronaAyLen, vcCoronaAxAyDot,
				       vcCoronaNearest.pullSent, vcCoronaNearest.ztest ? 1 : 0);
			}else{
				printf("[vc-corona] drawn=%d nearest dist=%.1fm (two-pass: per-eye matrices come from the eye pass itself, no diagnostics)\n",
				       vcCoronaDrawn, vcCoronaNearest.z);
			}
		}
	}
	vcCoronaNearest.valid = false; vcCoronaDrawn = 0;
}
#endif
#endif

struct FlareDef
{
	float position;
	float size;
	int16 red;
	int16 green;
	int16 blue;
	int16 alpha;
	int16 texture;
};

FlareDef SunFlareDef[] = {
	{ -0.5f, 15.0f, 50, 50, 0, 200, 1 },
	{ -1.0f, 10.0f, 50, 20, 0, 200, 2 },
	{ -1.5f, 15.0f, 50, 0, 0, 200, 3 },
	{ -2.5f, 25.0f, 50, 0, 0, 200, 1 },
	{ 0.5f, 12.5f, 40, 40, 25, 200, 1 },
	{ 0.05f, 20.0f, 30, 22, 9, 200, 2 },
	{ 1.3f, 7.5f, 50, 30, 9, 200, 3 },
	{ 0.0f, 0.0f, 255, 255, 255, 255, 0 }
};

FlareDef HeadLightsFlareDef[] = {
	{ -0.5f, 15.5, 70, 70, 70, 200, 1 },
	{ -1.0f, 10.0, 70, 70, 70, 200, 2 },
	{ -1.5f, 5.5f, 50, 50, 50, 200, 3 },
	{ 0.5f, 12.0f, 50, 50, 50, 200, 1 },
	{ 0.05f, 20.0f, 40, 40, 40, 200, 2 },
	{ 1.3f, 8.0f, 60, 60, 60, 200, 3 },
	{ -2.0f, 12.0f, 50, 50, 50, 200, 1 },
	{ -2.3f, 15.0f, 40, 40, 40, 200, 2 },
	{ -3.0f, 16.0f, 40, 40, 40, 200, 3 },
	{ 0.0f, 0.0f, 255, 255, 255, 255, 0 }
};


RwTexture *gpCoronaTexture[9] = { nil, nil, nil, nil, nil, nil, nil, nil, nil };

float CCoronas::LightsMult = 1.0f;
float CCoronas::SunScreenX;
float CCoronas::SunScreenY;
int CCoronas::MoonSize;
bool CCoronas::SunBlockedByClouds;
int CCoronas::bChangeBrightnessImmediately;

CRegisteredCorona CCoronas::aCoronas[NUMCORONAS];

const char aCoronaSpriteNames[][32] = {
	"coronastar",
	"corona",
	"coronamoon",
	"coronareflect",
	"coronaheadlightline",
	"coronahex",
	"coronacircle",
	"coronaringa",
	"streek"
};

void
CCoronas::Init(void)
{
	int i;

	CTxdStore::PushCurrentTxd();
	CTxdStore::SetCurrentTxd(CTxdStore::FindTxdSlot("particle"));

	for(i = 0; i < 9; i++)
		if(gpCoronaTexture[i] == nil)
			gpCoronaTexture[i] = RwTextureRead(aCoronaSpriteNames[i], nil);

	CTxdStore::PopCurrentTxd();

	for(i = 0; i < NUMCORONAS; i++)
		aCoronas[i].id = 0;
}

void
CCoronas::Shutdown(void)
{
	int i;
	for(i = 0; i < 9; i++)
		if(gpCoronaTexture[i]){
			RwTextureDestroy(gpCoronaTexture[i]);
			gpCoronaTexture[i] = nil;
		}
}

void
CCoronas::Update(void)
{
	int i;
	static int LastCamLook = 0;

	LightsMult = Min(LightsMult + 0.03f * CTimer::GetTimeStep(), 1.0f);

	int CamLook = 0;
	if(TheCamera.Cams[TheCamera.ActiveCam].LookingLeft) CamLook |= 1;
	if(TheCamera.Cams[TheCamera.ActiveCam].LookingRight) CamLook |= 2;
	if(TheCamera.Cams[TheCamera.ActiveCam].LookingBehind) CamLook |= 4;
	// BUG?
	if(TheCamera.GetLookDirection() == LOOKING_BEHIND) CamLook |= 8;

	if(LastCamLook != CamLook)
		bChangeBrightnessImmediately = 3;
	else
		bChangeBrightnessImmediately = Max(bChangeBrightnessImmediately-1, 0);
	LastCamLook = CamLook;

	for(i = 0; i < NUMCORONAS; i++)
		if(aCoronas[i].id != 0)
			aCoronas[i].Update();
}

void
CCoronas::RegisterCorona(uint32 id, uint8 red, uint8 green, uint8 blue, uint8 alpha,
	const CVector &coors, float size, float drawDist, RwTexture *tex,
	int8 flareType, uint8 reflection, uint8 LOScheck, uint8 drawStreak, float someAngle,
	bool useNearDist, float nearDist)
{
	int i;

	if(sq(drawDist) < (TheCamera.GetPosition() - coors).MagnitudeSqr2D())
		return;

	if(useNearDist){
		float dist = (TheCamera.GetPosition() - coors).Magnitude();
		if(dist < 35.0f)
			return;
		if(dist < 50.0f)
			alpha *= (dist - 35.0f)/(50.0f - 35.0f);
	}

	for(i = 0; i < NUMCORONAS; i++)
		if(aCoronas[i].id == id)
			break;

	if(i == NUMCORONAS){
		// add a new one

		// find empty slot
		for(i = 0; i < NUMCORONAS; i++)
			if(aCoronas[i].id == 0)
				break;
		if(i == NUMCORONAS)
			return;		// no space

		aCoronas[i].fadeAlpha = 0;
		aCoronas[i].offScreen = true;
		aCoronas[i].firstUpdate = true;
		aCoronas[i].renderReflection = false;
		aCoronas[i].lastLOScheck = 0;
		aCoronas[i].sightClear = false;
		aCoronas[i].hasValue[0] = false;
		aCoronas[i].hasValue[1] = false;
		aCoronas[i].hasValue[2] = false;
		aCoronas[i].hasValue[3] = false;
		aCoronas[i].hasValue[4] = false;
		aCoronas[i].hasValue[5] = false;

	}else{
		// use existing one

		if(aCoronas[i].fadeAlpha == 0 && alpha == 0){
			// unregister
			aCoronas[i].id = 0;
			return;
		}
	}

	aCoronas[i].id = id;
	aCoronas[i].red = red;
	aCoronas[i].green = green;
	aCoronas[i].blue = blue;
	aCoronas[i].alpha = alpha;
	aCoronas[i].coors = coors;
	aCoronas[i].size = size;
	aCoronas[i].someAngle = someAngle;
	aCoronas[i].registeredThisFrame = true;
	aCoronas[i].drawDist = drawDist;
	aCoronas[i].texture = tex;
	aCoronas[i].flareType = flareType;
	aCoronas[i].reflection = reflection;
	aCoronas[i].LOScheck = LOScheck;
	aCoronas[i].drawStreak = drawStreak;
	aCoronas[i].useNearDist = useNearDist;
	aCoronas[i].nearDist = nearDist;
}

void
CCoronas::RegisterCorona(uint32 id, uint8 red, uint8 green, uint8 blue, uint8 alpha,
	const CVector &coors, float size, float drawDist, uint8 type,
	int8 flareType, uint8 reflection, uint8 LOScheck, uint8 drawStreak, float someAngle,
	bool useNearDist, float nearDist)
{
	RegisterCorona(id, red, green, blue, alpha, coors, size, drawDist,
		gpCoronaTexture[type], flareType, reflection, LOScheck, drawStreak, someAngle,
		useNearDist, nearDist);
}

void
CCoronas::UpdateCoronaCoors(uint32 id, const CVector &coors, float drawDist, float someAngle)
{
	int i;

	if(sq(drawDist) < (TheCamera.GetPosition() - coors).MagnitudeSqr2D())
		return;

	for(i = 0; i < NUMCORONAS; i++)
		if(aCoronas[i].id == id)
			break;

	if(i == NUMCORONAS)
		return;

	if(aCoronas[i].fadeAlpha == 0)
		aCoronas[i].id = 0;	// faded out, remove
	else{
		aCoronas[i].coors = coors;
		aCoronas[i].someAngle = someAngle;
	}
}

static RwIm2DVertex vertexbufferX[2];

void
CCoronas::Render(void)
{
	int i, j;
	int screenw, screenh;

	PUSH_RENDERGROUP("CCoronas::Render");

	screenw = RwRasterGetWidth(RwCameraGetRaster(Scene.camera));
	screenh = RwRasterGetHeight(RwCameraGetRaster(Scene.camera));

	RwRenderStateSet(rwRENDERSTATEZWRITEENABLE, (void*)FALSE);
	RwRenderStateSet(rwRENDERSTATEVERTEXALPHAENABLE, (void*)TRUE);
	RwRenderStateSet(rwRENDERSTATESRCBLEND, (void*)rwBLENDONE);
	RwRenderStateSet(rwRENDERSTATEDESTBLEND, (void*)rwBLENDONE);
#ifdef LIBRW_VISIONOS
	// S1: the world quads go through the im3d shader, which fogs per vertex -- the 2D
	// sprites never were fogged (the game dims them via fogscale instead). Fog off and
	// no back-face cull for the billboards; both restored at the end of Render.
	void *vcSavedFog = nil, *vcSavedCull = nil;
	const bool vcWorldCorona = vcWorldCoronaOn();
	if(vcWorldCorona){
		RwRenderStateGet(rwRENDERSTATEFOGENABLE, &vcSavedFog);
		RwRenderStateGet(rwRENDERSTATECULLMODE, &vcSavedCull);
		RwRenderStateSet(rwRENDERSTATEFOGENABLE, (void*)FALSE);
		RwRenderStateSet(rwRENDERSTATECULLMODE, (void*)rwCULLMODECULLNONE);
	}
#endif

	for(i = 0; i < NUMCORONAS; i++){
		for(j = 5; j > 0; j--){
			aCoronas[i].prevX[j] = aCoronas[i].prevX[j-1];
			aCoronas[i].prevY[j] = aCoronas[i].prevY[j-1];
			aCoronas[i].prevRed[j] = aCoronas[i].prevRed[j-1];
			aCoronas[i].prevGreen[j] = aCoronas[i].prevGreen[j-1];
			aCoronas[i].prevBlue[j] = aCoronas[i].prevBlue[j-1];
			aCoronas[i].hasValue[j] = aCoronas[i].hasValue[j-1];
		}
		aCoronas[i].hasValue[0] = false;

		if(aCoronas[i].id == 0 ||
		   aCoronas[i].fadeAlpha == 0 && aCoronas[i].alpha == 0)
			continue;

		CVector spriteCoors;
		float spritew, spriteh;
		if(!CSprite::CalcScreenCoors(aCoronas[i].coors, &spriteCoors, &spritew, &spriteh, true)){
			aCoronas[i].offScreen = true;
			aCoronas[i].sightClear = false;
		}else{
			aCoronas[i].offScreen = false;

			if(spriteCoors.x < 0.0f || spriteCoors.y < 0.0f ||
			   spriteCoors.x > screenw || spriteCoors.y > screenh){
				aCoronas[i].offScreen = true;
				aCoronas[i].sightClear = false;
			}else{
				if(CTimer::GetTimeInMilliseconds() > aCoronas[i].lastLOScheck + 2000){
					aCoronas[i].lastLOScheck = CTimer::GetTimeInMilliseconds();
					aCoronas[i].sightClear = CWorld::GetIsLineOfSightClear(
						aCoronas[i].coors, TheCamera.Cams[TheCamera.ActiveCam].Source,
						true, true, false, false, false, true, false);
				}
			
				// add new streak point
				if(aCoronas[i].sightClear){
					aCoronas[i].prevX[0] = spriteCoors.x;
					aCoronas[i].prevY[0] = spriteCoors.y;
					aCoronas[i].prevRed[0] = aCoronas[i].red;
					aCoronas[i].prevGreen[0] = aCoronas[i].green;
					aCoronas[i].prevBlue[0] = aCoronas[i].blue;
					aCoronas[i].hasValue[0] = true;
				}
			
				// if distance too big, break streak
				if(aCoronas[i].hasValue[1]){
					if(Abs(aCoronas[i].prevX[0] - aCoronas[i].prevX[1]) > 50.0f ||
					   Abs(aCoronas[i].prevY[0] - aCoronas[i].prevY[1]) > 50.0f)
						aCoronas[i].hasValue[0] = false;
				}
			}


			if(aCoronas[i].fadeAlpha && spriteCoors.z < aCoronas[i].drawDist){
				float recipz = 1.0f/spriteCoors.z;
				float fadeDistance = aCoronas[i].drawDist / 2.0f;
				float distanceFade = spriteCoors.z < fadeDistance ? 1.0f : 1.0f - (spriteCoors.z - fadeDistance)/fadeDistance;
				int totalFade = aCoronas[i].fadeAlpha * distanceFade;

				if(aCoronas[i].LOScheck)
					RwRenderStateSet(rwRENDERSTATEZTESTENABLE, (void*)FALSE);
				else
					RwRenderStateSet(rwRENDERSTATEZTESTENABLE, (void*)TRUE);

				// render corona itself
				if(aCoronas[i].texture){
					float fogscale = CWeather::Foggyness*Min(spriteCoors.z, 40.0f)/40.0f + 1.0f;
					if(CCoronas::aCoronas[i].id == SUN_CORE)
						spriteCoors.z = 0.95f * RwCameraGetFarClipPlane(Scene.camera);
					RwRenderStateSet(rwRENDERSTATETEXTURERASTER, RwTextureGetRaster(aCoronas[i].texture));
					spriteCoors.z -= aCoronas[i].nearDist;
#ifdef LIBRW_VISIONOS
					if(vcWorldCorona){
						// S1: world-space billboard instead of the 2D sprite (see helper)
						const float zorig = 1.0f / recipz;
						if(aCoronas[i].texture == gpCoronaTexture[8]){
							float f = 1.0f - aCoronas[i].someAngle*2.0f/PI;
							float wscale = 6.0f*sq(sq(sq(f))) + 0.5f;
							float hscale = Max(0.35f - (wscale - 0.5f) * 0.06f, 0.15f);
							vcRenderCoronaWorldQuad(aCoronas[i].coors, zorig, spriteCoors.z, spriteh,
								spritew * aCoronas[i].size * wscale, spriteh * aCoronas[i].size * fogscale * hscale,
								aCoronas[i].red / fogscale, aCoronas[i].green / fogscale, aCoronas[i].blue / fogscale,
								totalFade, 0.0f, 255, false, spriteCoors.x);
						}else{
							vcRenderCoronaWorldQuad(aCoronas[i].coors, zorig, spriteCoors.z, spriteh,
								spritew * aCoronas[i].size * fogscale, spriteh * aCoronas[i].size * fogscale,
								aCoronas[i].red / fogscale, aCoronas[i].green / fogscale, aCoronas[i].blue / fogscale,
								totalFade, 20.0f * recipz, 255, true, spriteCoors.x);
						}
					}else
#endif
					if(aCoronas[i].texture == gpCoronaTexture[8]){
						// what's this?
						float f = 1.0f - aCoronas[i].someAngle*2.0f/PI;
						float wscale = 6.0f*sq(sq(sq(f))) + 0.5f;
						float hscale = 0.35f - (wscale - 0.5f) * 0.06f;
						hscale = Max(hscale, 0.15f);

						CSprite::RenderOneXLUSprite(spriteCoors.x, spriteCoors.y, spriteCoors.z,
							spritew * aCoronas[i].size * wscale,
							spriteh * aCoronas[i].size * fogscale * hscale,
							CCoronas::aCoronas[i].red / fogscale,
							CCoronas::aCoronas[i].green / fogscale,
							CCoronas::aCoronas[i].blue / fogscale,
							totalFade,
							recipz,
							255);
					}else{
						CSprite::RenderOneXLUSprite_Rotate_Aspect(
							spriteCoors.x, spriteCoors.y, spriteCoors.z,
							spritew * aCoronas[i].size * fogscale,
							spriteh * aCoronas[i].size * fogscale,
							CCoronas::aCoronas[i].red / fogscale,
							CCoronas::aCoronas[i].green / fogscale,
							CCoronas::aCoronas[i].blue / fogscale,
							totalFade,
							recipz,
							20.0f * recipz,
							255);
					}
				}

				// render flares
				bool renderFlares = aCoronas[i].flareType != FLARE_NONE;
#ifdef LIBRW_VISIONOS
				// Stereo VR decision (permanent, not a TODO): lens flares (SUN + HEADLIGHTS)
				// are placed in SCREEN space -- each flare element is strung along the line
				// from the corona toward the SCREEN CENTRE ((x - screenw/2)*pos + screenw/2).
				// That centre moves with the head, so the flare slides across the scene on
				// head-turn (headlights on moving cars are the worst) and breaks world-lock.
				// Drop all flares in stereo; the corona disc/glow itself stays world-anchored.
				if(vc_render_mode() == 1) renderFlares = false;
#endif
				if(renderFlares){
					FlareDef *flare;

					switch(aCoronas[i].flareType){
					case FLARE_SUN: flare = SunFlareDef; break;
					case FLARE_HEADLIGHTS: flare = HeadLightsFlareDef; break;
					default: assert(0);
					}

					for(; flare->texture; flare++){
						RwRenderStateSet(rwRENDERSTATETEXTURERASTER, RwTextureGetRaster(gpCoronaTexture[flare->texture + 4]));
						CSprite::RenderOneXLUSprite(
							(spriteCoors.x - (screenw/2)) * flare->position + (screenw/2),
							(spriteCoors.y - (screenh/2)) * flare->position + (screenh/2),
							spriteCoors.z,
							4.0f*flare->size * spritew/spriteh,
							4.0f*flare->size,
							(flare->red * aCoronas[i].red)>>8,
							(flare->green * aCoronas[i].green)>>8,
							(flare->blue * aCoronas[i].blue)>>8,
							(totalFade * flare->alpha)>>8,
							recipz, 255);
					}
				}
			}
		}
	}

	RwRenderStateSet(rwRENDERSTATEZWRITEENABLE, (void*)FALSE);
	RwRenderStateSet(rwRENDERSTATEZTESTENABLE, (void*)FALSE);
	RwRenderStateSet(rwRENDERSTATEVERTEXALPHAENABLE, (void*)TRUE);
	RwRenderStateSet(rwRENDERSTATESRCBLEND, (void*)rwBLENDONE);
	RwRenderStateSet(rwRENDERSTATEDESTBLEND, (void*)rwBLENDONE);
	RwRenderStateSet(rwRENDERSTATETEXTURERASTER, nil);
#ifdef LIBRW_VISIONOS
	if(vcWorldCorona){
		RwRenderStateSet(rwRENDERSTATEFOGENABLE, vcSavedFog);
		RwRenderStateSet(rwRENDERSTATECULLMODE, vcSavedCull);
		vcCoronaDiagFlush();
	}
#endif

	// streaks
	for(i = 0; i < NUMCORONAS; i++){
		if(aCoronas[i].id == 0 || !aCoronas[i].drawStreak)
			continue;

		for(j = 0; j < 5; j++){
			if(!aCoronas[i].hasValue[j] || !aCoronas[i].hasValue[j+1])
				continue;

			int alpha1 = (float)(6 - j) / 6 * 128;
			int alpha2 = (float)(6 - (j+1)) / 6 * 128;

			RwIm2DVertexSetScreenX(&vertexbufferX[0], aCoronas[i].prevX[j]);
			RwIm2DVertexSetScreenY(&vertexbufferX[0], aCoronas[i].prevY[j]);
			RwIm2DVertexSetIntRGBA(&vertexbufferX[0], aCoronas[i].prevRed[j] * alpha1 / 256, aCoronas[i].prevGreen[j] * alpha1 / 256, aCoronas[i].prevBlue[j] * alpha1 / 256, 255);
			RwIm2DVertexSetScreenX(&vertexbufferX[1], aCoronas[i].prevX[j+1]);
			RwIm2DVertexSetScreenY(&vertexbufferX[1], aCoronas[i].prevY[j+1]);
			RwIm2DVertexSetIntRGBA(&vertexbufferX[1], aCoronas[i].prevRed[j+1] * alpha2 / 256, aCoronas[i].prevGreen[j+1] * alpha2 / 256, aCoronas[i].prevBlue[j+1] * alpha2 / 256, 255);

#ifdef FIX_BUGS
			RwIm2DVertexSetScreenZ(&vertexbufferX[0], RwIm2DGetNearScreenZ());
			RwIm2DVertexSetCameraZ(&vertexbufferX[0], RwCameraGetNearClipPlane(Scene.camera));
			RwIm2DVertexSetRecipCameraZ(&vertexbufferX[0], 1.0f/RwCameraGetNearClipPlane(Scene.camera));
			RwIm2DVertexSetScreenZ(&vertexbufferX[1], RwIm2DGetNearScreenZ());
			RwIm2DVertexSetCameraZ(&vertexbufferX[1], RwCameraGetNearClipPlane(Scene.camera));
			RwIm2DVertexSetRecipCameraZ(&vertexbufferX[1], 1.0f/RwCameraGetNearClipPlane(Scene.camera));
#endif

			RwIm2DRenderLine(vertexbufferX, 2, 0, 1);
		}
	}

	RwRenderStateSet(rwRENDERSTATEVERTEXALPHAENABLE, (void*)FALSE);
	RwRenderStateSet(rwRENDERSTATEZTESTENABLE, (void*)TRUE);
	RwRenderStateSet(rwRENDERSTATEZWRITEENABLE, (void*)TRUE);

	POP_RENDERGROUP();
}

void
CCoronas::RenderReflections(void)
{
	int i;
	CColPoint point;
	CEntity *entity;

	if(CWeather::WetRoads > 0.0f){
		PUSH_RENDERGROUP("CCoronas::RenderReflections");

		CSprite::InitSpriteBuffer();

		RwRenderStateSet(rwRENDERSTATEFOGENABLE, (void*)FALSE);
		RwRenderStateSet(rwRENDERSTATEZWRITEENABLE, (void*)FALSE);
		RwRenderStateSet(rwRENDERSTATEZTESTENABLE, (void*)FALSE);
		RwRenderStateSet(rwRENDERSTATEVERTEXALPHAENABLE, (void*)TRUE);
		RwRenderStateSet(rwRENDERSTATESRCBLEND, (void*)rwBLENDONE);
		RwRenderStateSet(rwRENDERSTATEDESTBLEND, (void*)rwBLENDONE);
		RwRenderStateSet(rwRENDERSTATETEXTURERASTER, RwTextureGetRaster(gpCoronaTexture[3]));
#ifdef LIBRW_VISIONOS
		void *vcReflSavedCull = nil;
		RwRenderStateGet(rwRENDERSTATECULLMODE, &vcReflSavedCull);
		if(vcWorldReflectOn()) RwRenderStateSet(rwRENDERSTATECULLMODE, (void*)rwCULLMODECULLNONE);   // world quads have a back
#endif

		for(i = 0; i < NUMCORONAS; i++){
			if(aCoronas[i].id == 0 ||
			   aCoronas[i].fadeAlpha == 0 && aCoronas[i].alpha == 0 ||
			   aCoronas[i].reflection == 0)
				continue;

			// check if we want a reflection on this corona
			if(aCoronas[i].renderReflection){
				if(((CTimer::GetFrameCounter() + i) & 0xF) == 0 &&
				   CWorld::ProcessVerticalLine(aCoronas[i].coors, -1000.0f, point, entity, true, false, false, false, true, false, nil))
					aCoronas[i].heightAboveRoad = aCoronas[i].coors.z - point.point.z;
			}else{
				if(CWorld::ProcessVerticalLine(aCoronas[i].coors, -1000.0f, point, entity, true, false, false, false, true, false, nil)){
					aCoronas[i].heightAboveRoad = aCoronas[i].coors.z - point.point.z;
					aCoronas[i].renderReflection = true;
				}
			}

			// Don't draw if reflection is too high
			if(aCoronas[i].renderReflection && aCoronas[i].heightAboveRoad < 20.0f){
				// don't draw if camera is below road
				if(CCoronas::aCoronas[i].coors.z - aCoronas[i].heightAboveRoad > TheCamera.GetPosition().z)
					continue;

				CVector coors = aCoronas[i].coors;
				coors.z -= 2.0f*aCoronas[i].heightAboveRoad;

				CVector spriteCoors;
				float spritew, spriteh;
				if(CSprite::CalcScreenCoors(coors, &spriteCoors, &spritew, &spriteh, true)) {
					float drawDist = 0.75f * aCoronas[i].drawDist;
					drawDist = Min(drawDist, 55.0f);
					if(spriteCoors.z < drawDist){
						float fadeDistance = drawDist / 2.0f;
						float distanceFade = spriteCoors.z < fadeDistance ? 1.0f : 1.0f - (spriteCoors.z - fadeDistance)/fadeDistance;
						distanceFade = Clamp(distanceFade, 0.0f, 1.0f);
						float recipz = 1.0f/RwCameraGetNearClipPlane(Scene.camera);
						float heightFade = (20.0f - aCoronas[i].heightAboveRoad)/20.0f;
						int intensity = distanceFade*heightFade * 230.0 * CWeather::WetRoads;

#ifdef LIBRW_VISIONOS
						// S1 (wet-road reflections): same upright world billboard as the
						// coronas, at the MIRRORED position below the road; no depth pull
						// (ZTEST is off here, as in the 2D path), no near fade, no roll.
						if(vcWorldReflectOn()){
							vcRenderCoronaWorldQuad(coors, spriteCoors.z, spriteCoors.z, spriteh,
								spritew * aCoronas[i].size * 0.75f, spriteh * aCoronas[i].size * 2.0f,
								(intensity * CCoronas::aCoronas[i].red)>>8,
								(intensity * CCoronas::aCoronas[i].green)>>8,
								(intensity * CCoronas::aCoronas[i].blue)>>8,
								255, 0.0f, 255, false, spriteCoors.x);
						}else
#endif
						CSprite::RenderBufferedOneXLUSprite(
#ifdef FIX_BUGS
							spriteCoors.x, spriteCoors.y, spriteCoors.z,
#else
							spriteCoors.x, spriteCoors.y, RwIm2DGetNearScreenZ(),
#endif
							spritew * aCoronas[i].size * 0.75f,
							spriteh * aCoronas[i].size * 2.0f,
							(intensity * CCoronas::aCoronas[i].red)>>8,
							(intensity * CCoronas::aCoronas[i].green)>>8,
							(intensity * CCoronas::aCoronas[i].blue)>>8,
							255,
							recipz,
							255);
					}
				}
			}
		}
		CSprite::FlushSpriteBuffer();

		RwRenderStateSet(rwRENDERSTATESRCBLEND, (void*)rwBLENDSRCALPHA);
		RwRenderStateSet(rwRENDERSTATEDESTBLEND, (void*)rwBLENDINVSRCALPHA);
		RwRenderStateSet(rwRENDERSTATEVERTEXALPHAENABLE, (void*)FALSE);
		RwRenderStateSet(rwRENDERSTATEZWRITEENABLE, (void*)TRUE);
		RwRenderStateSet(rwRENDERSTATEZTESTENABLE, (void*)TRUE);
#ifdef LIBRW_VISIONOS
		RwRenderStateSet(rwRENDERSTATECULLMODE, vcReflSavedCull);
#endif

		POP_RENDERGROUP();
	}else{
		for(i = 0; i < NUMCORONAS; i++)
			aCoronas[i].renderReflection = false;
	}
}

void
CCoronas::RenderSunReflection(void)
{
	float sunZDir = CTimeCycle::GetSunDirection().z;
	if(sunZDir > -0.05f){
		float intensity = (0.3f - Abs(sunZDir - 0.25f))/0.3f *
			(1.0f - CWeather::CloudCoverage) *
			(1.0f - CWeather::Foggyness) *
			(1.0f - CWeather::Wind);
		if(intensity > 0.0f){
			int r = (CTimeCycle::GetSunCoreRed() + CTimeCycle::GetSunCoronaRed())*intensity*0.25f;
			int g = (CTimeCycle::GetSunCoreGreen() + CTimeCycle::GetSunCoronaGreen())*intensity*0.25f;
			int b = (CTimeCycle::GetSunCoreBlue() + CTimeCycle::GetSunCoronaBlue())*intensity*0.25f;

			CVector sunPos = 40.0f*CTimeCycle::GetSunDirection() + TheCamera.GetPosition();
			sunPos.z = 0.5f*CWeather::Wind + 6.1f;
			CVector sunDir = CTimeCycle::GetSunDirection();
			sunDir.z = 0.0;
			sunDir.Normalise();

			TempBufferIndicesStored = 6;
			TempBufferRenderIndexList[0] = 2;
			TempBufferRenderIndexList[1] = 1;
			TempBufferRenderIndexList[2] = 0;
			TempBufferRenderIndexList[3] = 2;
			TempBufferRenderIndexList[4] = 3;
			TempBufferRenderIndexList[5] = 1;

			// 60 unit square in sun direction
			TempBufferVerticesStored = 4;
			RwIm3DVertexSetRGBA(&TempBufferRenderVertices[0], r, g, b, 255);
			RwIm3DVertexSetPos(&TempBufferRenderVertices[0],
				sunPos.x + 30.0f*sunDir.y,
				sunPos.y - 30.0f*sunDir.x,
				sunPos.z);
			RwIm3DVertexSetRGBA(&TempBufferRenderVertices[1], r, g, b, 255);
			RwIm3DVertexSetPos(&TempBufferRenderVertices[1],
				sunPos.x - 30.0f*sunDir.y,
				sunPos.y + 30.0f*sunDir.x,
				sunPos.z);
			RwIm3DVertexSetRGBA(&TempBufferRenderVertices[2], r, g, b, 255);
			RwIm3DVertexSetPos(&TempBufferRenderVertices[2],
				sunPos.x + 60.0f*sunDir.x + 30.0f*sunDir.y,
				sunPos.y + 60.0f*sunDir.y - 30.0f*sunDir.x,
				sunPos.z);
			RwIm3DVertexSetRGBA(&TempBufferRenderVertices[3], r, g, b, 255);
			RwIm3DVertexSetPos(&TempBufferRenderVertices[3],
				sunPos.x + 60.0f*sunDir.x - 30.0f*sunDir.y,
				sunPos.y + 60.0f*sunDir.y + 30.0f*sunDir.x,
				sunPos.z);

			RwIm3DVertexSetU(&TempBufferRenderVertices[0], 0.0f);
			RwIm3DVertexSetV(&TempBufferRenderVertices[0], 1.0f);
			RwIm3DVertexSetU(&TempBufferRenderVertices[1], 1.0f);
			RwIm3DVertexSetV(&TempBufferRenderVertices[1], 1.0f);
			RwIm3DVertexSetU(&TempBufferRenderVertices[2], 0.0f);
			RwIm3DVertexSetV(&TempBufferRenderVertices[2], 0.5f);
			RwIm3DVertexSetU(&TempBufferRenderVertices[3], 1.0f);
			RwIm3DVertexSetV(&TempBufferRenderVertices[3], 0.5f);

			int timeInc = 0;
			int sideInc = 0;
			int fwdInc = 0;
			for(int i = 0; i < 20; i++){
				TempBufferRenderIndexList[TempBufferIndicesStored + 0] = TempBufferVerticesStored;
				TempBufferRenderIndexList[TempBufferIndicesStored + 1] = TempBufferVerticesStored-1;
				TempBufferRenderIndexList[TempBufferIndicesStored + 2] = TempBufferVerticesStored-2;
				TempBufferRenderIndexList[TempBufferIndicesStored + 3] = TempBufferVerticesStored;
				TempBufferRenderIndexList[TempBufferIndicesStored + 4] = TempBufferVerticesStored+1;
				TempBufferRenderIndexList[TempBufferIndicesStored + 5] = TempBufferVerticesStored-1;
				TempBufferIndicesStored += 6;

				// What a weird way to do it...
				float fwdLen = fwdInc/20 + 60;
				float sideLen = sideInc/20 + 30;
				sideLen += 10.0f*Sin((float)(CTimer::GetTimeInMilliseconds()+timeInc & 0x7FF)/0x800*TWOPI);
				timeInc += 900;
				sideInc += 970;
				fwdInc += 1440;

				RwIm3DVertexSetRGBA(&TempBufferRenderVertices[TempBufferVerticesStored+0], r, g, b, 255);
				RwIm3DVertexSetPos(&TempBufferRenderVertices[TempBufferVerticesStored+0],
					sunPos.x + fwdLen*sunDir.x + sideLen*sunDir.y,
					sunPos.y + fwdLen*sunDir.y - sideLen*sunDir.x,
					sunPos.z);

				RwIm3DVertexSetRGBA(&TempBufferRenderVertices[TempBufferVerticesStored+1], r, g, b, 255);
				RwIm3DVertexSetPos(&TempBufferRenderVertices[TempBufferVerticesStored+1],
					sunPos.x + fwdLen*sunDir.x - sideLen*sunDir.y,
					sunPos.y + fwdLen*sunDir.y + sideLen*sunDir.x,
					sunPos.z);

				RwIm3DVertexSetU(&TempBufferRenderVertices[TempBufferVerticesStored+0], 0.0f);
				RwIm3DVertexSetV(&TempBufferRenderVertices[TempBufferVerticesStored+0], 0.5f);
				RwIm3DVertexSetU(&TempBufferRenderVertices[TempBufferVerticesStored+1], 1.0f);
				RwIm3DVertexSetV(&TempBufferRenderVertices[TempBufferVerticesStored+1], 0.5f);
				TempBufferVerticesStored += 2;
			}


			RwRenderStateSet(rwRENDERSTATEZWRITEENABLE, (void*)FALSE);
			RwRenderStateSet(rwRENDERSTATEZTESTENABLE, (void*)TRUE);
			RwRenderStateSet(rwRENDERSTATEFOGENABLE, (void*)FALSE);
			RwRenderStateSet(rwRENDERSTATEFOGTYPE, (void*)rwFOGTYPELINEAR);
			RwRenderStateSet(rwRENDERSTATESRCBLEND, (void*)rwBLENDONE);
			RwRenderStateSet(rwRENDERSTATEDESTBLEND, (void*)rwBLENDONE);
			RwRenderStateSet(rwRENDERSTATEVERTEXALPHAENABLE, (void*)TRUE);
			RwRenderStateSet(rwRENDERSTATETEXTURERASTER, RwTextureGetRaster(gpCoronaTexture[4]));
			if(RwIm3DTransform(TempBufferRenderVertices, TempBufferVerticesStored, nil, rwIM3D_VERTEXUV)){
				RwIm3DRenderIndexedPrimitive(rwPRIMTYPETRILIST, TempBufferRenderIndexList, TempBufferIndicesStored);
				RwIm3DEnd();
			}
			RwRenderStateSet(rwRENDERSTATEZWRITEENABLE, (void*)TRUE);
			RwRenderStateSet(rwRENDERSTATEZTESTENABLE, (void*)TRUE);
			RwRenderStateSet(rwRENDERSTATESRCBLEND, (void*)rwBLENDSRCALPHA);
			RwRenderStateSet(rwRENDERSTATEDESTBLEND, (void*)rwBLENDINVSRCALPHA);
			RwRenderStateSet(rwRENDERSTATEFOGENABLE, (void*)FALSE);
			RwRenderStateSet(rwRENDERSTATEVERTEXALPHAENABLE, (void*)FALSE);
			TempBufferVerticesStored = 0;
			TempBufferIndicesStored = 0;
		}
	}
}

void 
CCoronas::DoSunAndMoon(void)
{
	// yeah, moon is done somewhere else....

	CVector sunCoors = CTimeCycle::GetSunDirection();
	sunCoors *= 150.0f;
	sunCoors += TheCamera.GetPosition();

	if(CTimeCycle::GetSunDirection().z > -0.2f){
		float size = ((CGeneral::GetRandomNumber()&0xFF) * 0.005f + 10.0f) * CTimeCycle::GetSunSize();
		RegisterCorona(SUN_CORE,
			CTimeCycle::GetSunCoreRed(), CTimeCycle::GetSunCoreGreen(), CTimeCycle::GetSunCoreBlue(),
			255, sunCoors, size,
			999999.88f, TYPE_STAR, FLARE_NONE, REFLECTION_OFF, LOSCHECK_OFF, STREAK_OFF, 0.0f);

		if(CTimeCycle::GetSunDirection().z > 0.0f && !CGame::IsInInterior())
			RegisterCorona(SUN_CORONA,
				CTimeCycle::GetSunCoronaRed(), CTimeCycle::GetSunCoronaGreen(), CTimeCycle::GetSunCoronaBlue(),
				255, sunCoors, 25.0f * CTimeCycle::GetSunSize(),
				999999.88f, TYPE_STAR, FLARE_SUN, REFLECTION_OFF, LOSCHECK_ON, STREAK_OFF, 0.0f);
	}

	CVector spriteCoors;
	float spritew, spriteh;
	if(CSprite::CalcScreenCoors(sunCoors, &spriteCoors, &spritew, &spriteh, true)) {
		SunScreenX = spriteCoors.x;
		SunScreenY = spriteCoors.y;
	}else{
		SunScreenX = 1000000.0f;
		SunScreenY = 1000000.0f;
	}
}

void
CRegisteredCorona::Update(void)
{
	if(!registeredThisFrame)
		alpha = 0;

	if(LOScheck &&
	   (CCoronas::SunBlockedByClouds && id == CCoronas::SUN_CORONA ||
	    !CWorld::GetIsLineOfSightClear(coors, TheCamera.GetPosition(), true, false, false, false, false, false))){
		// Corona is blocked, fade out
		fadeAlpha = Max(fadeAlpha - 15.0f*CTimer::GetTimeStep(), 0.0f);
	}else if(offScreen){
		// Same when off screen
		fadeAlpha = Max(fadeAlpha - 15.0f*CTimer::GetTimeStep(), 0.0f);
	}else{
		// Visible
		if(alpha > fadeAlpha){
			// fade in
			fadeAlpha = Min(fadeAlpha + 15.0f*CTimer::GetTimeStep(), alpha);
			if(CCoronas::bChangeBrightnessImmediately)
				fadeAlpha = alpha;
		}else if(alpha < fadeAlpha){
			// too visible, decrease alpha but not below alpha
			fadeAlpha = Max(fadeAlpha - 15.0f*CTimer::GetTimeStep(), alpha);
		}

		// darken scene when the sun is visible
		if(id == CCoronas::SUN_CORONA)
			CCoronas::LightsMult = Max(CCoronas::LightsMult - CTimer::GetTimeStep()*0.06f, 0.6f);
	}

	// remove if invisible
	if(fadeAlpha == 0 && !firstUpdate)
		id = 0;
	firstUpdate = false;
	registeredThisFrame = false;
}

void
CEntity::ProcessLightsForEntity(void)
{
	int i, n;
	C2dEffect *effect;
	CVector pos;
	bool lightOn, lightFlickering;
	uint32 flashTimer1, flashTimer2, flashTimer3;

	if(bRenderDamaged || !bIsVisible || GetUp().z < 0.96f)
		return;

	flashTimer1 = 0;
	flashTimer2 = 0;
	flashTimer3 = 0;

	n = CModelInfo::GetModelInfo(GetModelIndex())->GetNum2dEffects();
	for(i = 0; i < n; i++, flashTimer1 += 0x80, flashTimer2 += 0x100, flashTimer3 += 0x200){
		effect = CModelInfo::GetModelInfo(GetModelIndex())->Get2dEffect(i);

		switch(effect->type){
		case EFFECT_LIGHT:
			pos = GetMatrix() * effect->pos;

			lightOn = false;
			lightFlickering = false;
			switch(effect->light.lightType){
			case LIGHT_ON:
				lightOn = true;
				break;
			case LIGHT_ON_NIGHT:
				if(CClock::GetHours() > 18 || CClock::GetHours() < 7)
					lightOn = true;
				break;
			case LIGHT_FLICKER:
				if((CTimer::GetTimeInMilliseconds() ^ m_randomSeed) & 0x60)
					lightOn = true;
				else
					lightFlickering = true;
				if((CTimer::GetTimeInMilliseconds()>>11 ^ m_randomSeed) & 3)
					lightOn = true;
				break;
			case LIGHT_FLICKER_NIGHT:
				if(CClock::GetHours() > 18 || CClock::GetHours() < 7 || CWeather::WetRoads > 0.5f){
					if((CTimer::GetTimeInMilliseconds() ^ m_randomSeed) & 0x60)
						lightOn = true;
					else
						lightFlickering = true;
					if((CTimer::GetTimeInMilliseconds()>>11 ^ m_randomSeed) & 3)
						lightOn = true;
				}
				break;
			case LIGHT_FLASH1:
				if((CTimer::GetTimeInMilliseconds() + flashTimer1) & 0x200)
					lightOn = true;
				break;
			case LIGHT_FLASH1_NIGHT:
				if(CClock::GetHours() > 18 || CClock::GetHours() < 7)
					if((CTimer::GetTimeInMilliseconds() + flashTimer1) & 0x200)
						lightOn = true;
				break;
			case LIGHT_FLASH2:
				if((CTimer::GetTimeInMilliseconds() + flashTimer2) & 0x400)
					lightOn = true;
				break;
			case LIGHT_FLASH2_NIGHT:
				if(CClock::GetHours() > 18 || CClock::GetHours() < 7)
					if((CTimer::GetTimeInMilliseconds() + flashTimer2) & 0x400)
						lightOn = true;
				break;
			case LIGHT_FLASH3:
				if((CTimer::GetTimeInMilliseconds() + flashTimer3) & 0x800)
					lightOn = true;
				break;
			case LIGHT_FLASH3_NIGHT:
				if(CClock::GetHours() > 18 || CClock::GetHours() < 7)
					if((CTimer::GetTimeInMilliseconds() + flashTimer3) & 0x800)
						lightOn = true;
				break;
			case LIGHT_RANDOM_FLICKER:
				if(m_randomSeed > 16)
					lightOn = true;
				else{
					if((CTimer::GetTimeInMilliseconds() ^ m_randomSeed*8) & 0x60)
						lightOn = true;
					else
						lightFlickering = true;
					if((CTimer::GetTimeInMilliseconds()>>11 ^ m_randomSeed*8) & 3)
						lightOn = true;
				}
				break;
			case LIGHT_RANDOM_FLICKER_NIGHT:
				if(CClock::GetHours() > 18 || CClock::GetHours() < 7){
					if(m_randomSeed > 16)
						lightOn = true;
					else{
						if((CTimer::GetTimeInMilliseconds() ^ m_randomSeed*8) & 0x60)
							lightOn = true;
						else
							lightFlickering = true;
						if((CTimer::GetTimeInMilliseconds()>>11 ^ m_randomSeed*8) & 3)
							lightOn = true;
					}
				}
				break;
			case LIGHT_BRIDGE_FLASH1:
				if(CBridge::ShouldLightsBeFlashing() && CTimer::GetTimeInMilliseconds() & 0x200)
					lightOn = true;
				break;
			case LIGHT_BRIDGE_FLASH2:
				if(CBridge::ShouldLightsBeFlashing() && (CTimer::GetTimeInMilliseconds() & 0x1FF) < 60)
					lightOn = true;
				break;
			}

			if(effect->light.flags & LIGHTFLAG_HIDE_OBJECT){
				if(lightOn)
					bDoNotRender = false;
				else
					bDoNotRender = true;
				return;
			}

			// Corona
			if(lightOn)
				CCoronas::RegisterCorona((uintptr)this + i,
					effect->col.r, effect->col.g, effect->col.b, 255,
					pos, effect->light.size, effect->light.dist,
					effect->light.corona, effect->light.flareType, effect->light.roadReflection,
					effect->light.flags&LIGHTFLAG_LOSCHECK, CCoronas::STREAK_OFF, 0.0f,
					!!(effect->light.flags&LIGHTFLAG_LONG_DIST));
			else if(lightFlickering)
				CCoronas::RegisterCorona((uintptr)this + i,
					0, 0, 0, 255,
					pos, effect->light.size, effect->light.dist,
					effect->light.corona, effect->light.flareType, effect->light.roadReflection,
					effect->light.flags&LIGHTFLAG_LOSCHECK, CCoronas::STREAK_OFF, 0.0f,
					!!(effect->light.flags&LIGHTFLAG_LONG_DIST));

			// Pointlight
			bool alreadyProcessedFog;
			alreadyProcessedFog = false;
			if(effect->light.range != 0.0f && lightOn){
				if(effect->col.r == 0 && effect->col.g == 0 && effect->col.b == 0){
					CPointLights::AddLight(CPointLights::LIGHT_POINT,
						pos, CVector(0.0f, 0.0f, 0.0f),
						effect->light.range,
						0.0f, 0.0f, 0.0f,
						CPointLights::FOG_NONE, true);
				}else{
					CPointLights::AddLight(CPointLights::LIGHT_POINT,
						pos, CVector(0.0f, 0.0f, 0.0f),
						effect->light.range,
						effect->col.r*CTimeCycle::GetSpriteBrightness()/255.0f,
						effect->col.g*CTimeCycle::GetSpriteBrightness()/255.0f,
						effect->col.b*CTimeCycle::GetSpriteBrightness()/255.0f,
						(effect->light.flags & LIGHTFLAG_FOG) >> 1,
						true);
					alreadyProcessedFog = true; 
				}
			}

			if(!alreadyProcessedFog){
				if(effect->light.flags & LIGHTFLAG_FOG_ALWAYS){
					CPointLights::AddLight(CPointLights::LIGHT_FOGONLY_ALWAYS,
						pos, CVector(0.0f, 0.0f, 0.0f),
						0.0f,
						effect->col.r/255.0f, effect->col.g/255.0f, effect->col.b/255.0f,
						CPointLights::FOG_ALWAYS, true);
				}else if(effect->light.flags & LIGHTFLAG_FOG_NORMAL && lightOn && effect->light.range == 0.0f){
					CPointLights::AddLight(CPointLights::LIGHT_FOGONLY,
						pos, CVector(0.0f, 0.0f, 0.0f),
						0.0f,
						effect->col.r/255.0f, effect->col.g/255.0f, effect->col.b/255.0f,
						CPointLights::FOG_NORMAL, true);
				}
			}

			// Light shadow
			if(effect->light.shadowSize != 0.0f){
				if(lightOn){
					CShadows::StoreStaticShadow((uintptr)this + i, SHADOWTYPE_ADDITIVE,
						effect->light.shadow, &pos,
						effect->light.shadowSize, 0.0f,
						0.0f, -effect->light.shadowSize,
						128,
						effect->col.r*CTimeCycle::GetSpriteBrightness()*effect->light.shadowIntensity/255.0f,
						effect->col.g*CTimeCycle::GetSpriteBrightness()*effect->light.shadowIntensity/255.0f,
						effect->col.b*CTimeCycle::GetSpriteBrightness()*effect->light.shadowIntensity/255.0f,
						15.0f, 1.0f, 40.0f, false, 0.0f);
				}else if(lightFlickering){
					CShadows::StoreStaticShadow((uintptr)this + i, SHADOWTYPE_ADDITIVE,
						effect->light.shadow, &pos,
						effect->light.shadowSize, 0.0f,
						0.0f, -effect->light.shadowSize,
						0, 0.0f, 0.0f, 0.0f,
						15.0f, 1.0f, 40.0f, false, 0.0f);
				}
			}
			break;

		case EFFECT_SUNGLARE:
			if(CWeather::SunGlare >= 0.0f){
				CVector pos = GetMatrix() * effect->pos;
				CVector glareDir = pos - GetPosition();
				glareDir.Normalise();
				CVector camDir = TheCamera.GetPosition() - pos;
				float dist = camDir.Magnitude();
				camDir *= 2.0f/dist;
				glareDir += camDir;
				glareDir.Normalise();
				float camAngle = -DotProduct(glareDir, CTimeCycle::GetSunDirection());
				if(camAngle > 0.0f){
					float intens = Sqrt(camAngle) * CWeather::SunGlare;
					pos += camDir;
					CCoronas::RegisterCorona((uintptr)this + 33 + i,
						intens * (CTimeCycle::GetSunCoreRed() + 2*255)/3.0f,
						intens * (CTimeCycle::GetSunCoreGreen() + 2*255)/3.0f,
						intens * (CTimeCycle::GetSunCoreBlue() + 2*255)/3.0f,
						255,
						pos, 0.5f*CWeather::SunGlare*Sqrt(dist), 120.0f,
						CCoronas::TYPE_STAR, CCoronas::FLARE_NONE,
						CCoronas::REFLECTION_OFF, CCoronas::LOSCHECK_OFF,
						CCoronas::STREAK_OFF, 0.0f);
				}
			}
			break;
		}
	}
}
