#include "common.h"
#ifdef LIBRW_VISIONOS

#include "Camera.h"
#include "Draw.h"
#include "Timecycle.h"
#include "General.h"
#include "WorldSprite.h"

// See WorldSprite.h. Moved out of Clouds.cpp (S6) when the particles (S2) needed the same
// batch; the corona path (Coronas.cpp) keeps its own single-quad helper with the per-quad
// depth pull and probe.
extern "C" void vc_im3d_pull(float delta);   // librw gl3immed: per-eye view-space homothety

#define VC_WS_MAX 256   // quads per draw (particles: up to 750 live, flushed per texture)
static RwIm3DVertex vcWsVerts[VC_WS_MAX*4];
static RwImVertexIndex vcWsIdx[VC_WS_MAX*6];
static int vcWsCount = 0;
static float vcWsMaxZ = 0.0f;

bool
vcWsAxes(const CVector &pos, CVector &ax, CVector &ay)
{
	const CMatrix &V = TheCamera.m_viewMatrix;
	CVector vx(V.GetRight().x, V.GetForward().x, V.GetUp().x);   // view +x in world
	CVector vy(V.GetRight().y, V.GetForward().y, V.GetUp().y);   // view +y in world (screen down)
	CVector fwd = pos - TheCamera.GetPosition();
	float fl = fwd.Magnitude(); if(fl < 1e-4f) return false;
	fwd *= 1.0f / fl;
	ax = CrossProduct(fwd, CVector(0.0f, 0.0f, 1.0f));
	if(ax.Magnitude() > 0.05f){
		ax.Normalise();
		ay = CrossProduct(ax, fwd);
		ay.Normalise();
		if(DotProduct(ax, vx) < 0.0f) ax = -ax;
		if(DotProduct(ay, vy) < 0.0f) ay = -ay;
	}else{
		ax = vx; ay = vy;
		float lx = ax.Magnitude(), ly = ay.Magnitude();
		if(lx < 1e-4f || ly < 1e-4f) return false;
		ax *= 1.0f / lx; ay -= ax * DotProduct(ax, ay);
		float l = ay.Magnitude(); if(l < 1e-4f) return false;
		ay *= 1.0f / l;
	}
	return true;
}

void
vcWsAdd(const CVector &pos, float zview, float halfW, float halfH, float rotation,
	int r1, int g1, int b1, int r2, int g2, int b2, float cx, float cy, int a)
{
	if(vcWsCount >= VC_WS_MAX) vcWsFlush();
	CVector ax, ay;
	if(!vcWsAxes(pos, ax, ay)) return;
	float c = Cos(rotation), sn = Sin(rotation);
	// corner order/uv of the 2D sprite: 0 (-w,-h) 1 (-w,+h) 2 (+w,+h) 3 (+w,-h)
	float sx[4] = { halfW*(-c-sn), halfW*(-c+sn), halfW*(+c+sn), halfW*(+c-sn) };
	float sy[4] = { halfH*(-c+sn), halfH*(+c+sn), halfH*(+c-sn), halfH*(-c-sn) };
	float cf[4] = { (cx*(-c-sn) + cy*(-c+sn))*0.5f + 0.5f, (cx*(-c+sn) + cy*( c+sn))*0.5f + 0.5f,
	                (cx*( c+sn) + cy*( c-sn))*0.5f + 0.5f, (cx*( c-sn) + cy*(-c-sn))*0.5f + 0.5f };
	static const float us[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
	static const float vs[4] = { 0.0f, 1.0f, 1.0f, 0.0f };
	RwIm3DVertex *v = &vcWsVerts[vcWsCount*4];
	for(int i = 0; i < 4; i++){
		float f = Clamp(cf[i], 0.0f, 1.0f);
		CVector p = pos + ax*sx[i] + ay*sy[i];
		RwIm3DVertexSetPos(&v[i], p.x, p.y, p.z);
		RwIm3DVertexSetRGBA(&v[i], (int)(r1*f + r2*(1.0f-f)), (int)(g1*f + g2*(1.0f-f)), (int)(b1*f + b2*(1.0f-f)), a);
		RwIm3DVertexSetU(&v[i], us[i]);
		RwIm3DVertexSetV(&v[i], vs[i]);
	}
	static const RwImVertexIndex q[6] = { 0, 1, 2, 0, 2, 3 };
	for(int i = 0; i < 6; i++)
		vcWsIdx[vcWsCount*6 + i] = q[i] + vcWsCount*4;
	if(zview > vcWsMaxZ) vcWsMaxZ = zview;
	vcWsCount++;
}

void
vcWsFlush(void)
{
	if(vcWsCount == 0) return;
	// far pull for the whole batch: one homothety about the eye keeps every quad's screen
	// position/size; only sprites beyond the far clip are affected
	float farLimit = 0.9f * CDraw::GetFarClipZ();
	float pull = 0.0f;
	if(vcWsMaxZ > farLimit && vcWsMaxZ > 0.0f) pull = farLimit / vcWsMaxZ - 1.0f;
	if(pull < -0.95f) pull = -0.95f;
	void *savedFog = nil, *savedCull = nil;
	RwRenderStateGet(rwRENDERSTATEFOGENABLE, &savedFog);
	RwRenderStateGet(rwRENDERSTATECULLMODE, &savedCull);
	RwRenderStateSet(rwRENDERSTATEFOGENABLE, (void*)FALSE);
	RwRenderStateSet(rwRENDERSTATECULLMODE, (void*)rwCULLMODECULLNONE);
	vc_im3d_pull(pull);
	if(RwIm3DTransform(vcWsVerts, vcWsCount*4, nil, rwIM3D_VERTEXXYZ|rwIM3D_VERTEXRGBA|rwIM3D_VERTEXUV)){
		RwIm3DRenderIndexedPrimitive(rwPRIMTYPETRILIST, vcWsIdx, vcWsCount*6);
		RwIm3DEnd();
	}
	vc_im3d_pull(0.0f);
	RwRenderStateSet(rwRENDERSTATEFOGENABLE, savedFog);
	RwRenderStateSet(rwRENDERSTATECULLMODE, savedCull);
	vcWsCount = 0;
	vcWsMaxZ = 0.0f;
}

float
vcWsSunAngle(const CVector &pos)
{
	CVector ax, ay;
	if(!vcWsAxes(pos, ax, ay)) return 0.0f;
	CVector sunWorld = TheCamera.GetPosition() + CTimeCycle::GetSunDirection() * 2000.0f;
	CVector d = pos - sunWorld;
	return CGeneral::GetATanOfXY(DotProduct(d, ax), DotProduct(d, ay));
}

#endif
