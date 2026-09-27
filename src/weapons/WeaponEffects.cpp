#include "common.h"

#include "main.h"
#include "WeaponEffects.h"
#ifdef LIBRW_VISIONOS
#include "Timer.h"
extern "C" int vc_get_eye_view_mv(int eye, float m[16]);   // one-pass: per-eye view the GPU twins used
extern "C" int vc_get_eye_proj(float m[16]);
#endif
#include "TxdStore.h"
#include "Sprite.h"
#include "PlayerPed.h"
#include "World.h"
#include "WeaponType.h"

RwTexture *gpCrossHairTex;

CWeaponEffects gCrossHair;

CWeaponEffects::CWeaponEffects()
{
	
}

CWeaponEffects::~CWeaponEffects()
{
	
}

void
CWeaponEffects::Init(void)
{
	gCrossHair.m_bActive = false;
	gCrossHair.m_vecPos = CVector(0.0f, 0.0f, 0.0f);
	gCrossHair.m_nRed = 255;
	gCrossHair.m_nGreen = 0;
	gCrossHair.m_nBlue = 0;
	gCrossHair.m_nAlpha = 127;
	gCrossHair.m_fSize = 1.0f;
	gCrossHair.m_fRotation = 0.0f;
	
	
	CTxdStore::PushCurrentTxd();
	int32 slot = CTxdStore::FindTxdSlot("particle");
	CTxdStore::SetCurrentTxd(slot);
	
	gpCrossHairTex    = RwTextureRead("target256", "target256m");
	
	CTxdStore::PopCurrentTxd();
}

void
CWeaponEffects::Shutdown(void)
{
	RwTextureDestroy(gpCrossHairTex);
	gpCrossHairTex = nil;
}

void
CWeaponEffects::MarkTarget(CVector pos, uint8 red, uint8 green, uint8 blue, uint8 alpha, float size)
{
	gCrossHair.m_bActive = true;
	gCrossHair.m_vecPos = pos;
	gCrossHair.m_fSize = size;
}

void
CWeaponEffects::ClearCrossHair(void)
{
	gCrossHair.m_bActive = false;
}

void
CWeaponEffects::Render(void)
{
	static float aCrossHairSize[WEAPONTYPE_TOTALWEAPONS] =
	{
		1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
		0.4f, 0.4f,
		0.5f,
		0.3f,
		0.9f, 0.9f, 0.9f,
		0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
		0.1f, 0.1f,
		1.0f,
		0.6f,
		0.7f,
		0.0f, 0.0f
	};



	if ( gCrossHair.m_bActive )
	{
		float size = aCrossHairSize[FindPlayerPed()->GetWeapon()->m_eWeaponType];
		
		RwRenderStateSet(rwRENDERSTATEZWRITEENABLE,      (void *)FALSE);
		RwRenderStateSet(rwRENDERSTATEZTESTENABLE,       (void *)FALSE);
		RwRenderStateSet(rwRENDERSTATEVERTEXALPHAENABLE, (void *)TRUE);
		RwRenderStateSet(rwRENDERSTATESRCBLEND,          (void *)rwBLENDSRCALPHA);
#ifdef FIX_BUGS
		RwRenderStateSet(rwRENDERSTATEDESTBLEND,         (void *)rwBLENDINVSRCALPHA);
#else
		RwRenderStateSet(rwRENDERSTATEDESTBLEND,         (void *)rwBLENDINVDESTALPHA);
#endif
		RwRenderStateSet(rwRENDERSTATETEXTURERASTER,     (void *)RwTextureGetRaster(gpCrossHairTex));

		RwV3d pos;
		float w, h;
		if ( CSprite::CalcScreenCoors(gCrossHair.m_vecPos, &pos, &w, &h, true) )
		{
			PUSH_RENDERGROUP("CWeaponEffects::Render");
#ifdef LIBRW_VISIONOS
			// VC_XHAIR_DIAG=1 (multiview-plan.md 5.1/5.2): in the one-pass render this
			// sprite is projected ONCE with the centre camera and drawn at the same
			// position in both views (= optical infinity). Log, once per second, where
			// each eye's own view would put it -- the disparity the sprite lacks, in
			// slice pixels and degrees. 5.2 (world quad, per-view projection in the
			// shader) must reproduce exactly these per-eye positions.
			{
				static int diagOn = -1;
				if(diagOn < 0){ const char *e = getenv("VC_XHAIR_DIAG"); diagOn = (e && e[0] == '1') ? 1 : 0; }
				static float lastT = 0.0f;
				float nowT = CTimer::GetTimeInMilliseconds() * 0.001f;
				if(diagOn && nowT - lastT >= 1.0f){
					lastT = nowT;
					float v0[16], v1[16], p[16];
					if(vc_get_eye_view_mv(0, v0) && vc_get_eye_view_mv(1, v1) && vc_get_eye_proj(p)){
						auto screenX = [&](const float *v) -> float {
							const CVector &q = gCrossHair.m_vecPos;
							float lx = v[0]*q.x + v[4]*q.y + v[8]*q.z  + v[12];
							float ly = v[1]*q.x + v[5]*q.y + v[9]*q.z  + v[13];
							float lz = v[2]*q.x + v[6]*q.y + v[10]*q.z + v[14];
							float cx = p[0]*lx + p[4]*ly + p[8]*lz + p[12];
							float cw = p[3]*lx + p[7]*ly + p[11]*lz + p[15];
							return (cw > 0.0001f) ? (cx/cw * 0.5f + 0.5f) * SCREEN_WIDTH : -1.0f;
						};
						float x0 = screenX(v0), x1 = screenX(v1);
						float dispPx = x0 - x1;
						float degPerPx = (p[0] > 0.0001f) ? (2.0f * Atan(1.0f / p[0]) * 180.0f / PI) / SCREEN_WIDTH : 0.0f;
						printf("[vc-xhair] dist=%.1fm drawn x=%.0f (both eyes) | per-eye would be x0=%.0f x1=%.0f -> missing disparity %.0f px = %.2f deg (sprite w=%.0f px)\n",
						       pos.z, pos.x, x0, x1, dispPx, dispPx * degPerPx, w);
					}
				}
			}
#endif

			float recipz = 1.0f / pos.z;
			CSprite::RenderOneXLUSprite_Rotate_Aspect(pos.x, pos.y, pos.z,
				w, h,
				255, 88, 100, 158,
				recipz, gCrossHair.m_fRotation, gCrossHair.m_nAlpha);
				
			float recipz2 = 1.0f / pos.z;
			
			CSprite::RenderOneXLUSprite_Rotate_Aspect(pos.x, pos.y, pos.z,
				size*w, size*h,
				107, 134, 247, 158,
				recipz2, TWOPI - gCrossHair.m_fRotation, gCrossHair.m_nAlpha);
						
			gCrossHair.m_fRotation += 0.02f;
			if ( gCrossHair.m_fRotation > TWOPI )
				gCrossHair.m_fRotation = 0.0;

			POP_RENDERGROUP();
		}
			
		RwRenderStateSet(rwRENDERSTATEVERTEXALPHAENABLE, (void *)FALSE);
		RwRenderStateSet(rwRENDERSTATEZWRITEENABLE,      (void *)TRUE);
		RwRenderStateSet(rwRENDERSTATEZTESTENABLE,       (void *)TRUE);
	}
}
