#pragma once
// visionOS: world-space upright billboards for the game's CPU-projected sprite families
// (multiview-plan.md S1/S2/S6). The stock game draws coronas, particles and sky sprites as
// SCREEN-aligned quads; in the headset those roll with the head and change shape off-axis.
// Here a sprite is a quad at its true world position whose horizontal axis is the true
// horizon (world up x line of sight), drawn through RwIm3D so the multiview twin puts it
// in both views with the right per-eye matrices. Batched per texture; the caller flushes
// wherever it would flush CSprite's buffer (texture or blend change, end of pass).
//
// Sizes are world HALF sizes in metres. The game's sprite "size" factors (szx*55,
// m_fSize*w, ...) are "pixels per metre" (focal/z) times metres, so a world quad takes the
// metre part directly and divides out the per-metre factor (see callers).
#ifdef LIBRW_VISIONOS

class CVector;

// Upright billboard axes at pos: ax = world-horizontal screen-right, ay = screen-down,
// sign-aligned to the view axes so texture orientation matches the 2D sprite. Falls back to
// the view axes when looking (almost) straight up/down. Returns false if degenerate.
bool vcWsAxes(const CVector &pos, CVector &ax, CVector &ay);

// Queue one sprite. zview = view depth (for the batch far pull), rotation as the 2D path,
// colours c1/c2 blended per corner exactly like RenderBufferedOneXLUSprite_Rotate_2Colours
// (cx/cy = gradient direction; pass the same colour twice for a flat colour).
void vcWsAdd(const CVector &pos, float zview, float halfW, float halfH, float rotation,
	int r1, int g1, int b1, int r2, int g2, int b2, float cx, float cy, int a);

// Draw and clear the batch (fog off, cull none around the draw; sprites beyond 0.9*far are
// pulled inside with the per-eye im3d homothety -- screen position/size unchanged).
void vcWsFlush(void);

// Angle of the sun as seen from pos, measured in the billboard plane like the 2D path
// measures it on screen (x right, y down).
float vcWsSunAngle(const CVector &pos);

#endif
