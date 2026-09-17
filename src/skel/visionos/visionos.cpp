// visionos.cpp
//
// Third skeleton implementation for reVC, alongside src/skel/glfw/glfw.cpp and
// src/skel/win/win.cpp. It fulfils the platform contract in src/skel/platform.h
// (plus the extra globals/helpers the core expects) for arm64-apple-xros.
//
// This is a STUB layer: it is meant to LINK and not immediately abort, not to
// run. There is no window, no monitors, no cursor and no SwapBuffers on
// visionOS; the GL context comes from ANGLE and the result is blitted to a
// Metal texture outside reVC. The real ANGLE/CompositorServices and file-system
// wiring arrive in the next steps. Every stub is marked // TODO(visionos):.
//
// Structure follows glfw.cpp, but none of the GLFW logic is carried over.
// The whole file is guarded by LIBRW_VISIONOS (mirroring how glfw.cpp guards
// itself) so it contributes no symbols to the GLFW/Win builds.

#ifdef LIBRW_VISIONOS

#include <mach/mach_time.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>   // getenv
#include <string.h>   // strncmp
#include <strings.h>  // strcasecmp
#include <unistd.h>   // getcwd, chdir
#include <dirent.h>   // opendir, readdir
#include <sys/stat.h> // mkdir, stat
#include <locale.h>   // setlocale
#include <pthread.h>  // game thread
#include <atomic>     // stop flag
#include <time.h>     // clock_gettime (input debug cadence)

#include "common.h"
#include "rwcore.h"
#include "skeleton.h"
#include "platform.h"
#include "crossplatform.h"
#include "FileMgr.h"
#include "Frontend.h"
#include "Text.h"
#include "Game.h"
#include "PCSave.h"
#include "main.h"     // LoadingScreen, InitialiseGame
#include "Timer.h"    // CTimer
#include "Pad.h"      // CPad, CControllerState (gamepad input)
#include "DMAudio.h"  // DMAudio (music mode on restart/load)

// Game render-target size. THE single source: feeds the two vcrt MTLTextures,
// rsCAMERASIZE, RsGlobal and the librw open params. This is the OFFSCREEN size
// the game renders into for display on the world-anchored canvas -- a design
// choice (16:9), deliberately independent of the physical drawable dimensions;
// the canvas is a quad, not the drawable.
// TODO(visionos): later derive dynamically from the quad geometry and the
// angular resolution instead of a fixed 1920x1080.
#define VISIONOS_SCREEN_WIDTH  1920
#define VISIONOS_SCREEN_HEIGHT 1080

// Render resolution PER EYE, runtime-selectable so it can be swept for the quality/cost
// tradeoff (and M2 vs M5) without a rebuild. Drives the eye-slice + cinema texture size
// (vcrt_create), the GL viewport (vc_screen_size), reVC's RsGlobal and the camera. The
// stereo WORLD projection is the compositor's per-eye computeProjection (independent of
// this), so raising it only adds pixel density -- no world distortion; but the 2D/HUD/menu
// layout IS in these screen coords, so a non-16:9 step repositions the HUD. Native drawable
// is 2048x1984 (~1:1). VC_RES=0..3 steps, or explicit VC_RES_W / VC_RES_H.
static int g_vcScrW = 0, g_vcScrH = 0;
static void vcScreenInit(void)
{
	if (g_vcScrW) return;
	int w = VISIONOS_SCREEN_WIDTH, h = VISIONOS_SCREEN_HEIGHT;
	const char *rw = getenv("VC_RES_W"), *rh = getenv("VC_RES_H");
	if (rw && rh && atoi(rw) > 0 && atoi(rh) > 0) { w = atoi(rw); h = atoi(rh); }
	else switch (getenv("VC_RES") ? atoi(getenv("VC_RES")) : 4) {   // default 4 (native drawable, 1:1 centre)
		// Ladder to plot the perf curve vs area (megapixels/eye) and bytes:
		case 0:  w = 1920; h = 1080; break;   // 2.07 MP
		case 1:  w = 2200; h = 2100; break;   // 4.62 MP
		case 2:  w = 2450; h = 2350; break;   // 5.76 MP
		case 3:  w = 2600; h = 2500; break;   // 6.50 MP
		case 4:  w = 2720; h = 2624; break;   // 7.14 MP = the drawable texture at maxRenderQuality=1.0
		                                      //   (measured 2720x2624) -> 1:1 in the centre, no upscale.
		default: w = 2720; h = 2624; break;   // unknown value -> best (step 4)
	}
	if (w < 640) w = 640;  if (w > 4096) w = 4096;
	if (h < 480) h = 480;  if (h > 4096) h = 4096;
	g_vcScrW = w; g_vcScrH = h;
	printf("[vc-res] render resolution per eye = %dx%d\n", w, h);
}
static int vcScreenW(void) { vcScreenInit(); return g_vcScrW; }
static int vcScreenH(void) { vcScreenInit(); return g_vcScrH; }

// Single source for the render-target size, queried by librw (gl3device) over
// the C seam -- same pattern as vc_external_framebuffer().
extern "C" void vc_screen_size(int *w, int *h)
{
	if (w) *w = vcScreenW();
	if (h) *h = vcScreenH();
}

// VC_DOUBLE_RENDER cost probe (read from main.cpp's RenderScene hook):
//   0 = off
//   1 = render scene twice sharing depth (LEQUAL: visible frags re-shade)
//   2 = clear depth between passes (full 2x fragment work)
//   3 = 2nd pass with GL_LESS (all equal-depth frags fail -> vertex/draw-call only)
extern "C" int vc_double_render_mode(void)
{
	static int mode = -1;
	if (mode < 0) {
		const char *v = getenv("VC_DOUBLE_RENDER");
		mode = (v && v[0] >= '1' && v[0] <= '3') ? (v[0] - '0') : 0;
		printf("[vc-dr] VC_DOUBLE_RENDER mode = %d (%s)\n", mode,
		       mode == 0 ? "off" :
		       mode == 1 ? "double render, shared depth (LEQUAL)" :
		       mode == 2 ? "double render, depth cleared between" :
		                   "double render, GL_LESS (2nd pass frags rejected)");
	}
	return mode;
}

// Render mode seam. Mirror of vc_render_mode_t in AvpViceCity/VCPlatform.h --
// values MUST match. Read once from VC_RENDER_MODE: default STEREO; only
// VC_RENDER_MODE=cinema selects the single flat cinema screen (comparison/cutscene
// fallback). Kept in a mutable global (no compile-time bake-in) so a later runtime
// switch stays possible.
enum { VC_MODE_CINEMA = 0, VC_MODE_STEREO = 1 };
static int g_renderMode = -1;   // -1 = not yet resolved

extern "C" int vc_render_mode(void)
{
	if (g_renderMode < 0) {
		const char *v = getenv("VC_RENDER_MODE");
		// Default is now STEREO; cinema only on explicit request (kept as the
		// comparison/cutscene fallback). VC_RENDER_MODE=cinema -> cinema.
		int requested = (v && strcasecmp(v, "cinema") == 0) ? VC_MODE_CINEMA : VC_MODE_STEREO;
		if (requested == VC_MODE_STEREO) {
			printf("[vc-mode] VC_RENDER_MODE = stereo (default; set VC_RENDER_MODE=cinema for mono)\n");
			// Phase 5.5: stereo now renders two eye passes into a 2D-array texture
			// (proof via read-back; publish-naht + Swift are the next step). The
			// cinema back buffer is still what gets published, so the display stays
			// mono for now -- the stereo slices are validated off-screen.
			g_renderMode = VC_MODE_STEREO;
		} else {
			g_renderMode = VC_MODE_CINEMA;
		}
		printf("[vc-mode] active render mode = %s\n",
		       g_renderMode == VC_MODE_CINEMA ? "cinema" : "stereo");
	}
	return g_renderMode;
}

// VC_HUD_ASPECT (default ON, stereo only): make the 2D layout scale (SCREEN_SCALE_AR in
// common.h) use the 4:3 DESIGN aspect instead of the physical buffer aspect. Needed because
// the near-square buffer from VC_RES=2 up (aspect < 4:3) makes the stock widescreen factor
// magnify x until the 640-wide menu/HUD design runs off the right edge. The host draws the
// overlay quad at 4:3 to undo the resulting squeeze, so proportions stay correct at full
// resolution. =0 restores the stock (cut-off) behaviour for A/B. Cinema keeps stock.
extern "C" int vc_hud_aspect_fixed(void)
{
	static int e = -1;
	if (e < 0) {
		const char *s = getenv("VC_HUD_ASPECT");
		e = (s && s[0] == '0') ? 0 : 1;
		printf("[vc-hud-aspect] 2D layout aspect = %s (VC_HUD_ASPECT=%s)\n",
		       e ? "4:3 design (fixed)" : "buffer aspect (stock)", s ? s : "unset");
	}
	return (e && vc_render_mode() == VC_MODE_STEREO) ? 1 : 0;
}

// Map legend geometry (PrintMap in Frontend.cpp + CRadar::DrawLegend in Radar.cpp). The
// stock legend covers a large part of the map; in VR that is worse than on a monitor.
// ONE source for both knobs so the text (Frontend) and the blip ICONS (Radar) can never
// drift apart -- a half-scaled legend (small text, full-size symbols) would look worse
// than the stock one.
//   VC_MAP_LEGEND       design units the box is moved UP          (default 100, 0 = stock)
//   VC_MAP_LEGEND_SCALE size factor about (design-centre x, box-top y) (default 0.6)
// Shift default history: 90 (pure move) -> 40 when the scaling landed (the map's top edge is
// at design y 63 = centre 225 - size 162, so ~37 puts the box flush with it) -> 100 after the
// device test, i.e. the box top sits at design y 0. That is only usable because the legend is
// now drawn AFTER the menu border polygons (CMenuManager::PrintMapLegend): at 40 it kept the
// map's upper band, at 100 it sits in the frame band above the map and is out of the way.
// Anchoring x at the design centre keeps the box CENTRED at any scale (the stock box spans
// design x 95..555, so a left-edge anchor pushed it visibly off-centre); anchoring y at the
// box top keeps the two knobs ORTHOGONAL: the shift moves the top edge without changing the
// size, the scale changes the size without moving that edge.
static float g_mapLegendShift = -1.0f;
static float g_mapLegendScale = -1.0f;
static void vcMapLegendInit(void)
{
	if (g_mapLegendShift >= 0.0f) return;
	const char *sh = getenv("VC_MAP_LEGEND");
	const char *sc = getenv("VC_MAP_LEGEND_SCALE");
	g_mapLegendShift = sh ? (float)atof(sh) : 100.0f;
	if (g_mapLegendShift < 0.0f) g_mapLegendShift = 0.0f;
	g_mapLegendScale = sc ? (float)atof(sc) : 0.6f;
	// Clamped: below ~0.2 the font is unreadable, above 1.0 it would grow past the map.
	if (g_mapLegendScale < 0.2f) g_mapLegendScale = 0.2f;
	if (g_mapLegendScale > 1.0f) g_mapLegendScale = 1.0f;
	printf("[vc-map] legend shift=%.0f design units up, scale=%.2f (VC_MAP_LEGEND=%s VC_MAP_LEGEND_SCALE=%s)\n",
	       g_mapLegendShift, g_mapLegendScale, sh ? sh : "unset", sc ? sc : "unset");
}
extern "C" float vc_map_legend_shift(void) { vcMapLegendInit(); return g_mapLegendShift; }
extern "C" float vc_map_legend_scale(void) { vcMapLegendInit(); return g_mapLegendScale; }


// --- Camera matrix override seam (stereo injection point) -------------------
// gl3device beginUpdate consumes these (getters below) after computing its own
// view/proj; when active it uploads ours instead. Buffered under a lock: the
// setters will later be driven from the compositor (main thread) while the
// getters run on the game thread. 16 floats each, column-major, librw convention.
static pthread_mutex_t g_mtxMutex = PTHREAD_MUTEX_INITIALIZER;
static float g_ovView[16];
static float g_ovProj[16];
static int   g_ovActive = 0;
// Diagnostic: host time (mach_absolute_time) when the head pose was last PUSHED
// (Swift render thread, predicted for this frame's presentationTime). The game
// thread reads it in vc_get_view_matrix to measure how OLD the pose is when reVC
// actually renders with it -- the decoupling latency behind the head-turn "aura".
static uint64_t g_ovSetTime = 0;

extern "C" void vc_set_view_matrix(const float m[16])
{
	if (!m) return;
	uint64_t now = mach_absolute_time();
	pthread_mutex_lock(&g_mtxMutex); memcpy(g_ovView, m, 16 * sizeof(float)); g_ovSetTime = now; pthread_mutex_unlock(&g_mtxMutex);
}
// Host reads this right after pushing to key its DeviceAnchor ring to the exact
// g_ovSetTime that will arrive back on a buffer as pose_set_time (reprojection fix).
extern "C" uint64_t vc_last_pushed_pose_time(void)
{
	pthread_mutex_lock(&g_mtxMutex); uint64_t t = g_ovSetTime; pthread_mutex_unlock(&g_mtxMutex);
	return t;
}
extern "C" void vc_set_projection_matrix(const float m[16])
{
	if (!m) return;
	pthread_mutex_lock(&g_mtxMutex); memcpy(g_ovProj, m, 16 * sizeof(float)); pthread_mutex_unlock(&g_mtxMutex);
}
extern "C" void vc_set_matrix_override(int active)
{
	pthread_mutex_lock(&g_mtxMutex); g_ovActive = active ? 1 : 0; pthread_mutex_unlock(&g_mtxMutex);
}
// Consumed by gl3device (librw). Not in VCPlatform.h -- reVC-internal.
extern "C" int vc_matrix_override_active(void)
{
	pthread_mutex_lock(&g_mtxMutex); int a = g_ovActive; pthread_mutex_unlock(&g_mtxMutex); return a;
}
// Push time of the head pose reVC LAST consumed (game-thread only: set here, read
// in vcrt_publish_frame). Lets the publish tag the buffer so vc_acquire_ready_frame
// can report the full push -> acquire (~display) latency of that slice.
static uint64_t g_lastConsumedPoseSetTime = 0;
extern "C" uint64_t vc_last_consumed_pose_time(void) { return g_lastConsumedPoseSetTime; }
extern "C" int vc_perf_log(void);   // defined below; gates verbose perf logs

// [vc-stereo] diagnostic: how old (ms) is the most recently PUSHED head pose right now
// (push->now), and how long (ms) since this was last called (game-thread frame spacing --
// balloons if a synchronous interior load blocks the game thread). Uses the mach timebase.
extern "C" void vc_stereo_probe(double *poseAgeMs, double *frameDtMs)
{
	static uint64_t sNum = 0, sDen = 0;
	if (sDen == 0) { mach_timebase_info_data_t tb; mach_timebase_info(&tb); sNum = tb.numer; sDen = tb.denom; }
	uint64_t now = mach_absolute_time();
	pthread_mutex_lock(&g_mtxMutex); uint64_t setT = g_ovSetTime; pthread_mutex_unlock(&g_mtxMutex);
	if (poseAgeMs) *poseAgeMs = (setT != 0) ? (double)(now - setT) * (double)sNum / (double)sDen / 1.0e6 : -1.0;
	static uint64_t sLast = 0;
	if (frameDtMs) *frameDtMs = (sLast != 0) ? (double)(now - sLast) * (double)sNum / (double)sDen / 1.0e6 : 0.0;
	sLast = now;
}

extern "C" void vc_get_view_matrix(float m[16])
{
	uint64_t now = mach_absolute_time();
	pthread_mutex_lock(&g_mtxMutex);
	memcpy(m, g_ovView, 16 * sizeof(float));
	uint64_t setT = g_ovSetTime;
	pthread_mutex_unlock(&g_mtxMutex);
	g_lastConsumedPoseSetTime = setT;
	if (!vc_perf_log()) return;
	// VC_PERF_LOG: age of the pose reVC is rendering with (push -> consume leg) and
	// the reVC consume interval (game-thread frame time) + jitter.
	static uint64_t sNum = 0, sDen = 0;
	if (sDen == 0) { mach_timebase_info_data_t tb; mach_timebase_info(&tb); sNum = tb.numer; sDen = tb.denom; }
	static uint64_t sLastConsume = 0;
	double intervalMs = (sLastConsume != 0) ? (double)(now - sLastConsume) * (double)sNum / (double)sDen / 1.0e6 : 0.0;
	sLastConsume = now;
	static int sCtr = 0;
	if (setT != 0 && (sCtr++ % 90) == 0) {
		double ageMs = (double)(now - setT) * (double)sNum / (double)sDen / 1.0e6;
		printf("[vc-pose-age] pose %.2f ms old (push->consume); reVC consume interval %.2f ms (%.1f Hz)\n",
		       ageMs, intervalMs, intervalMs > 0 ? 1000.0 / intervalMs : 0.0);
	}
}
extern "C" void vc_get_projection_matrix(float m[16])
{
	pthread_mutex_lock(&g_mtxMutex); memcpy(m, g_ovProj, 16 * sizeof(float)); pthread_mutex_unlock(&g_mtxMutex);
}

// Frame-phase timing probe (game thread). main.cpp calls vc_frame_mark(id) at the
// phase boundaries; at the final mark we log the per-segment ms (throttled 1/s) so
// we can see where reVC's ~22 ms/frame actually goes. IDs:
//   0 start  6 ConstructRenderList done  7 PreRender done  1 setup done (StartOfFrame)
//   2 eyes done (both eye passes)  3 readback done  4 post-3d done  5 frame done
// Verbose perf logging (env VC_PERF_LOG). The [vc-frame]/[vc-miss] summaries stay
// on always (throttled 10 s -- first line of defence for a future perf problem);
// the finer probes ([vc-pose-age], [vc-pose-latency], [vc-begin], [vc-publish])
// are gated behind this so normal runs are quiet.
extern "C" int vc_perf_log(void)
{
	static int v = -1;
	if (v < 0) v = getenv("VC_PERF_LOG") ? 1 : 0;
	return v;
}

// Accumulate the time spent in RenderEffects across the two eye passes (the cost of
// pulling the world effects into the slices), reported as the [vc-frame] "fx" segment
// -- so we can decide whether 2x RenderEffects fits the budget before committing.
static uint64_t g_fxStart = 0, g_fxAccum = 0;
extern "C" void vc_frame_fx_begin(void) { g_fxStart = mach_absolute_time(); }
extern "C" void vc_frame_fx_end(void)
{
	if (g_fxStart) { g_fxAccum += mach_absolute_time() - g_fxStart; g_fxStart = 0; }
}

// Stereo fade-into-slices guard. DoFade() both MUTATES fade/music state (the
// StillToFadeOut transition) and DRAWS the fullscreen dim rect. In stereo the dim
// belongs IN each eye slice (it dims the world, not the head-locked HUD), so DoFade
// is called once per eye. To keep the state mutation happening exactly once per
// frame, eye 0 runs the full DoFade and eye 1 runs "draw-only": DoFade skips its
// mutation block when this flag is set. Cinema/macOS never set it (flag stays 0).
static int g_fadeDrawOnly = 0;
extern "C" void vc_fade_set_draw_only(int on) { g_fadeDrawOnly = on ? 1 : 0; }
extern "C" int  vc_fade_draw_only(void)       { return g_fadeDrawOnly; }

extern "C" void vc_frame_mark(int id)
{
	if (id < 0 || id >= 10) return;
	static uint64_t t[10] = {0};
	t[id] = mach_absolute_time();
	// Thread CPU time (only advances while this thread actually RUNS). If a frame's
	// wall time (mach_absolute_time) is huge but its CPU time is ~0, the thread was
	// SUSPENDED (OS paused the app -- e.g. a system dialog / head-anchored-content
	// throttle), not doing work. cpuWall gap = external stall, not a reVC/render bug.
	static uint64_t cpu0 = 0;
	if (id == 0) {
		struct timespec cts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cts);
		cpu0 = (uint64_t)cts.tv_sec * 1000000000ull + (uint64_t)cts.tv_nsec;
		g_fxAccum = 0;                       // reset per frame; fx accrues across the two eye passes
		for (int i = 1; i < 10; i++) t[i] = 0;  // marks 2/3/4 are set ONLY in the in-game block; zero
		                                        // every frame so menu/splash/fade/loading frames (which
		                                        // skip that block) are detected as INCOMPLETE below and
		                                        // don't emit a bogus [vc-frame] (stale t[4] -> finish=seconds).
	}
	if (id != 5) return;
	if (vc_render_mode() != 1) return;   // stereo only (cinema doesn't set marks 2/3)
	// Completeness gate (also covers warmup): marks are zeroed at id==0, so this frame
	// must have set ALL of 0..7 to be a real in-game frame. Menu/splash/fade/loading frames
	// skip marks 2/3/4 -> incomplete -> skip (no bogus segment deltas / finish=seconds).
	for (int i = 0; i <= 9; i++) if (t[i] == 0) return;
	static uint64_t sNum = 0, sDen = 0;
	if (sDen == 0) { mach_timebase_info_data_t tb; mach_timebase_info(&tb); sNum = tb.numer; sDen = tb.denom; }
	#define VC_SEG_MS(a,b) ((double)(t[b] - t[a]) * (double)sNum / (double)sDen / 1.0e6)
	double total = VC_SEG_MS(0,5);
	// Worst frame this interval + which phase dominated it: a single 40 ms hitch is
	// invisible in an average but very noticeable, so keep the peak, not just the mean.
	// finish (mark 4->5) split: menus (4->8 RenderMenus), afterfade (8->9 DoFade+
	// Render2dStuffAfterFade), present (9->5 DoRWStuffEndOfFrame/ShowRaster/publish).
	double segMs[7]      = { VC_SEG_MS(0,1), VC_SEG_MS(1,2), VC_SEG_MS(2,3), VC_SEG_MS(3,4),
	                         VC_SEG_MS(4,8), VC_SEG_MS(8,9), VC_SEG_MS(9,5) };
	const char *segNm[7] = { "setup", "eyes", "readback", "post-3d", "menus", "afterfade", "present" };
	// CPU time actually spent this frame (see cpu0 above). Compare to total (wall).
	uint64_t cpuNow; { struct timespec cts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cts);
	                   cpuNow = (uint64_t)cts.tv_sec * 1000000000ull + (uint64_t)cts.tv_nsec; }
	double cpuMs = (cpu0 && cpuNow >= cpu0) ? (double)(cpuNow - cpu0) / 1.0e6 : 0.0;
	static double maxTotal = 0.0, maxSeg = 0.0, maxCpu = 0.0;
	static const char *maxPhase = "-";
	if (total > maxTotal) {
		maxTotal = total; maxSeg = 0.0; maxPhase = "-"; maxCpu = cpuMs;
		for (int i = 0; i < 7; i++) if (segMs[i] > maxSeg) { maxSeg = segMs[i]; maxPhase = segNm[i]; }
	}
	static double lastLog = 0.0;
	double nowS = (double)t[5] * (double)sNum / (double)sDen / 1.0e9;
	if (nowS - lastLog < 10.0) { (void)total; return; }
	lastLog = nowS;
	double fxMs = (double)g_fxAccum * (double)sNum / (double)sDen / 1.0e6;   // 2x RenderEffects
	printf("[vc-frame] last: cnstrList=%.1f prerender=%.1f startframe=%.1f eyes=%.1f (fx=%.1f) post-3d=%.1f menus=%.1f afterfade=%.1f present=%.1f total=%.1f | PEAK total=%.1f ms (%s=%.1f) cpu=%.1f ms [cpu<<total => OS suspended, not work]\n",
	       VC_SEG_MS(0,6), VC_SEG_MS(6,7), VC_SEG_MS(7,1), VC_SEG_MS(1,2), fxMs, VC_SEG_MS(3,4), VC_SEG_MS(4,8), VC_SEG_MS(8,9), VC_SEG_MS(9,5), total, maxTotal, maxPhase, maxSeg, maxCpu);
	maxTotal = 0.0; maxSeg = 0.0; maxPhase = "-"; maxCpu = 0.0;
	#undef VC_SEG_MS
}

// View compose flag: when set, gl3device left-multiplies the override view onto
// the game view (head-pose offset) instead of replacing it. See VCPlatform.h.
static int g_ovCompose = 0;
extern "C" void vc_set_view_compose(int on)
{
	pthread_mutex_lock(&g_mtxMutex); g_ovCompose = on ? 1 : 0; pthread_mutex_unlock(&g_mtxMutex);
}
extern "C" int vc_view_compose_active(void)
{
	pthread_mutex_lock(&g_mtxMutex); int c = g_ovCompose; pthread_mutex_unlock(&g_mtxMutex); return c;
}

// VC_MATRIX_TEST: 0 off, 1 "identity" (feed reVC's own matrices back -> image
// must be unchanged), 2 "shift" (view shifted 0.5 m -> image must move).
extern "C" int vc_matrix_test_mode(void)
{
	static int m = -1;
	if (m < 0) {
		const char *v = getenv("VC_MATRIX_TEST");
		m = (v && strcasecmp(v, "shift") == 0)    ? 2 :
		    (v && strcasecmp(v, "identity") == 0)  ? 1 : 0;
		printf("[vc-mtx] VC_MATRIX_TEST = %s\n", m == 0 ? "off" : m == 1 ? "identity" : "shift");
	}
	return m;
}

// ANGLE bring-up lives in visionos_angle.mm (ObjC++/Metal, kept out of this
// C++ unit to avoid Foundation vs. reVC macro clashes). Returns success and,
// via the out-param, ANGLE's eglGetProcAddress (as void*) for librw's glad.
extern "C" bool  vcgl_init_angle(void **outGetProcAddress);
extern "C" bool  vcgl_make_current_on_this_thread(void); // claim GL ctx on game thread
extern "C" void  vcgl_release_current(void);             // release GL ctx (init thread)
extern "C" void *vcgl_get_proc_address(void);            // ANGLE eglGetProcAddress

// Double-buffered render target + throttle + publish, all in visionos_angle.mm.
extern "C" bool         vcrt_create(int width, int height);
extern "C" void         vcrt_slice_test(void);   // VC_SLICE_TEST diagnostic (visionos_angle.mm)
extern "C" bool         vc_use_external_framebuffer(void);
extern "C" unsigned int vc_external_framebuffer(void);
extern "C" bool         vcrt_begin_frame(void);   // throttle: false = park this frame
extern "C" void         vcrt_publish_frame(void); // publish the just-rendered buffer

// ---------------------------------------------------------------------------
// Globals the core expects but that only ever lived in glfw.cpp / win.cpp.
// ---------------------------------------------------------------------------

long     _dwOperatingSystemVersion;
size_t   _dwMemAvailPhys;
RwUInt32 gGameState;

#ifdef DETECT_JOYSTICK_MENU
char gSelectedJoystickName[128] = "";
#endif

// Backing storage that RsGlobal.ps points at (see PSGLOBAL()).
static psGlobalType PsGlobal;

// ---------------------------------------------------------------------------
// File system (visionOS): the game data lives in the app's Documents folder.
// casepath() resolves relative "models\\coll\\..." against the cwd, and
// CFileMgr::Initialise() freezes the cwd as the data root, so we must chdir()
// into the data root BEFORE CFileMgr::Initialise() runs.
// ---------------------------------------------------------------------------

// The seven top-level folders reVC expects in the data root.
static const char *kExpectedDataDirs[7] = {
	"anim", "audio", "data", "models", "movies", "TEXT", "txd"
};

static char g_dataRoot[1024]  = "";
static char g_userFiles[1024] = "";

// $HOME/Documents in the app sandbox (no ObjC needed on iOS/visionOS).
static bool
vc_documents_path(char *out, size_t outsz)
{
	const char *home = getenv("HOME");
	if (home == nil || home[0] == '\0')
		return false;
	snprintf(out, outsz, "%s/Documents", home);
	return true;
}

static bool
vc_path_exists(const char *path)
{
	struct stat st;
	return stat(path, &st) == 0;
}

// Case-insensitive check whether directory `root` contains an entry `name`.
static bool
vc_dir_has(const char *root, const char *name)
{
	DIR *d = opendir(root);
	if (d == nil)
		return false;
	bool found = false;
	struct dirent *e;
	while ((e = readdir(d)) != nil) {
		if (strcasecmp(e->d_name, name) == 0) { found = true; break; }
	}
	closedir(d);
	return found;
}

// Locate the game-data root under Documents, chdir into it, set up the
// user-files folder path. Logs the chosen root and which folders were found.
// Returns false (with a clear message) if the data is missing.
static bool
vcfs_setup(void)
{
	char documents[1024];
	if (!vc_documents_path(documents, sizeof(documents))) {
		printf("[vc-fs] ERROR: could not determine Documents folder (HOME unset)\n");
		return false;
	}
	printf("[vc-fs] Documents = %s\n", documents);

	// Candidate data roots, in order of preference.
	char cand[3][1024];
	snprintf(cand[0], sizeof(cand[0]), "%s", documents);
	snprintf(cand[1], sizeof(cand[1]), "%s/Game", documents);
	snprintf(cand[2], sizeof(cand[2]), "%s/GTAVC", documents);

	const char *chosen = nil;
	for (int i = 0; i < 3; i++) {
		bool exists    = vc_path_exists(cand[i]);
		bool hasModels = exists && vc_dir_has(cand[i], "models");
		printf("[vc-fs] candidate '%s': exists=%s models=%s\n",
		       cand[i], exists ? "yes" : "no", hasModels ? "yes" : "no");
		if (hasModels && chosen == nil)
			chosen = cand[i];
	}

	if (chosen == nil) {
		printf("[vc-fs] ERROR: game data not found.\n");
		printf("[vc-fs]   Looked in: %s | %s | %s\n", cand[0], cand[1], cand[2]);
		printf("[vc-fs]   Copy the seven data folders (anim, audio, data, models,\n");
		printf("[vc-fs]   movies, TEXT, txd) into: %s\n", documents);
		return false;
	}

	snprintf(g_dataRoot, sizeof(g_dataRoot), "%s", chosen);
	printf("[vc-fs] using data root: %s\n", g_dataRoot);
	for (int i = 0; i < 7; i++) {
		printf("[vc-fs]   %-7s : %s\n", kExpectedDataDirs[i],
		       vc_dir_has(g_dataRoot, kExpectedDataDirs[i]) ? "found" : "MISSING");
	}

	if (chdir(g_dataRoot) != 0) {
		printf("[vc-fs] ERROR: chdir to data root failed\n");
		return false;
	}

	// Settings/saves live in a separate, writable sub-folder in Documents.
	snprintf(g_userFiles, sizeof(g_userFiles), "%s/GTA Vice City User Files", documents);
	return true;
}

// Defined further down; used by psInitialize (same signatures the core expects).
const char *_psGetUserFilesFolder();
void        _psCreateFolder(const char *path);

// ===========================================================================
// Timer  --  REAL implementation (the one exception to "stub everything").
// A game without a clock does nothing; this is trivial and must be correct.
// Returns milliseconds, matching the GLFW/Win psTimer() semantics.
// ===========================================================================
double
psTimer(void)
{
	static mach_timebase_info_data_t timebase;
	if (timebase.denom == 0)
		mach_timebase_info(&timebase);

	uint64_t now = mach_absolute_time();
	double nanos = (double)now * (double)timebase.numer / (double)timebase.denom;
	return nanos / 1.0e6; // -> milliseconds
}

// ===========================================================================
// Lifecycle
// ===========================================================================
RwBool
psInitialize(void)
{
	// Minimal bring-up: give the cross-platform core a valid ps-global so
	// PSGLOBAL() dereferences don't crash, and set the values other classes
	// read early.
	PsGlobal.lastMousePos.x = PsGlobal.lastMousePos.y = 0.0f;
	RsGlobal.ps = &PsGlobal;

	PsGlobal.fullScreen       = FALSE;
	PsGlobal.cursorIsInWindow = FALSE;
	PsGlobal.joy1id           = -1;
	PsGlobal.joy2id           = -1;
	PsGlobal.window           = nil; // TODO(visionos): ANGLE-Kontext-Handle hinterlegen

	gGameState = GS_START_UP;

	// glfw/win set this to "fool other classes"; keep the same lie.
	_dwOperatingSystemVersion = OS_WINXP;

	// TODO(visionos): echten freien Speicher ermitteln (sysctl HW_MEMSIZE / mach).
	// Fester, plausibler Wert statt 0 - Streaming.cpp rechnet (_dwMemAvailPhys - 10*MB)/2,
	// bei 0 gaebe der unsigned-Underflow ein absurdes Streaming-Budget.
	_dwMemAvailPhys = (size_t)4 * 1024 * 1024 * 1024; // 4 GiB

	// --- File system: find the game data and make it the working directory ---
	// Must run BEFORE CFileMgr::Initialise() (which captures the cwd as the data
	// root). On missing data: clear message + orderly abort (return FALSE).
	if (!vcfs_setup()) {
		printf("[vc-fs] ERROR: aborting init - game data not available\n");
		return FALSE;
	}

	// With the cwd at the data root, wire up reVC's file/text/settings layer
	// exactly as glfw.cpp's psInitialize does.
	CFileMgr::Initialise();                            // captures cwd = data root
	_psCreateFolder(_psGetUserFilesFolder());          // settings/saves folder
	C_PcSave::SetSaveDirectory(_psGetUserFilesFolder());
	InitialiseLanguage();                              // sets language + loads TEXT
	FrontEndMenuManager.LoadSettings();                // defaults on first run
	TheText.Unload();

	// --- ANGLE bring-up (context only; NOT made current here) ------------
	// Create the EGL display + GLES 3.0 context, but do not make it current on
	// this (init) thread. The game thread is the sole owner: it makes the
	// context current and opens RenderWare (Initialise3D) there. Keeping the
	// context on one thread for its whole life avoids the earlier handoff.
	if (!vcgl_init_angle(nil)) {
		printf("[vc-gl] FAIL: ANGLE/EGL bring-up failed\n");
		return FALSE;
	}
	printf("[vc-gl] ANGLE ready; RenderWare opens on the game thread\n");

	return TRUE;
}

void
psTerminate(void)
{
	// TODO(visionos): nichts zu tun, solange kein Kontext/Fenster existiert.
	return;
}

// ===========================================================================
// Camera / frame presentation
// ===========================================================================
RwBool
psCameraBeginUpdate(RwCamera *camera)
{
	// Throttle at frame start: block until a free back buffer is available, then
	// select it (vc_external_framebuffer() then returns its FBO). On timeout the
	// game thread parks -> returning FALSE makes reVC skip the frame (don't
	// render), which is the intended park state, not an error. This is only ever
	// the main scene camera (shadow/RTT cameras call RwCameraBeginUpdate direct).
	if (!vcrt_begin_frame())
		return FALSE;

	// Real begin-update: binds the camera's raster/FBO via librw's
	// setFrameBuffer, which (for the main camera) redirects the surfaceless
	// default framebuffer to the current back buffer's EGLImage FBO.
	if (!RwCameraBeginUpdate(camera))
		return FALSE;
	return TRUE;
}

void
psCameraShowRaster(RwCamera *camera)
{
	// No SwapBuffers on visionOS. End-of-frame "present": publish the just-
	// rendered back buffer (enqueue the shared-event signal, mark it ready).
	vcrt_publish_frame();
}

RwImage *
psGrabScreen(RwCamera *camera)
{
	// TODO(visionos): Screenshot/Frame-Grab spaeter ueber die Metal-Seite.
	return nil;
}

// ===========================================================================
// Mouse / keyboard input  (entfaellt auf visionOS ersatzlos)
// ===========================================================================
void
psMouseSetPos(RwV2d *pos)
{
	// TODO(visionos): keine Maus/kein Cursor; spaeter ggf. ueber Eingabegeraete.
	return;
}

long
_InputInitialiseMouse(bool exclusive)
{
	// TODO(visionos): keine Maus.
	return 0;
}

void
_InputShutdownMouse()
{
	// TODO(visionos): keine Maus.
}

bool
_InputMouseNeedsExclusive()
{
	// TODO(visionos): keine Maus, nie exklusiv.
	return false;
}

void
_InputTranslateShiftKeyUpDown(RsKeyCodes *rs)
{
	// TODO(visionos): keine Hardware-Tastatur; Eingabe kommt spaeter anders.
	return;
}

// ===========================================================================
// Joystick / gamepad  (spaeter ueber GCController)
// ===========================================================================
void
_InputInitialiseJoys()
{
	// TODO(visionos): Controller-Erkennung ueber GCController; vorerst keiner.
	PsGlobal.joy1id = -1;
	PsGlobal.joy2id = -1;
}

// ===========================================================================
// Gamepad input seam (Swift main thread -> reVC game thread)
// ---------------------------------------------------------------------------
// Mirror of vc_gamepad_t in AvpViceCity/VCPlatform.h -- layout MUST match.
// ===========================================================================
typedef struct {
	float         left_x, left_y;
	float         right_x, right_y;
	float         left_trigger, right_trigger;
	unsigned char south, east, west, north;
	unsigned char dpad_up, dpad_down, dpad_left, dpad_right;
	unsigned char left_shoulder, right_shoulder;
	unsigned char left_thumb, right_thumb;
	unsigned char menu, options;
} vc_gamepad_t;

static vc_gamepad_t    g_pad = {};
static pthread_mutex_t g_padMutex = PTHREAD_MUTEX_INITIALIZER;

// Called from the Swift main thread. Buffers the snapshot under a lock so it
// never overlaps the game thread's read in CapturePad.
extern "C" void
vc_set_gamepad_state(const vc_gamepad_t *state)
{
	if (state == nil)
		return;
	pthread_mutex_lock(&g_padMutex);
	g_pad = *state;
	pthread_mutex_unlock(&g_padMutex);
}

// Sticks: reVC uses up = negative and (matching the glfw path) an ~8-bit range
// with a 0.3 deadzone. GCController gives up = +1, so Y is negated by the caller.
static int16
vcpad_axis(float v)
{
	float a = v < 0.0f ? -v : v;
	return a > 0.3f ? (int16)(v * 128.0f) : 0;
}

static double
vcpad_now_seconds(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

// Log on CHANGE only (one line per press/release, by name), plus -- behind
// VC_DEBUG_INPUT=1 -- the axes once per second. Game thread only.
static void
vcpad_log(const vc_gamepad_t &s)
{
	static bool         init = false;
	static bool         dbgAxes = false;
	static vc_gamepad_t prev;
	if (!init) {
		init = true;
		memset(&prev, 0, sizeof(prev));
		dbgAxes = (getenv("VC_DEBUG_INPUT") != NULL);
		printf("[vc-input] gamepad wired; VC_DEBUG_INPUT %s\n", dbgAxes ? "ENABLED" : "disabled");
	}

#define VCPAD_BTN(field, name) \
	if (dbgAxes && s.field != prev.field) printf("[vc-input] %s %s\n", name, s.field ? "DOWN" : "UP");
	VCPAD_BTN(south,         "Cross(A)")
	VCPAD_BTN(east,          "Circle(B)")
	VCPAD_BTN(west,          "Square(X)")
	VCPAD_BTN(north,         "Triangle(Y)")
	VCPAD_BTN(dpad_up,       "DPadUp")
	VCPAD_BTN(dpad_down,     "DPadDown")
	VCPAD_BTN(dpad_left,     "DPadLeft")
	VCPAD_BTN(dpad_right,    "DPadRight")
	VCPAD_BTN(left_shoulder, "L1")
	VCPAD_BTN(right_shoulder,"R1")
	VCPAD_BTN(left_thumb,    "L3")
	VCPAD_BTN(right_thumb,   "R3")
	VCPAD_BTN(menu,          "Start")
	VCPAD_BTN(options,       "Select")
#undef VCPAD_BTN
	prev = s;

	if (dbgAxes) {
		static double last = 0.0;
		double now = vcpad_now_seconds();
		if (now - last >= 1.0) {
			last = now;
			printf("[vc-input] axes L(%.2f,%.2f) R(%.2f,%.2f) LT=%.2f RT=%.2f\n",
			       s.left_x, s.left_y, s.right_x, s.right_y, s.left_trigger, s.right_trigger);
		}
	}
}

void
CapturePad(RwInt32 padID)
{
	if (padID != 0)
		return;

	// Snapshot the buffered state under the lock, then release before touching
	// reVC state.
	vc_gamepad_t s;
	pthread_mutex_lock(&g_padMutex);
	s = g_pad;
	pthread_mutex_unlock(&g_padMutex);

	// Fill PCTempJoyState (a CControllerState) DIRECTLY. CPad::Update reconciles
	// it with the empty PCTempKeyState into NewState (and rotates OldState itself),
	// so we bypass the controller-config binding layer entirely -- no
	// ControllerConfig / MapIdToButtonId is involved. Buttons are 0/255.
	CPad *pad = CPad::GetPad(0);
	CControllerState &js = pad->PCTempJoyState;

	js.Cross          = s.south          ? 255 : 0;
	js.Circle         = s.east           ? 255 : 0;
	js.Square         = s.west           ? 255 : 0;
	js.Triangle       = s.north          ? 255 : 0;
	js.DPadUp         = s.dpad_up        ? 255 : 0;
	js.DPadDown       = s.dpad_down      ? 255 : 0;
	js.DPadLeft       = s.dpad_left      ? 255 : 0;
	js.DPadRight      = s.dpad_right     ? 255 : 0;
	js.Start          = s.menu           ? 255 : 0;
	js.Select         = s.options        ? 255 : 0;
	js.LeftShoulder1  = s.left_shoulder  ? 255 : 0;
	js.RightShoulder1 = s.right_shoulder ? 255 : 0;
	js.LeftShock      = s.left_thumb     ? 255 : 0;
	js.RightShock     = s.right_thumb    ? 255 : 0;
	js.LeftShoulder2  = (int16)(s.left_trigger  * 255.0f);
	js.RightShoulder2 = (int16)(s.right_trigger * 255.0f);

	js.LeftStickX  = vcpad_axis( s.left_x);
	js.LeftStickY  = vcpad_axis(-s.left_y);   // GCController up = +1 -> reVC up = negative
	js.RightStickX = vcpad_axis( s.right_x);
	js.RightStickY = vcpad_axis(-s.right_y);

	vcpad_log(s);
}

// ===========================================================================
// Device / video mode  (kein Fenster, kein Monitor, genau ein Modus)
// ===========================================================================
RwBool
psSelectDevice()
{
	// TODO(visionos): echtes Device/Subsystem ueber ANGLE waehlen.
	// Vorerst die feste Drawable-Aufloesung der Vision Pro melden, damit
	// SCREEN_WIDTH/HEIGHT & Menue-Layout sinnvolle Werte haben.
	RsGlobal.maximumWidth  = vcScreenW();
	RsGlobal.maximumHeight = vcScreenH();
	RsGlobal.width         = vcScreenW();
	RsGlobal.height        = vcScreenH();

	PsGlobal.fullScreen = TRUE;
	return TRUE;
}

RwBool
_psSetVideoMode(RwInt32 subSystem, RwInt32 videoMode)
{
	// TODO(visionos): hier muss spaeter rsRWINITIALIZE (RwEngineOpen ueber ANGLE)
	// und rsCAMERASIZE ausgeloest werden. Ohne das gibt es keinen RW-Kontext und
	// somit (noch) kein Rendering.
	return TRUE;
}

void
_psSelectScreenVM(RwInt32 videoMode)
{
	// TODO(visionos): Aufloesungswechsel entfaellt (feste Drawable-Groesse).
	return;
}

// The graphics menu prints this string verbatim (Frontend.cpp:1295 AsciiToUnicode of
// _psGetVideoModeList()[m_nDisplayVideoMode]). It used to be STRINGIFIED FROM THE
// COMPILE-TIME DEFINES, so it always claimed 1920x1080 no matter what VC_RES /
// VC_RES_W/H selected -- the render resolution became a runtime value (vcScreenW/H)
// when the resolution switch landed, and this string never followed. Built on first
// request instead: vcScreenInit() runs inside vcScreenW/H, so the numbers are correct
// even if the menu asks before psSelectDevice.
static RwChar  _VMString[40] = { 0 };
static RwChar *_VMList[1]  = { _VMString };

RwChar **
_psGetVideoModeList()
{
	// Still exactly one mode (fixed drawable), but now labelled with the real size.
	if (_VMString[0] == '\0')
		snprintf(_VMString, sizeof(_VMString), "%d X %d X 32", vcScreenW(), vcScreenH());
	return _VMList;
}

RwInt32
_psGetNumVideModes()
{
	// TODO(visionos): genau ein Modus.
	return 1;
}

// ===========================================================================
// Memory / filesystem / textures
// ===========================================================================
RwMemoryFunctions *
psGetMemoryFunctions(void)
{
	// nil ist hier KEIN Fehler: librw (engine.cpp Engine::init) interpretiert nil
	// als "Standard-Allokator verwenden" (defaultMemfuncs = malloc/realloc/free).
	// Entspricht dem Verhalten des macOS/GLFW-Builds ohne USE_CUSTOM_ALLOCATOR.
	return nil;
}

RwBool
psInstallFileSystem(void)
{
	// TODO(visionos): eigenes Dateisystem kommt im naechsten Schritt.
	return TRUE;
}

RwBool
psNativeTextureSupport(void)
{
	return TRUE;
}

// ===========================================================================
// Misc
// ===========================================================================
RwBool
IsForegroundApp()
{
	// TODO(visionos): App-Aktivitaet spaeter ueber den Scene-Phase-Zustand melden.
	return TRUE;
}

void
HandleExit()
{
	// TODO(visionos): Beenden laeuft ueber die App/den ImmersiveSpace, nicht hier.
	return;
}

void
InitialiseLanguage()
{
	// Locale-based language selection, ported from glfw.cpp's non-Windows path
	// (no GLFW dependency). Determines nasty/german/french flags and the menu
	// language, then loads the text.
	setlocale(LC_ALL, "");
	char *systemLang   = setlocale(LC_ALL, NULL);
	char *keyboardLang = setlocale(LC_CTYPE, NULL);

	short primUserLCID, primSystemLCID;
	primUserLCID = primSystemLCID = !strncmp(systemLang, "fr_", 3) ? LANG_FRENCH :
	                                !strncmp(systemLang, "de_", 3) ? LANG_GERMAN :
	                                !strncmp(systemLang, "en_", 3) ? LANG_ENGLISH :
	                                !strncmp(systemLang, "it_", 3) ? LANG_ITALIAN :
	                                !strncmp(systemLang, "es_", 3) ? LANG_SPANISH :
	                                LANG_OTHER;
	short primLayout = !strncmp(keyboardLang, "fr_", 3) ? LANG_FRENCH :
	                   (!strncmp(keyboardLang, "de_", 3) ? LANG_GERMAN : LANG_ENGLISH);

	short subUserLCID, subSystemLCID;
	subUserLCID = subSystemLCID = !strncmp(systemLang, "en_AU", 5) ? SUBLANG_ENGLISH_AUS : SUBLANG_OTHER;
	short subLayout = !strncmp(keyboardLang, "en_AU", 5) ? SUBLANG_ENGLISH_AUS : SUBLANG_OTHER;

	if (primUserLCID == LANG_GERMAN || primSystemLCID == LANG_GERMAN || primLayout == LANG_GERMAN) {
		CGame::nastyGame = false;
		FrontEndMenuManager.m_PrefsAllowNastyGame = false;
		CGame::germanGame = true;
	}
	if (primUserLCID == LANG_FRENCH || primSystemLCID == LANG_FRENCH || primLayout == LANG_FRENCH) {
		CGame::nastyGame = false;
		FrontEndMenuManager.m_PrefsAllowNastyGame = false;
		CGame::frenchGame = true;
	}
	if (subUserLCID == SUBLANG_ENGLISH_AUS || subSystemLCID == SUBLANG_ENGLISH_AUS || subLayout == SUBLANG_ENGLISH_AUS)
		CGame::noProstitutes = true;

#ifdef NASTY_GAME
	CGame::nastyGame = true;
	FrontEndMenuManager.m_PrefsAllowNastyGame = true;
	CGame::noProstitutes = false;
#endif

	int32 lang;
	switch (primSystemLCID) {
		case LANG_GERMAN:  lang = LANG_GERMAN;  break;
		case LANG_FRENCH:  lang = LANG_FRENCH;  break;
		case LANG_SPANISH: lang = LANG_SPANISH; break;
		case LANG_ITALIAN: lang = LANG_ITALIAN; break;
		default: lang = (subSystemLCID == SUBLANG_ENGLISH_AUS) ? -99 : LANG_ENGLISH; break;
	}

	FrontEndMenuManager.OS_Language = primUserLCID;

	switch (lang) {
		case LANG_GERMAN:  FrontEndMenuManager.m_PrefsLanguage = CMenuManager::LANGUAGE_GERMAN;  break;
		case LANG_SPANISH: FrontEndMenuManager.m_PrefsLanguage = CMenuManager::LANGUAGE_SPANISH; break;
		case LANG_FRENCH:  FrontEndMenuManager.m_PrefsLanguage = CMenuManager::LANGUAGE_FRENCH;  break;
		case LANG_ITALIAN: FrontEndMenuManager.m_PrefsLanguage = CMenuManager::LANGUAGE_ITALIAN; break;
		default:           FrontEndMenuManager.m_PrefsLanguage = CMenuManager::LANGUAGE_AMERICAN; break;
	}

	// Needed for strcasecmp to behave the same across locales.
	setlocale(LC_CTYPE, "C");
	setlocale(LC_COLLATE, "C");
	setlocale(LC_NUMERIC, "C");

	TheText.Unload();
	TheText.Load();
}

// ===========================================================================
// File-system helpers (nicht in platform.h, aber vom Kern forward-declared:
// core/FileMgr.cpp, save/PCSave.cpp, save/MemoryCard.cpp).
// ===========================================================================
const char *
_psGetUserFilesFolder()
{
	// Absolute path under Documents, computed by vcfs_setup(). Empty (valid
	// C-string) until then so strcpy/strcat never see nil.
	return g_userFiles;
}

void
_psCreateFolder(const char *path)
{
	if (path == nil || path[0] == '\0')
		return;

	// Normalise Windows-style backslashes, then create the (single-level) dir.
	char tmp[1024];
	snprintf(tmp, sizeof(tmp), "%s", path);
	for (char *c = tmp; *c; c++)
		if (*c == '\\') *c = '/';

	if (!vc_path_exists(tmp))
		mkdir(tmp, 0755);
}

// ===========================================================================
// Frame publish
// ---------------------------------------------------------------------------
// The double-buffer state machine, the throttle (vcrt_begin_frame), the
// shared-event signal and the publish (vcrt_publish_frame) all live on the
// ANGLE side in visionos_angle.mm, next to the MTLTexture/EGLImage/FBO buffers
// they operate on. Here we only expose the stop flag they poll and forward the
// depth-renderbuffer hook librw calls when it creates the shared Z buffer.
// ===========================================================================

// Attach librw's shared depth renderbuffer to BOTH back-buffer FBOs (once, at
// FBO creation). Defined on the ANGLE side; called from gl3device.cpp.
extern "C" void vc_attach_depth_renderbuffer(unsigned int rbo);

// ===========================================================================
// Game loop  (ported from glfw.cpp main(); runs on its OWN thread)
// ===========================================================================

// These were file-scope statics in glfw.cpp; keep local equivalents here.
static RwBool ForegroundApp   = TRUE;  // TODO(visionos): kein Fokus-Modell; immer Vordergrund
static RwBool WindowIconified = FALSE; // TODO(visionos): kein Fenster; nie minimiert
static RwBool RwInitialised   = TRUE;

static std::atomic<bool> g_stop{false};

// Polled by the ANGLE-side throttle (vcrt_begin_frame) so a parked game thread
// wakes and unwinds cleanly when shutdown is requested.
extern "C" bool vc_should_stop(void) { return g_stop.load(); }
static pthread_t         g_gameThread;
static bool              g_gameThreadRunning = false;

static const char *
gGameStateName(RwUInt32 s)
{
	switch (s) {
		case GS_START_UP:          return "GS_START_UP";
		case GS_INIT_LOGO_MPEG:    return "GS_INIT_LOGO_MPEG";
		case GS_LOGO_MPEG:         return "GS_LOGO_MPEG";
		case GS_INIT_INTRO_MPEG:   return "GS_INIT_INTRO_MPEG";
		case GS_INTRO_MPEG:        return "GS_INTRO_MPEG";
		case GS_INIT_ONCE:         return "GS_INIT_ONCE";
		case GS_INIT_FRONTEND:     return "GS_INIT_FRONTEND";
		case GS_FRONTEND:          return "GS_FRONTEND";
		case GS_INIT_PLAYING_GAME: return "GS_INIT_PLAYING_GAME";
		case GS_PLAYING_GAME:      return "GS_PLAYING_GAME";
		default:                   return "GS_?";
	}
}

static void
run_game_loop(void)
{
	// Claim the GL context on this (game) thread. All reVC rendering happens here.
	if (!vcgl_make_current_on_this_thread()) {
		printf("[vc-loop] FAIL: could not make GL context current on game thread\n");
		return;
	}

	// Resolve + log the render mode once at startup. Cinema for now (stereo is a
	// stub that falls back). Future mode-dependent setup would branch here.
	vc_render_mode();

	// Create the render target (MTLTexture on ANGLE's device -> EGLImage -> GL
	// FBO). Must happen before any rendering and before Initialise3D so the
	// external-framebuffer redirect is armed when the first camera pass runs.
	// TODO(visionos): resolution hardcoded; later from the CompositorServices drawable.
	if (!vcrt_create(vcScreenW(), vcScreenH())) {
		printf("[vc-loop] FAIL: could not create render target\n");
		return;
	}

	// VC_SLICE_TEST: one-shot probe of EGLImage wrapping of Metal 2D-array slices
	// (RGBA8/RGBA16Float). Runs here (GL context current, ANGLE resolved), then
	// the normal path continues unchanged.
	if (getenv("VC_SLICE_TEST"))
		vcrt_slice_test();

	// Once-before-RW init (Stufe 1): CGame::InitialiseOnceBeforeRW() runs
	// CdStreamInit(MAX_CDCHANNELS). On the glfw path this fires from the
	// rsINITIALIZE handler BEFORE rsRWINITIALIZE; we call it directly here (the
	// RsInitialize half of rsINITIALIZE is done manually elsewhere in visionos).
	//
	// CWD must be exactly the data root: CdStreamInit does
	// statvfs("models/gta3.img") with a RAW relative path (no casepath), and
	// CFileMgr::Initialise() re-captures the cwd as the data root. The cwd was
	// last set to the data root by chdir(g_dataRoot) in vcfs_setup() (this file,
	// ~line 174) during psInitialize -- but InitialiseLanguage()/LoadSettings()
	// there use CFileMgr::SetDir() and can leave it in a subdir, so we don't rely
	// on it: re-assert it and log before/after as proof.
	{
		char cwdBefore[1024] = {0};
		(void)getcwd(cwdBefore, sizeof(cwdBefore));
		printf("[vc-loop] before InitialiseOnceBeforeRW: cwd='%s'\n", cwdBefore);

		if (chdir(g_dataRoot) != 0)
			printf("[vc-loop] WARN: chdir(data root '%s') failed before CdStreamInit\n", g_dataRoot);

		extern int32 gNumChannels;   // defined in CdStream_posix.cpp
		CGame::InitialiseOnceBeforeRW();

		char cwdAfter[1024] = {0};
		(void)getcwd(cwdAfter, sizeof(cwdAfter));
		printf("[vc-loop] after InitialiseOnceBeforeRW: cwd='%s' gNumChannels=%d\n",
		       cwdAfter, (int)gNumChannels);
	}

	// Open RenderWare on THIS thread via the event handler (Initialise3D is
	// static in main.cpp; reVC's rsRWINITIALIZE case maps to Initialise3D(param)
	// = RsRwInitialize (engine open via ANGLE) + CGame::InitialiseRenderWare()
	// which creates Scene.camera). The param must point to a rw::EngineOpenParams
	// whose .window carries ANGLE's eglGetProcAddress (librw loads glad from it).
	static rw::EngineOpenParams openParams;
	openParams.width       = vcScreenW();
	openParams.height      = vcScreenH();
	openParams.windowtitle = "reVC";
	openParams.window      = vcgl_get_proc_address();
	printf("[vc-loop] rsRWINITIALIZE (Initialise3D: RwEngineOpen/Start + InitialiseRenderWare) ...\n");
	if (RsEventHandler(rsRWINITIALIZE, &openParams) == rsEVENTERROR) {
		printf("[vc-loop] FAIL: rsRWINITIALIZE\n");
		return;
	}
	printf("[vc-loop] rsRWINITIALIZE OK\n");

	// Size the main camera so it gets its CAMERA + Z rasters (CameraSize with a
	// non-nil RwRect). Without this Scene.camera->frameBuffer stays nil and im2d
	// crashes. // TODO(visionos): rect from the drawable later.
	RwRect camRect;
	camRect.x = 0; camRect.y = 0;
	camRect.w = vcScreenW();
	camRect.h = vcScreenH();
	RsEventHandler(rsCAMERASIZE, &camRect);
	printf("[vc-loop] rsCAMERASIZE %dx%d done\n", vcScreenW(), vcScreenH());

	printf("[vc-loop] game thread started\n");

	// Outer restart loop, mirroring glfw.cpp main().
	while (!g_stop.load()) {
		RwInitialised = TRUE;

		// Initial mouse position (no-op on visionOS). // TODO(visionos): keine Maus
		RwV2d pos;
		pos.x = RsGlobal.maximumWidth * 0.5f;
		pos.y = RsGlobal.maximumHeight * 0.5f;
		RsMouseSetPos(&pos);

		RwUInt32 lastState = 0xFFFFFFFFu;

		// Inner state-machine loop. glfwWindowShouldClose() -> our stop flag.
		while (!g_stop.load() && !RsGlobal.quit && !FrontEndMenuManager.m_bWantToRestart) {
			// TODO(visionos): glfwPollEvents() entfaellt; Eingaben spaeter via GCController.

			if (gGameState != lastState) {
				printf("[vc-loop] gGameState = %s\n", gGameStateName(gGameState));
				lastState = gGameState;
			}

			if (ForegroundApp) {  // always TRUE on visionOS
				switch (gGameState) {
					case GS_START_UP:
						gGameState = GS_INIT_ONCE; // TODO(visionos): keine Movies -> direkt weiter
						break;

					case GS_INIT_ONCE:
						LoadingScreen(nil, nil, "loadsc0");
						if (!CGame::InitialiseOnceAfterRW())
							RsGlobal.quit = TRUE;
						gGameState = GS_INIT_FRONTEND;
						break;

					case GS_INIT_FRONTEND:
						LoadingScreen(nil, nil, "loadsc0");
						FrontEndMenuManager.m_bGameNotLoaded = true;
						FrontEndMenuManager.m_bStartUpFrontEndRequested = true;
						gGameState = GS_FRONTEND;
						break;

					case GS_FRONTEND:
						if (!WindowIconified)
							RsEventHandler(rsFRONTENDIDLE, nil);
						if (!FrontEndMenuManager.m_bMenuActive || FrontEndMenuManager.m_bWantToLoad)
							gGameState = GS_INIT_PLAYING_GAME;
						if (FrontEndMenuManager.m_bWantToLoad) {
							InitialiseGame();
							FrontEndMenuManager.m_bGameNotLoaded = false;
							gGameState = GS_PLAYING_GAME;
						}
						break;

					case GS_INIT_PLAYING_GAME:
						InitialiseGame();
						FrontEndMenuManager.m_bGameNotLoaded = false;
						gGameState = GS_PLAYING_GAME;
						break;

					case GS_PLAYING_GAME: {
						float ms = (float)CTimer::GetCurrentTimeInCycles() /
						           (float)CTimer::GetCyclesPerMillisecond();
						if (RwInitialised) {
							if (!FrontEndMenuManager.m_PrefsFrameLimiter ||
							    (1000.0f / (float)RsGlobal.maxFPS) < ms)
								RsEventHandler(rsIDLE, (void *)TRUE);
						}
						break;
					}

					default:
						// Movie states are skipped on visionOS.
						gGameState = GS_INIT_ONCE;
						break;
				}
			}
		}

		RwInitialised = FALSE;
		FrontEndMenuManager.UnloadTextures();

		// Quit (RsGlobal.quit) or hard stop -> leave the outer loop; the app/immersive
		// space teardown happens Swift-side (see vc_wants_quit / vc_game_thread_stop).
		if (!FrontEndMenuManager.m_bWantToRestart)
			break;

		// Restart path, mirroring glfw.cpp's non-PS2 cleanup. Loading a save sets BOTH
		// m_bWantToRestart AND m_bWantToLoad (DoSettingsBeforeStartingAGame); a plain
		// restart (e.g. quit-to-menu / new game) sets only m_bWantToRestart. The old
		// stub dropped all of this, so the load never actually happened -- the world
		// stayed frozen with m_bWantToLoad stuck true.
		CPad::ResetCheats();
		CPad::StopPadsShaking();

		DMAudio.ChangeMusicMode(MUSICMODE_DISABLE);

		CTimer::Stop();

		if (FrontEndMenuManager.m_bWantToLoad) {
			CGame::ShutDownForRestart();
			CGame::InitialiseWhenRestarting();
			DMAudio.ChangeMusicMode(MUSICMODE_GAME);
			LoadSplash(GetLevelSplashScreen(CGame::currLevel));
			FrontEndMenuManager.m_bWantToLoad = false;
		} else {
			if (gGameState == GS_PLAYING_GAME)
				CGame::ShutDown();

			CTimer::Stop();

			if (FrontEndMenuManager.m_bFirstTime == true)
				gGameState = GS_INIT_FRONTEND;
			else
				gGameState = GS_INIT_PLAYING_GAME;
		}

		FrontEndMenuManager.m_bFirstTime = false;
		FrontEndMenuManager.m_bWantToRestart = false;
	}

	printf("[vc-loop] game loop exited (quit=%d stop=%d)\n",
	       (int)RsGlobal.quit, (int)g_stop.load());
	vcgl_release_current();
}

static void *
game_thread_main(void *arg)
{
	(void)arg;
	pthread_setname_np("reVC game");
	run_game_loop();
	return nil;
}

// --- C interface for the Swift side ----------------------------------------
// Start AFTER psInitialize() returned TRUE. Stop on teardown.
extern "C" void
vc_game_thread_start(void)
{
	if (g_gameThreadRunning) {
		printf("[vc-loop] game thread already running\n");
		return;
	}
	g_stop.store(false);
	if (pthread_create(&g_gameThread, nil, game_thread_main, nil) == 0) {
		g_gameThreadRunning = true;
		printf("[vc-loop] game thread created\n");
	} else {
		printf("[vc-loop] FAIL: pthread_create\n");
	}
}

extern "C" void
vc_game_thread_stop(void)
{
	if (!g_gameThreadRunning)
		return;
	g_stop.store(true);
	RsGlobal.quit = TRUE;
	pthread_join(g_gameThread, nil);
	g_gameThreadRunning = false;
	printf("[vc-loop] game thread stopped\n");
}

#endif // LIBRW_VISIONOS
