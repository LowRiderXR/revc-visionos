// Stutter diagnostics recorder (visionOS, Diagnostics build configuration only).
//
// Everything in this header exists only when VC_DIAGNOSTICS is defined by the build
// (Xcode configuration "Diagnostics"; Release and the macOS reference build never set it,
// so no probe code, not even a flag test, reaches those binaries). Hook sites in the engine
// and in librw wrap their calls in #if VC_DIAGNOSTICS and use the VC_DIAG_* macros below.
//
// Threading contract (the reason this module exists at all):
//   * game thread, render thread, CD reader thread: integer counters and fixed structs
//     only -- no allocation, no lock, no string, no printf, no file I/O.
//   * once per second the game thread copies a summary into a fixed ring slot;
//   * a utility-QoS dispatch timer drains the rings, formats and appends to
//     Documents/Diagnostics/stutter.log (rotated at ~2 MB, one older file kept).
// Hooks are no-ops until vc_diag_init() has read VC_DIAG_RECORD=1 (set by the launcher
// toggle "Record stutter diagnostics"); then each hook costs an atomic add or a few
// integer ops. The marker string "[vc-diag]" (header line, console log) is the
// release-bundle cross-check: `strings` on a Release binary must not contain it.
#ifndef VC_DIAG_H
#define VC_DIAG_H

#if VC_DIAGNOSTICS

#include <stdint.h>
#include <atomic>
#include <mach/mach_time.h>   // mach_absolute_time for the hook-site timers

// Monotonic counters (sums; the per-second line reports the delta).
enum {
	VC_DIAG_C_STREAM_REQ = 0,   // CStreaming::RequestModel: a model entered the request queue
	VC_DIAG_C_STREAM_LOADED,    // ConvertBufferToObject / FinishLoadingLargeFile marked a model loaded
	VC_DIAG_C_CD_READS,         // CdStreamThread read() calls (reader thread)
	VC_DIAG_C_CD_BYTES,         // bytes read by those calls
	VC_DIAG_C_BLOCKING_LOADS,   // CStreaming::LoadAllRequestedModels calls (synchronous, blocks the game thread)
	VC_DIAG_C_TEX_UPLOADS,      // librw rasterUnlock texel uploads (glTexImage2D / glCompressedTexImage2D)
	VC_DIAG_C_TEX_BYTES,        // bytes handed to those uploads
	VC_DIAG_C_GL_LINKS,         // librw linkprogram: glLinkProgram + the GL_LINK_STATUS query that resolves it
	VC_DIAG_C_PUBLISHES,        // vcrt_publish_frame: every game-side frame handed to the compositor (world, menu, splash)
	VC_DIAG_C_DROPS,            // vc_acquire_ready_frame: a READY frame recycled unshown (newest: superseded; fifo: age/overflow/drain)
	VC_DIAG_C_WAIT,             // fifo: acquires after which a frame was still waiting (one slot of extra latency)
	VC_DIAG_C_DRAIN,            // fifo: drops by the drain rule (the waiting frame given up to cut the latency)
	VC_DIAG_C_N
};

// Per-window maxima in mach ticks (reset by the per-second tick).
enum {
	VC_DIAG_M_CD_READ = 0,      // longest single read()
	VC_DIAG_M_BLOCKING_LOAD,    // longest LoadAllRequestedModels
	VC_DIAG_M_TEX_UPLOAD,       // longest single texel upload
	VC_DIAG_M_GL_LINK,          // longest link (glLinkProgram + GL_LINK_STATUS)
	VC_DIAG_M_N
};

// Event kinds (E lines). a/b/c/d are kind-specific integers.
// VC_DIAG_EV_LAYER_PAUSED / _RESUMED are mirrored for Swift in AvpViceCity/VCDiagnostics.h --
// keep the two in sync.
enum {
	VC_DIAG_EV_RATE = 1,        // raw display rate changed by > 3 Hz: a = old Hz*10, b = new Hz*10
	VC_DIAG_EV_CAP,             // one frame-cap transition: a = start Hz, b = extreme Hz, c = end Hz, d = duration ms
	VC_DIAG_EV_HOST_GAP,        // acquire gap (compositor silent): a = gap ms
	VC_DIAG_EV_LAYER_PAUSED,    // Swift: LayerRenderer paused (headset off / crown)
	VC_DIAG_EV_LAYER_RESUMED,   // Swift: LayerRenderer running again
	VC_DIAG_EV_SPLASH,          // a = 0/1 (vc_splash_active)
	VC_DIAG_EV_LOADING,         // a = 0/1 (vc_loading_active)
	VC_DIAG_EV_MENU,            // a = 0/1 (pause menu)
	VC_DIAG_EV_TIMER_PAUSED,    // a = 0/1 (CTimer user/code pause)
	VC_DIAG_EV_LEVEL,           // a = old, b = new (CGame::currLevel)
	VC_DIAG_EV_AREA,            // a = old, b = new (CGame::currArea)
	VC_DIAG_EV_BLOCKING_LOAD,   // a = ms (a LoadAllRequestedModels that took > 20 ms)
	VC_DIAG_EV_LONG_FRAME,      // internal: the long-frame record ring carries the details
	VC_DIAG_EV_FADE,            // a = screen fade status (0 none, 1 fading, 2 fully faded)
	VC_DIAG_EV_CUTSCENE,        // a = 0/1 (CCutsceneMgr::IsRunning)
};

// Render sections of one Idle iteration, in mach ticks. The first group comes from the
// vc_frame_mark deltas, the second from the RenderScene stage accumulators (both eyes).
enum {
	VC_DIAG_SEC_CNSTR = 0,   // mark 0 -> 6  ConstructRenderList
	VC_DIAG_SEC_PRE,         // mark 6 -> 7  PreRender
	VC_DIAG_SEC_SETUP,       // mark 7 -> 1  StartOfFrame / RTT
	VC_DIAG_SEC_SKY,         // clouds + horizon
	VC_DIAG_SEC_ROADS,       // CRenderer::RenderRoads
	VC_DIAG_SEC_EBR,         // CRenderer::RenderEverythingBarRoads (buildings + vehicles + peds)
	VC_DIAG_SEC_VEH,         // vehicles inside EBR (CRenderer::RenderOneNonRoad)
	VC_DIAG_SEC_PED,         // peds inside EBR
	VC_DIAG_SEC_WATER,       // water (opaque + transparent)
	VC_DIAG_SEC_ALPHA,       // CRenderer::RenderFadingInEntities (sorted transparent entities)
	VC_DIAG_SEC_MISC,        // reflections, boats, rain streaks
	VC_DIAG_SEC_FX,          // RenderEffects (particles, shadows, glass, coronas, rest)
	VC_DIAG_SEC_EYEPASS,     // eye-pass setup/teardown (slice FBO bind, clear, matrices, MSAA resolve)
	VC_DIAG_SEC_READBACK,    // mark 2 -> 3
	VC_DIAG_SEC_HUD,         // mark 3 -> 4  Render2dStuff (HUD)
	VC_DIAG_SEC_MENU,        // mark 4 -> 8  RenderMenus
	VC_DIAG_SEC_AFTERFADE,   // mark 8 -> 9  DoFade + Render2dStuffAfterFade
	VC_DIAG_SEC_PRESENT,     // mark 9 -> 5  DoRWStuffEndOfFrame / publish
	VC_DIAG_SEC_N
};

// Scene-stage slot ids for the two sections that live outside main.cpp (Renderer.cpp).
// Must equal VC_SC_VEH / VC_SC_PED in the visionos.cpp enum (static_assert there).
#define VC_DIAG_SC_VEH 17
#define VC_DIAG_SC_PED 18

// Per-iteration record handed over by visionos.cpp at vc_frame_mark(5), i.e. once per Idle
// iteration -- world frames AND the menu/fade iterations that skip the world block. Ticks
// unless noted.
typedef struct vc_diag_frame_t {
	uint64_t periodTicks;   // previous mark 5 -> this mark 5 (0 on the first)
	uint64_t logicTicks;    // CGame::Process of this iteration
	uint64_t streamTicks;   // CStreaming::Update share of it
	uint64_t renderTicks;   // mark 0 -> mark 5 (0 when the world block did not run)
	uint64_t sec[VC_DIAG_SEC_N];
	int32_t  world;         // 1 = the world block ran (mark 0 seen since the previous mark 5)
	int32_t  splash, loading, menu, timerPaused, fade, cutscene;
	int32_t  level, area;
} vc_diag_frame_t;

extern std::atomic<int> g_vcDiagOn;   // 1 = recording (VC_DIAG_RECORD=1 seen at init)

extern "C" {
void vc_diag_init(void);                                    // game thread, once, before the game loop
void vc_diag_frame(const vc_diag_frame_t *f);              // game thread, once per Idle iteration (mark 5)
void vc_diag_acquire(uint64_t periodTicks, int hit, int ewmaHz, int capHz);   // render thread, per acquire
// Render thread, per acquired frame: publish -> acquire distance of the frame handed out
// (how long it waited in READY). Small values = the frame was finished just in time.
void vc_diag_pub_to_acq(uint64_t ageTicks);
// visionos_angle.mm: active buffer mode and pool size, for the session header.
const char *vcrt_buffer_mode_name(void);
int vcrt_num_buffers(void);
void vc_diag_count(int counter, uint64_t add);              // any thread
void vc_diag_max(int slot, uint64_t ticks);                 // any thread
void vc_diag_event(int kind, int a, int b, int c);          // any thread
void vc_diag_blocking_load(uint64_t ticks);                 // game thread: one LoadAllRequestedModels took `ticks`
// First-use tracking (librw gl3device.cpp flushCache): `key` identifies the draw-state
// combination (program, blend, depth, framebuffer, multiview, vertex attributes). The
// fixed hash table counts combinations never seen before in this session.
void vc_diag_combo(uint64_t key);
// Set by librw whenever a key component changed (program, blend/depth GL state, FBO,
// attribute set); flushCache rebuilds the key only then. Game thread only.
extern int g_vcDiagComboDirty;
extern unsigned g_vcDiagAttribMask;   // enabled vertex-attribute bit mask of the current draw
}

static inline bool vc_diag_on(void) { return g_vcDiagOn.load(std::memory_order_relaxed) != 0; }

#define VC_DIAG_COUNT(c, n)   do { if (vc_diag_on()) vc_diag_count((c), (uint64_t)(n)); } while (0)
#define VC_DIAG_MAX(s, t)     do { if (vc_diag_on()) vc_diag_max((s), (uint64_t)(t)); } while (0)
#define VC_DIAG_EVENT(k, a, b, c) do { if (vc_diag_on()) vc_diag_event((k), (a), (b), (c)); } while (0)
// Timestamp for the hook sites: 0 when off, so a hook can do `t = VC_DIAG_NOW(); ... if (t) ...`.
#define VC_DIAG_NOW()         (vc_diag_on() ? mach_absolute_time() : 0ull)

#endif // VC_DIAGNOSTICS
#endif // VC_DIAG_H
