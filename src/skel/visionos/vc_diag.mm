// Stutter diagnostics recorder -- implementation. See vc_diag.h for the contract.
// Compiled only in the Xcode "Diagnostics" configuration (VC_DIAGNOSTICS=1); the file is
// empty in Release and is not part of the macOS reference build at all.
#if defined(LIBRW_VISIONOS) && VC_DIAGNOSTICS

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <sys/stat.h>
#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "vc_diag.h"

std::atomic<int> g_vcDiagOn{0};
extern "C" { int g_vcDiagComboDirty = 0; unsigned g_vcDiagAttribMask = 0; }

// ---- time ------------------------------------------------------------------------------
static uint64_t g_ticksPerSec = 0;      // mach ticks in one second
static double   g_ticksToMs   = 0.0;    // mach ticks -> ms
static double   g_ticksToQms  = 0.0;    // mach ticks -> quarter-ms (histogram bucket)
static inline double vcdMs(uint64_t t) { return (double)t * g_ticksToMs; }
static inline uint64_t vcdWallNs(void)
{
	struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// ---- cross-thread counters (any thread) ------------------------------------------------
static std::atomic<uint64_t> g_cnt[VC_DIAG_C_N];     // monotonic sums
static std::atomic<uint64_t> g_maxv[VC_DIAG_M_N];    // per-window maxima (ticks), reset at the tick

// ---- render thread (vc_acquire_ready_frame) ---------------------------------------------
static std::atomic<uint64_t> g_acqTotal{0}, g_acqHit{0}, g_acqPeriodTicks{0}, g_acqPeriodN{0};   // monotonic
static std::atomic<uint64_t> g_acqMaxPeriod{0};        // window max acquire period (ticks)
static std::atomic<uint64_t> g_runMaxWin{0};           // window: longest run of repeats (ticks)
static std::atomic<uint64_t> g_acqLastPeriod{0};       // most recent acquire period (ticks); sizes a frame's repeats
// Publish -> acquire distance of every frame handed out, 0.1 ms buckets up to 25.5 ms (last
// bucket = overflow). Filled on the render thread, drained (exchange 0) by the game-thread tick.
#define VCD_P2A_N 256
static std::atomic<uint32_t> g_p2aHist[VCD_P2A_N];
static std::atomic<int> g_ewmaHz{0}, g_capHz{0};
// Owned by the render thread: a 16-period window for the "rate changed" detector. The
// window only counts as a reading when all 16 periods agree (max-min < 20 % of the mean),
// so a single doubled frame (a hitch) cannot fake a display-rate change.
#define VCD_RT_RING 16
static uint64_t g_rtRing[VCD_RT_RING]; static int g_rtPos = 0; static int g_rtFilled = 0;
static int      g_rtStableHz10 = 0;        // last accepted raw rate, Hz*10
static int      g_rtLastCap = -1;
static uint32_t g_rtRunCount = 0;          // current run of consecutive misses
// Cap transition: one event per excursion (start -> extreme -> end) instead of one per Hz.
static bool     g_capOpen = false;
static int      g_capStart = 0, g_capExt = 0;
static uint64_t g_capStartTick = 0, g_capLastChangeTick = 0;

// ---- game thread (vc_diag_frame) --------------------------------------------------------
#define VCD_HIST_N 128                      // 0.25 ms buckets: 0..32 ms, last bucket = overflow
static uint32_t g_hist[VCD_HIST_N];
static uint32_t g_frames = 0, g_worldFrames = 0, g_missed = 0;
static uint64_t g_maxPeriod = 0, g_maxLogic = 0, g_maxStream = 0, g_maxRender = 0;
static uint64_t g_probeSum = 0, g_probeMax = 0; static uint32_t g_probeN = 0;
static uint64_t g_winStart = 0, g_nextTick = 0;
static int g_stSplash = -1, g_stLoading = -1, g_stMenu = -1, g_stTimerPaused = -1, g_stFade = -1, g_stCutscene = -1;
static int g_stLevel = -999, g_stArea = -999;
static uint64_t g_prevCnt[VC_DIAG_C_N];
static uint64_t g_prevAcqTotal = 0, g_prevAcqHit = 0, g_prevAcqPeriod = 0, g_prevAcqPeriodN = 0;
static uint64_t g_frAcqTotal = 0, g_frAcqHit = 0;    // acquire counters at the previous game frame
static uint32_t g_comboCount = 0, g_comboOverflow = 0, g_comboPrevWin = 0, g_comboPrevFrame = 0;

// First-use table: open addressing, fixed, game thread only.
#define VCD_COMBO_N 4096
static uint64_t g_comboTab[VCD_COMBO_N];

// ---- per-second ring (game thread -> writer) --------------------------------------------
typedef struct {
	uint64_t wallNs, winTicks;
	uint32_t frames, worldFrames, missed;
	uint64_t maxPeriod, maxLogic, maxStream, maxRender;   // ticks
	int32_t  p95Bucket;                                    // histogram bucket (0.25 ms), -1 = none
	uint64_t cnt[VC_DIAG_C_N];                             // window deltas
	uint64_t maxv[VC_DIAG_M_N];                            // window maxima (ticks)
	uint64_t acqTotal, acqHit, acqPeriodTicks, acqPeriodN, acqMaxPeriod, runMax;
	int32_t  p2aMin, p2aP5, p2aP50;   // publish -> acquire, 0.1 ms buckets; -1 = no sample
	int32_t  ewmaHz, capHz;
	int32_t  splash, loading, menu, timerPaused, fade, cutscene, level, area;
	uint32_t comboNew, comboTotal;
	uint64_t probeSum, probeMax; uint32_t probeN;
	uint32_t writerUs;
} VcdSecond;
#define VCD_SEC_RING 64
static VcdSecond g_sec[VCD_SEC_RING];
static std::atomic<uint32_t> g_secWrite{0};
static uint32_t g_secRead = 0;   // writer only

// ---- long-frame ring (game thread -> writer) --------------------------------------------
typedef struct {
	uint64_t wallNs;
	uint64_t period, logic, stream, render;                // ticks
	uint64_t sec[VC_DIAG_SEC_N];
	uint32_t repeats, hits, comboNew;
	uint64_t runMax;                                       // longest repeat run during the frame (ticks)
	int32_t  world, splash, loading, menu, timerPaused, fade, cutscene;
} VcdLong;
#define VCD_LONG_RING 32
static VcdLong g_long[VCD_LONG_RING];
static std::atomic<uint32_t> g_longWrite{0};
static uint32_t g_longRead = 0;  // writer only

// ---- event ring (any thread -> writer) --------------------------------------------------
typedef struct {
	uint64_t wallNs;
	int32_t kind, a, b, c, d;
	std::atomic<uint32_t> seq;   // sequence+1 once the payload is complete
} VcdEvent;
#define VCD_EV_RING 128
static VcdEvent g_ev[VCD_EV_RING];
static std::atomic<uint32_t> g_evWrite{0};
static uint32_t g_evRead = 0;    // writer only

// ---- writer (utility-QoS dispatch timer) -------------------------------------------------
static FILE *g_file = NULL;
static char g_path[1024], g_pathOld[1024];
static dispatch_source_t g_timer = NULL;
static std::atomic<uint32_t> g_writerUs{0};   // last writer run, microseconds (reported in the next S line)
static int g_thermal = -1;
static uint64_t g_lastSlotWallNs = 0;
static bool g_silentReported = false;
#define VCD_MAX_BYTES (2u * 1024u * 1024u)

// =========================================================================================
// Hot paths
// =========================================================================================

extern "C" void vc_diag_count(int counter, uint64_t add)
{
	if ((unsigned)counter < VC_DIAG_C_N) g_cnt[counter].fetch_add(add, std::memory_order_relaxed);
}

static inline void vcdAtomicMax(std::atomic<uint64_t> &slot, uint64_t v)
{
	uint64_t cur = slot.load(std::memory_order_relaxed);
	while (v > cur && !slot.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {}
}

extern "C" void vc_diag_max(int slot, uint64_t ticks)
{
	if ((unsigned)slot < VC_DIAG_M_N) vcdAtomicMax(g_maxv[slot], ticks);
}

static void vcdEvent4(int kind, int a, int b, int c, int d)
{
	uint32_t n = g_evWrite.fetch_add(1, std::memory_order_relaxed);
	VcdEvent &e = g_ev[n % VCD_EV_RING];
	e.wallNs = vcdWallNs();
	e.kind = kind; e.a = a; e.b = b; e.c = c; e.d = d;
	e.seq.store(n + 1, std::memory_order_release);
}

extern "C" void vc_diag_event(int kind, int a, int b, int c)
{
	if (!vc_diag_on()) return;
	vcdEvent4(kind, a, b, c, 0);
}

extern "C" void vc_diag_blocking_load(uint64_t ticks)
{
	if (!vc_diag_on()) return;
	vc_diag_count(VC_DIAG_C_BLOCKING_LOADS, 1);
	vc_diag_max(VC_DIAG_M_BLOCKING_LOAD, ticks);
	if (ticks * 50 > g_ticksPerSec)   // > 20 ms: worth an event of its own
		vc_diag_event(VC_DIAG_EV_BLOCKING_LOAD, (int)vcdMs(ticks), 0, 0);
}

// Game thread (librw flushCache, only after a key component changed).
extern "C" void vc_diag_combo(uint64_t key)
{
	if (!vc_diag_on()) return;
	key |= 1;   // 0 marks an empty slot
	uint32_t idx = (uint32_t)((key ^ (key >> 29)) * 0x9E3779B1u) & (VCD_COMBO_N - 1);
	for (int probe = 0; probe < 64; probe++) {
		uint64_t v = g_comboTab[idx];
		if (v == key) return;
		if (v == 0) { g_comboTab[idx] = key; g_comboCount++; return; }
		idx = (idx + 1) & (VCD_COMBO_N - 1);
	}
	g_comboOverflow++;
}

// Render thread, once per compositor acquire. periodTicks = time since the previous
// acquire call (0 on the first), hit = a new game frame was handed out.
extern "C" void vc_diag_acquire(uint64_t periodTicks, int hit, int ewmaHz, int capHz)
{
	if (!vc_diag_on()) return;
	g_acqTotal.fetch_add(1, std::memory_order_relaxed);
	if (hit) g_acqHit.fetch_add(1, std::memory_order_relaxed);
	g_ewmaHz.store(ewmaHz, std::memory_order_relaxed);
	g_capHz.store(capHz, std::memory_order_relaxed);

	// Repeats: consecutive misses = the compositor re-showed the held frame. Run length is
	// approximated as misses x the current period (no extra clock read).
	if (!hit) {
		g_rtRunCount++;
	} else if (g_rtRunCount) {
		vcdAtomicMax(g_runMaxWin, (uint64_t)g_rtRunCount * periodTicks);
		g_rtRunCount = 0;
	}
	if (periodTicks) g_acqLastPeriod.store(periodTicks, std::memory_order_relaxed);

	if (periodTicks) {
		if (periodTicks * 10 > g_ticksPerSec) {          // > 100 ms: the compositor was silent
			vc_diag_event(VC_DIAG_EV_HOST_GAP, (int)vcdMs(periodTicks), 0, 0);
		} else if (periodTicks * 250 > g_ticksPerSec) {  // > 4 ms: a real display period
			g_acqPeriodTicks.fetch_add(periodTicks, std::memory_order_relaxed);
			g_acqPeriodN.fetch_add(1, std::memory_order_relaxed);
			vcdAtomicMax(g_acqMaxPeriod, periodTicks);
			g_rtRing[g_rtPos] = periodTicks;
			g_rtPos = (g_rtPos + 1) % VCD_RT_RING;
			if (g_rtFilled < VCD_RT_RING) g_rtFilled++;
			if (g_rtPos == 0 && g_rtFilled == VCD_RT_RING) {   // every 16th acquire: evaluate the window
				uint64_t sum = 0, mn = ~0ull, mx = 0;
				for (int i = 0; i < VCD_RT_RING; i++) { uint64_t v = g_rtRing[i]; sum += v; if (v < mn) mn = v; if (v > mx) mx = v; }
				uint64_t mean = sum / VCD_RT_RING;
				if ((mx - mn) * 5 < mean) {                      // steady window: a reading
					int hz10 = (int)((double)g_ticksPerSec * 10.0 / (double)mean + 0.5);
					if (g_rtStableHz10 == 0) {
						g_rtStableHz10 = hz10;
					} else if (hz10 - g_rtStableHz10 > 30 || g_rtStableHz10 - hz10 > 30) {
						vc_diag_event(VC_DIAG_EV_RATE, g_rtStableHz10, hz10, 0);
						g_rtStableHz10 = hz10;
					}
				}
			}
		}
	}

	// Cap transitions: open on the first change away from the base value, track the
	// extreme, close when the cap is back at the base (excursion) or has rested for 1 s
	// on a new value (a real change). One event either way.
	if (capHz != g_rtLastCap) {
		if (g_rtLastCap >= 0) {
			uint64_t now = mach_absolute_time();
			if (!g_capOpen) { g_capOpen = true; g_capStart = g_rtLastCap; g_capExt = capHz; g_capStartTick = now; }
			else {
				int dExt = g_capExt - g_capStart, dNew = capHz - g_capStart;
				if ((dNew < 0 ? -dNew : dNew) > (dExt < 0 ? -dExt : dExt)) g_capExt = capHz;
			}
			g_capLastChangeTick = now;
		}
		g_rtLastCap = capHz;
	}
	if (g_capOpen) {
		if (capHz == g_capStart) {
			vcdEvent4(VC_DIAG_EV_CAP, g_capStart, g_capExt, capHz, (int)vcdMs(g_capLastChangeTick - g_capStartTick));
			g_capOpen = false;
		} else if (periodTicks && (uint64_t)g_rtRunCount == 0 && mach_absolute_time() - g_capLastChangeTick > g_ticksPerSec) {
			vcdEvent4(VC_DIAG_EV_CAP, g_capStart, g_capExt, capHz, (int)vcdMs(g_capLastChangeTick - g_capStartTick));
			g_capOpen = false;
		}
	}
}

extern "C" void vc_diag_pub_to_acq(uint64_t ageTicks)
{
	if (!vc_diag_on()) return;
	uint32_t b = (uint32_t)((double)ageTicks * g_ticksToMs * 10.0);   // 0.1 ms buckets
	if (b >= VCD_P2A_N) b = VCD_P2A_N - 1;
	g_p2aHist[b].fetch_add(1, std::memory_order_relaxed);
}

static inline void vcdState(int *slot, int cur, int kind)
{
	if (*slot == cur) return;
	if (*slot != -1) vc_diag_event(kind, cur, 0, 0);   // -1 = first observation, no event
	*slot = cur;
}

// Game thread, once per second: copy the window into the ring and reset it.
static void vcdTick(uint64_t now)
{
	uint32_t w = g_secWrite.load(std::memory_order_relaxed);
	VcdSecond &s = g_sec[w % VCD_SEC_RING];
	s.wallNs = vcdWallNs();
	s.winTicks = now - g_winStart;
	s.frames = g_frames; s.worldFrames = g_worldFrames; s.missed = g_missed;
	s.maxPeriod = g_maxPeriod; s.maxLogic = g_maxLogic; s.maxStream = g_maxStream; s.maxRender = g_maxRender;
	s.p95Bucket = -1;
	if (g_frames) {
		uint32_t acc = 0, target = (g_frames * 95u + 99u) / 100u;
		for (int i = 0; i < VCD_HIST_N; i++) { acc += g_hist[i]; if (acc >= target) { s.p95Bucket = i; break; } }
	}
	for (int i = 0; i < VC_DIAG_C_N; i++) {
		uint64_t cur = g_cnt[i].load(std::memory_order_relaxed);
		s.cnt[i] = cur - g_prevCnt[i]; g_prevCnt[i] = cur;
	}
	for (int i = 0; i < VC_DIAG_M_N; i++) s.maxv[i] = g_maxv[i].exchange(0, std::memory_order_relaxed);
	{
		uint64_t t = g_acqTotal.load(std::memory_order_relaxed), h = g_acqHit.load(std::memory_order_relaxed);
		uint64_t p = g_acqPeriodTicks.load(std::memory_order_relaxed), n = g_acqPeriodN.load(std::memory_order_relaxed);
		s.acqTotal = t - g_prevAcqTotal; s.acqHit = h - g_prevAcqHit;
		s.acqPeriodTicks = p - g_prevAcqPeriod; s.acqPeriodN = n - g_prevAcqPeriodN;
		g_prevAcqTotal = t; g_prevAcqHit = h; g_prevAcqPeriod = p; g_prevAcqPeriodN = n;
		s.acqMaxPeriod = g_acqMaxPeriod.exchange(0, std::memory_order_relaxed);
		s.runMax = g_runMaxWin.exchange(0, std::memory_order_relaxed);
	}
	{
		// publish -> acquire distribution of the window: min, p5, median (bucket indices)
		uint32_t h[VCD_P2A_N], total = 0;
		for (int i = 0; i < VCD_P2A_N; i++) { h[i] = g_p2aHist[i].exchange(0, std::memory_order_relaxed); total += h[i]; }
		s.p2aMin = s.p2aP5 = s.p2aP50 = -1;
		if (total) {
			uint32_t acc = 0, t5 = (total * 5u + 99u) / 100u, t50 = (total * 50u + 99u) / 100u;
			for (int i = 0; i < VCD_P2A_N; i++) {
				if (!h[i]) continue;
				if (s.p2aMin < 0) s.p2aMin = i;
				acc += h[i];
				if (s.p2aP5 < 0 && acc >= t5) s.p2aP5 = i;
				if (acc >= t50) { s.p2aP50 = i; break; }
			}
		}
	}
	s.ewmaHz = g_ewmaHz.load(std::memory_order_relaxed);
	s.capHz = g_capHz.load(std::memory_order_relaxed);
	s.splash = g_stSplash; s.loading = g_stLoading; s.menu = g_stMenu; s.timerPaused = g_stTimerPaused;
	s.fade = g_stFade; s.cutscene = g_stCutscene;
	s.level = g_stLevel; s.area = g_stArea;
	s.comboNew = g_comboCount - g_comboPrevWin; s.comboTotal = g_comboCount; g_comboPrevWin = g_comboCount;
	s.probeSum = g_probeSum; s.probeMax = g_probeMax; s.probeN = g_probeN;
	s.writerUs = g_writerUs.load(std::memory_order_relaxed);
	g_secWrite.store(w + 1, std::memory_order_release);

	memset(g_hist, 0, sizeof(g_hist));
	g_frames = g_worldFrames = g_missed = 0;
	g_maxPeriod = g_maxLogic = g_maxStream = g_maxRender = 0;
	g_probeSum = g_probeMax = 0; g_probeN = 0;
	g_winStart = now;
	g_nextTick = now + g_ticksPerSec;
}

extern "C" void vc_diag_frame(const vc_diag_frame_t *f)
{
	if (!vc_diag_on() || !f) return;
	uint64_t t0 = mach_absolute_time();

	// Compositor view of this frame: new images handed out and repeats since the last one.
	uint64_t acqT = g_acqTotal.load(std::memory_order_relaxed), acqH = g_acqHit.load(std::memory_order_relaxed);
	uint32_t hits = (uint32_t)(acqH - g_frAcqHit);
	uint32_t repeats = (uint32_t)((acqT - g_frAcqTotal) - hits);
	g_frAcqTotal = acqT; g_frAcqHit = acqH;
	// The repeats of one long frame are consecutive compositor slots, so their run length is
	// repeats x the display period (the run itself only closes with the NEXT hit, which the
	// render thread sees after this call).
	uint64_t runMax = (uint64_t)repeats * g_acqLastPeriod.load(std::memory_order_relaxed);
	uint32_t comboNew = g_comboCount - g_comboPrevFrame; g_comboPrevFrame = g_comboCount;

	g_frames++;
	if (f->world) g_worldFrames++;
	if (f->periodTicks) {
		uint32_t b = (uint32_t)((double)f->periodTicks * g_ticksToQms);
		if (b >= VCD_HIST_N) b = VCD_HIST_N - 1;
		g_hist[b]++;
		if (f->periodTicks > g_maxPeriod) g_maxPeriod = f->periodTicks;
		int cap = g_capHz.load(std::memory_order_relaxed); if (cap <= 0) cap = 90;
		// budget = 1 s / cap; missed when period > 1.5 budget; a long frame when > 3 budgets
		if (f->periodTicks * (uint64_t)cap * 2 > g_ticksPerSec * 3) g_missed++;
		if (f->periodTicks * (uint64_t)cap > g_ticksPerSec * 3) {
			uint32_t w = g_longWrite.load(std::memory_order_relaxed);
			VcdLong &l = g_long[w % VCD_LONG_RING];
			l.wallNs = vcdWallNs();
			l.period = f->periodTicks; l.logic = f->logicTicks; l.stream = f->streamTicks; l.render = f->renderTicks;
			memcpy(l.sec, f->sec, sizeof(l.sec));
			l.repeats = repeats; l.hits = hits; l.comboNew = comboNew; l.runMax = runMax;
			l.world = f->world; l.splash = f->splash; l.loading = f->loading; l.menu = f->menu;
			l.timerPaused = f->timerPaused; l.fade = f->fade; l.cutscene = f->cutscene;
			g_longWrite.store(w + 1, std::memory_order_release);
		}
	}
	if (f->logicTicks  > g_maxLogic)  g_maxLogic  = f->logicTicks;
	if (f->streamTicks > g_maxStream) g_maxStream = f->streamTicks;
	if (f->renderTicks > g_maxRender) g_maxRender = f->renderTicks;

	vcdState(&g_stSplash, f->splash ? 1 : 0, VC_DIAG_EV_SPLASH);
	vcdState(&g_stLoading, f->loading ? 1 : 0, VC_DIAG_EV_LOADING);
	vcdState(&g_stMenu, f->menu ? 1 : 0, VC_DIAG_EV_MENU);
	vcdState(&g_stTimerPaused, f->timerPaused ? 1 : 0, VC_DIAG_EV_TIMER_PAUSED);
	vcdState(&g_stFade, f->fade, VC_DIAG_EV_FADE);
	vcdState(&g_stCutscene, f->cutscene ? 1 : 0, VC_DIAG_EV_CUTSCENE);
	if (f->level != g_stLevel) { if (g_stLevel != -999) vc_diag_event(VC_DIAG_EV_LEVEL, g_stLevel, f->level, 0); g_stLevel = f->level; }
	if (f->area  != g_stArea)  { if (g_stArea  != -999) vc_diag_event(VC_DIAG_EV_AREA,  g_stArea,  f->area,  0); g_stArea  = f->area; }

	if (t0 >= g_nextTick) vcdTick(t0);

	uint64_t dt = mach_absolute_time() - t0;   // own cost, including a tick when one ran
	g_probeSum += dt; g_probeN++;
	if (dt > g_probeMax) g_probeMax = dt;
}

// =========================================================================================
// Writer (background, utility QoS, once per second)
// =========================================================================================

static const char *vcdThermalName(int s)
{
	switch (s) {
		case NSProcessInfoThermalStateNominal:  return "nominal";
		case NSProcessInfoThermalStateFair:     return "fair";
		case NSProcessInfoThermalStateSerious:  return "serious";
		case NSProcessInfoThermalStateCritical: return "critical";
	}
	return "?";
}

static void vcdFmtTime(uint64_t wallNs, bool withMs, char *out, size_t n)
{
	time_t secs = (time_t)(wallNs / 1000000000ull);
	struct tm tmv; localtime_r(&secs, &tmv);
	if (withMs) snprintf(out, n, "%02d:%02d:%02d.%03d", tmv.tm_hour, tmv.tm_min, tmv.tm_sec, (int)((wallNs / 1000000ull) % 1000ull));
	else        snprintf(out, n, "%02d:%02d:%02d", tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
}

// st=<flags>: what the game was doing (play, or any of menu, splash, load, fade, fading, cut, tpause).
static void vcdStateStr(int menu, int splash, int loading, int timerPaused, int fade, int cutscene, char *st, size_t n)
{
	st[0] = '\0';
	if (menu) strlcat(st, "menu,", n);
	if (splash) strlcat(st, "splash,", n);
	if (loading) strlcat(st, "load,", n);
	if (fade == 2) strlcat(st, "fade,", n); else if (fade == 1) strlcat(st, "fading,", n);
	if (cutscene) strlcat(st, "cut,", n);
	if (timerPaused) strlcat(st, "tpause,", n);
	size_t sl = strlen(st); if (sl) st[sl - 1] = '\0'; else strlcpy(st, "play", n);
}

static void vcdWriteHeader(void)
{
	char ts[32]; vcdFmtTime(vcdWallNs(), false, ts, sizeof(ts));
	time_t secs = time(NULL); struct tm tmv; localtime_r(&secs, &tmv);
	NSString *dev = @"?";
	@autoreleasepool { id<MTLDevice> d = MTLCreateSystemDefaultDevice(); if (d) dev = d.name; }
	fprintf(g_file, "# [vc-diag] stutter diagnostics v2  session %04d-%02d-%02d %s  device='%s'  pid=%d  buffers=%d mode=%s\n",
	        tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, ts, dev.UTF8String, (int)getpid(),
	        vcrt_num_buffers(), vcrt_buffer_mode_name());
	fprintf(g_file, "# S HH:MM:SS win=<window s> frames=<game iterations> world=<of which with the world block> pub=<frames handed to the compositor>"
	                " miss=<intervals > 1.5x budget> max/p95=<frame interval ms>"
	                " | disp raw=<compositor rate Hz> ewma=<g_displayHz> cap=<frame cap Hz> acq=<new images>/<compositor frames> rep=<compositor frames without a new image> maxrep=<longest such run ms> acqmax=<longest acquire period ms>"
	                " p2a=<publish -> acquire wait of the frames shown: min/median/p5 ms> drop=<finished frames recycled unshown> wait=<acquires that left a frame waiting> drain=<drops by the fifo drain rule>"
	                " | logic/stream/render=<max ms in window> | strm req ld rd kb rdmax(ms) blk=<n>/<max ms>"
	                " | tex <uploads>/<kb>/<max ms> | link <n>/<max ms> | combo <new this second>/<total this session>"
	                " | thermal mem(MB) lvl area st=<play|menu,splash,load,fade,fading,cut,tpause>"
	                " | probe <avg>/<max> us per frame (own cost, game thread) wr=<writer ms>\n");
	fprintf(g_file, "# E HH:MM:SS.mmm <event>: display-rate, cap <start> -> <extreme> -> <end> Hz over <ms>, host gap, layer paused/resumed,"
	                " splash/loading/menu/fade/cutscene/timer-pause, level/area (0 generic 1 beach 2 mainland), blocking load, thermal, game thread silent, log rotated\n");
	fprintf(g_file, "# E long frame <ms> visible|hidden (<repeats>, <new images shown>, longest repeat <ms>) st=<state at the frame> world=0|1"
	                " | logic <ms> (stream <ms>) render <ms> | combos +<new draw-state combinations in this frame>"
	                " | sections ms: cnstr pre setup | sky roads world veh ped water alpha misc fx eyepass | readback hud menu afterfade present"
	                "  (world = buildings only: EBR minus veh and ped)\n");
}

static void vcdRotateIfNeeded(void)
{
	off_t pos = ftello(g_file);
	if (pos < (off_t)VCD_MAX_BYTES) return;
	fclose(g_file);
	rename(g_path, g_pathOld);   // one older file kept, overwritten each rotation
	g_file = fopen(g_path, "a");
	if (!g_file) return;
	vcdWriteHeader();
	char ts[32]; vcdFmtTime(vcdWallNs(), true, ts, sizeof(ts));
	fprintf(g_file, "%s E log rotated (previous file -> stutter.1.log)\n", ts);
}

static const char *vcdFadeName(int f) { return f == 2 ? "fully faded" : f == 1 ? "fading" : "none"; }

static void vcdWriteEvent(const VcdEvent &e)
{
	char ts[32]; vcdFmtTime(e.wallNs, true, ts, sizeof(ts));
	switch (e.kind) {
		case VC_DIAG_EV_RATE:
			fprintf(g_file, "%s E display-rate %.1f -> %.1f Hz (16-period window)\n", ts, e.a / 10.0, e.b / 10.0); break;
		case VC_DIAG_EV_CAP:
			if (e.a == e.c) fprintf(g_file, "%s E cap %d -> %d -> %d Hz over %d ms (excursion, back at base)\n", ts, e.a, e.b, e.c, e.d);
			else            fprintf(g_file, "%s E cap %d -> %d -> %d Hz over %d ms (settled on a new value)\n", ts, e.a, e.b, e.c, e.d);
			break;
		case VC_DIAG_EV_HOST_GAP:      fprintf(g_file, "%s E host gap %d ms (no compositor acquire)\n", ts, e.a); break;
		case VC_DIAG_EV_LAYER_PAUSED:  fprintf(g_file, "%s E layer paused\n", ts); break;
		case VC_DIAG_EV_LAYER_RESUMED: fprintf(g_file, "%s E layer resumed\n", ts); break;
		case VC_DIAG_EV_SPLASH:        fprintf(g_file, "%s E splash %s\n", ts, e.a ? "on" : "off"); break;
		case VC_DIAG_EV_LOADING:       fprintf(g_file, "%s E loading %s\n", ts, e.a ? "on" : "off"); break;
		case VC_DIAG_EV_MENU:          fprintf(g_file, "%s E menu %s\n", ts, e.a ? "on" : "off"); break;
		case VC_DIAG_EV_TIMER_PAUSED:  fprintf(g_file, "%s E timer-pause %s\n", ts, e.a ? "on" : "off"); break;
		case VC_DIAG_EV_FADE:          fprintf(g_file, "%s E fade %s\n", ts, vcdFadeName(e.a)); break;
		case VC_DIAG_EV_CUTSCENE:      fprintf(g_file, "%s E cutscene %s\n", ts, e.a ? "on" : "off"); break;
		case VC_DIAG_EV_LEVEL:         fprintf(g_file, "%s E level %d -> %d\n", ts, e.a, e.b); break;
		case VC_DIAG_EV_AREA:          fprintf(g_file, "%s E area %d -> %d\n", ts, e.a, e.b); break;
		case VC_DIAG_EV_BLOCKING_LOAD: fprintf(g_file, "%s E blocking load %d ms (LoadAllRequestedModels)\n", ts, e.a); break;
		default:                       fprintf(g_file, "%s E kind=%d a=%d b=%d c=%d d=%d\n", ts, e.kind, e.a, e.b, e.c, e.d); break;
	}
}

static void vcdWriteLong(const VcdLong &l)
{
	char ts[32]; vcdFmtTime(l.wallNs, true, ts, sizeof(ts));
	char st[64]; vcdStateStr(l.menu, l.splash, l.loading, l.timerPaused, l.fade, l.cutscene, st, sizeof(st));
	fprintf(g_file, "%s E long frame %d ms %s (%u repeats, %u new images shown, longest repeat %.0f ms) st=%s world=%d"
	                " | logic %.1f (stream %.1f) render %.1f | combos +%u",
	        ts, (int)vcdMs(l.period), l.repeats ? "visible" : "hidden", l.repeats, l.hits, vcdMs(l.runMax), st, l.world,
	        vcdMs(l.logic), vcdMs(l.stream), vcdMs(l.render), l.comboNew);
	if (l.world) {
		const uint64_t *s = l.sec;
		uint64_t ebr = s[VC_DIAG_SEC_EBR], vp = s[VC_DIAG_SEC_VEH] + s[VC_DIAG_SEC_PED];
		double world = vcdMs(ebr > vp ? ebr - vp : 0);
		fprintf(g_file, " | sections cnstr %.1f pre %.1f setup %.1f | sky %.1f roads %.1f world %.1f veh %.1f ped %.1f water %.1f alpha %.1f misc %.1f fx %.1f eyepass %.1f"
		                " | readback %.1f hud %.1f menu %.1f afterfade %.1f present %.1f\n",
		        vcdMs(s[VC_DIAG_SEC_CNSTR]), vcdMs(s[VC_DIAG_SEC_PRE]), vcdMs(s[VC_DIAG_SEC_SETUP]),
		        vcdMs(s[VC_DIAG_SEC_SKY]), vcdMs(s[VC_DIAG_SEC_ROADS]), world, vcdMs(s[VC_DIAG_SEC_VEH]), vcdMs(s[VC_DIAG_SEC_PED]),
		        vcdMs(s[VC_DIAG_SEC_WATER]), vcdMs(s[VC_DIAG_SEC_ALPHA]), vcdMs(s[VC_DIAG_SEC_MISC]), vcdMs(s[VC_DIAG_SEC_FX]), vcdMs(s[VC_DIAG_SEC_EYEPASS]),
		        vcdMs(s[VC_DIAG_SEC_READBACK]), vcdMs(s[VC_DIAG_SEC_HUD]), vcdMs(s[VC_DIAG_SEC_MENU]), vcdMs(s[VC_DIAG_SEC_AFTERFADE]), vcdMs(s[VC_DIAG_SEC_PRESENT]));
	} else {
		fprintf(g_file, " | sections: none (world block skipped: menu, fade or splash iteration)\n");
	}
}

static void vcdWriteSecond(const VcdSecond &s, int thermal, double memMb)
{
	char ts[32]; vcdFmtTime(s.wallNs, false, ts, sizeof(ts));
	double rawHz = s.acqPeriodN ? (double)s.acqPeriodN * (double)g_ticksPerSec / (double)s.acqPeriodTicks : 0.0;
	double p95 = s.p95Bucket >= 0 ? (s.p95Bucket + 0.5) * 0.25 : 0.0;
	char st[64]; vcdStateStr(s.menu, s.splash, s.loading, s.timerPaused, s.fade, s.cutscene, st, sizeof(st));
	double probeAvg = s.probeN ? vcdMs(s.probeSum) * 1000.0 / (double)s.probeN : 0.0;
	uint64_t rep = s.acqTotal > s.acqHit ? s.acqTotal - s.acqHit : 0;
	// publish -> acquire: bucket index -> ms text ("-" when no frame was handed out)
	char p2aMin[16], p2aP50[16], p2aP5[16];
	const int32_t p2aIdx[3] = { s.p2aMin, s.p2aP50, s.p2aP5 };
	char *p2aOut[3] = { p2aMin, p2aP50, p2aP5 };
	for (int i = 0; i < 3; i++) {
		if (p2aIdx[i] < 0) strlcpy(p2aOut[i], "-", 16);
		else if (p2aIdx[i] >= VCD_P2A_N - 1) strlcpy(p2aOut[i], ">25", 16);
		else snprintf(p2aOut[i], 16, "%.1f", (p2aIdx[i] + 0.5) * 0.1);
	}
	fprintf(g_file,
	        "%s S win=%.2f frames=%u world=%u pub=%llu miss=%u max=%.1f p95=%.1f"
	        " | disp raw=%.1f ewma=%d cap=%d acq=%llu/%llu rep=%llu maxrep=%.0f acqmax=%.1f p2a=%s/%s/%s drop=%llu wait=%llu drain=%llu"
	        " | logic=%.1f stream=%.1f render=%.1f"
	        " | strm req=%llu ld=%llu rd=%llu kb=%llu rdmax=%.1f blk=%llu/%.1f"
	        " | tex %llu/%llukb/%.1f | link %llu/%.1f | combo %u/%u"
	        " | thermal=%s mem=%.0f lvl=%d area=%d st=%s"
	        " | probe %.2f/%.1fus wr=%.2f\n",
	        ts, vcdMs(s.winTicks) / 1000.0, s.frames, s.worldFrames, (unsigned long long)s.cnt[VC_DIAG_C_PUBLISHES], s.missed, vcdMs(s.maxPeriod), p95,
	        rawHz, s.ewmaHz, s.capHz, (unsigned long long)s.acqHit, (unsigned long long)s.acqTotal, (unsigned long long)rep, vcdMs(s.runMax), vcdMs(s.acqMaxPeriod),
	        p2aMin, p2aP50, p2aP5, (unsigned long long)s.cnt[VC_DIAG_C_DROPS],
	        (unsigned long long)s.cnt[VC_DIAG_C_WAIT], (unsigned long long)s.cnt[VC_DIAG_C_DRAIN],
	        vcdMs(s.maxLogic), vcdMs(s.maxStream), vcdMs(s.maxRender),
	        (unsigned long long)s.cnt[VC_DIAG_C_STREAM_REQ], (unsigned long long)s.cnt[VC_DIAG_C_STREAM_LOADED],
	        (unsigned long long)s.cnt[VC_DIAG_C_CD_READS], (unsigned long long)(s.cnt[VC_DIAG_C_CD_BYTES] / 1024ull),
	        vcdMs(s.maxv[VC_DIAG_M_CD_READ]), (unsigned long long)s.cnt[VC_DIAG_C_BLOCKING_LOADS], vcdMs(s.maxv[VC_DIAG_M_BLOCKING_LOAD]),
	        (unsigned long long)s.cnt[VC_DIAG_C_TEX_UPLOADS], (unsigned long long)(s.cnt[VC_DIAG_C_TEX_BYTES] / 1024ull), vcdMs(s.maxv[VC_DIAG_M_TEX_UPLOAD]),
	        (unsigned long long)s.cnt[VC_DIAG_C_GL_LINKS], vcdMs(s.maxv[VC_DIAG_M_GL_LINK]), s.comboNew, s.comboTotal,
	        vcdThermalName(thermal), memMb, s.level, s.area, st,
	        probeAvg, vcdMs(s.probeMax) * 1000.0, s.writerUs / 1000.0);
}

static void vcdWriterTick(void)
{
	if (!g_file) return;
	uint64_t t0 = mach_absolute_time();

	// Thermal state and memory footprint are read here, off the game and render threads.
	int thermal = (int)[[NSProcessInfo processInfo] thermalState];
	if (thermal != g_thermal) {
		if (g_thermal != -1) {
			char ts[32]; vcdFmtTime(vcdWallNs(), true, ts, sizeof(ts));
			fprintf(g_file, "%s E thermal %s -> %s\n", ts, vcdThermalName(g_thermal), vcdThermalName(thermal));
		}
		g_thermal = thermal;
	}
	double memMb = -1.0;
	{
		task_vm_info_data_t info; mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
		if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &cnt) == KERN_SUCCESS)
			memMb = (double)info.phys_footprint / (1024.0 * 1024.0);
	}

	// Events and long frames first (they carry their own timestamps), then the completed seconds.
	uint32_t evW = g_evWrite.load(std::memory_order_acquire);
	if (evW - g_evRead > VCD_EV_RING) g_evRead = evW - VCD_EV_RING;   // fell behind: skip the overwritten ones
	for (; g_evRead != evW; g_evRead++) {
		VcdEvent &e = g_ev[g_evRead % VCD_EV_RING];
		uint32_t seq = e.seq.load(std::memory_order_acquire);
		if (seq != g_evRead + 1) { if (seq < g_evRead + 1) break; else continue; }   // not yet written / overwritten
		vcdWriteEvent(e);
	}
	uint32_t lgW = g_longWrite.load(std::memory_order_acquire);
	if (lgW - g_longRead > VCD_LONG_RING) g_longRead = lgW - VCD_LONG_RING;
	for (; g_longRead != lgW; g_longRead++) vcdWriteLong(g_long[g_longRead % VCD_LONG_RING]);
	uint32_t secW = g_secWrite.load(std::memory_order_acquire);
	if (secW - g_secRead > VCD_SEC_RING) g_secRead = secW - VCD_SEC_RING;
	bool gotSlot = false;
	for (; g_secRead != secW; g_secRead++) {
		vcdWriteSecond(g_sec[g_secRead % VCD_SEC_RING], thermal, memMb);
		g_lastSlotWallNs = g_sec[g_secRead % VCD_SEC_RING].wallNs;
		gotSlot = true;
	}
	if (gotSlot) g_silentReported = false;
	else if (g_lastSlotWallNs && !g_silentReported) {
		uint64_t now = vcdWallNs();
		if (now - g_lastSlotWallNs >= 3000000000ull) {   // no game frame for 3 s: a stall or a load
			char ts[32]; vcdFmtTime(now, true, ts, sizeof(ts));
			fprintf(g_file, "%s E game thread silent for %d s (no frame since the last S line)\n", ts, (int)((now - g_lastSlotWallNs) / 1000000000ull));
			g_silentReported = true;
		}
	}
	fflush(g_file);
	vcdRotateIfNeeded();

	g_writerUs.store((uint32_t)(vcdMs(mach_absolute_time() - t0) * 1000.0), std::memory_order_relaxed);
}

// =========================================================================================
// Init (game thread, once)
// =========================================================================================

extern "C" void vc_diag_init(void)
{
	static bool done = false;
	if (done) return;
	done = true;

	const char *e = getenv("VC_DIAG_RECORD");
	if (!e || e[0] != '1') {
		printf("[vc-diag] recorder off (launcher toggle 'Record stutter diagnostics' is off)\n");
		return;
	}

	mach_timebase_info_data_t tb; mach_timebase_info(&tb);
	g_ticksPerSec = (uint64_t)((double)tb.denom * 1.0e9 / (double)tb.numer + 0.5);
	g_ticksToMs   = (double)tb.numer / (double)tb.denom / 1.0e6;
	g_ticksToQms  = g_ticksToMs * 4.0;

	const char *home = getenv("HOME");
	if (!home || !home[0]) { printf("[vc-diag] HOME unset -> recorder off\n"); return; }
	char dir[1024];
	snprintf(dir, sizeof(dir), "%s/Documents/Diagnostics", home);
	mkdir(dir, 0755);
	snprintf(g_path, sizeof(g_path), "%s/stutter.log", dir);
	snprintf(g_pathOld, sizeof(g_pathOld), "%s/stutter.1.log", dir);
	g_file = fopen(g_path, "a");
	if (!g_file) { printf("[vc-diag] cannot open %s -> recorder off\n", g_path); return; }
	vcdWriteHeader();
	fflush(g_file);
	vcdRotateIfNeeded();

	g_thermal = -1;
	uint64_t now = mach_absolute_time();
	g_winStart = now;
	g_nextTick = now + g_ticksPerSec;
	g_vcDiagComboDirty = 1;   // the first draw after start is a new combination by definition

	dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_UTILITY, 0);
	g_timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, q);
	dispatch_source_set_timer(g_timer, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC), NSEC_PER_SEC, 100 * NSEC_PER_MSEC);
	dispatch_source_set_event_handler(g_timer, ^{ @autoreleasepool { vcdWriterTick(); } });
	dispatch_resume(g_timer);

	g_vcDiagOn.store(1, std::memory_order_release);
	printf("[vc-diag] recording to %s (1 line/s, rotates at 2 MB)\n", g_path);
}

#endif // LIBRW_VISIONOS && VC_DIAGNOSTICS
