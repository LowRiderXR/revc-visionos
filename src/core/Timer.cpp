#define WITHWINDOWS
#include "common.h"
#include "crossplatform.h"

#include "DMAudio.h"
#include "Record.h"
#include "Timer.h"
#include "SpecialFX.h"

// visionOS gets the same high-resolution performance-counter path as Win32
// (QueryPerformance* are shimmed over mach_absolute_time in crossplatform.cpp),
// instead of the quantised millisecond fallback. macOS/Linux keep the fallback.
#if defined(_WIN32) || defined(LIBRW_VISIONOS)
#define RW_HIRES_TIMER
#endif

#ifdef LIBRW_VISIONOS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if VC_DIAGNOSTICS
#include "skel/visionos/vc_diag.h"   // stutter recorder: simulated timestep per frame
#endif

// Quantized timestep (2026-11-17). The game's frames are paced to the display (frame cap in
// vcrt_begin_frame) and shown in whole display slots, but the wall-clock delta between two
// Idle calls still scatters by up to +-35 % of a period. Integrating the camera and the
// physics with that scattered delta while the pictures appear in even slots is micro-judder
// with every slot filled. Mountain 64 avoids it by taking the delta from the presentation
// schedule. Here: round the measured delta to whole display periods, carrying the remainder
// so game time keeps following the wall clock (bounded by about half a period).
//   VC_TIMESTEP=wall restores the measured delta (A/B); default quant.
extern "C" double vc_timestep_period_ms(void);   // visionos_angle.mm: cap period, 0 = not paced
static int vcTimestepQuant(void)
{
	static int quant = -1;
	if (quant < 0) {
		const char *e = getenv("VC_TIMESTEP");
		quant = (e && strcmp(e, "wall") == 0) ? 0 : 1;
		printf("[vc-timer] timestep = %s\n", quant ? "quant (whole display periods, remainder carried)" : "wall (measured delta, VC_TIMESTEP=wall)");
	}
	return quant;
}
extern "C" const char *vc_timestep_mode_name(void) { return vcTimestepQuant() ? "quant" : "wall"; }

// Returns the delta to simulate, in ms, for a measured wall-clock delta of rawMs.
static double vcQuantizeDelta(double rawMs)
{
	static double carryMs = 0.0;
	double periodMs = vcTimestepQuant() ? vc_timestep_period_ms() : 0.0;
	if (periodMs <= 0.0) { carryMs = 0.0; return rawMs; }           // not paced: wall clock
	if (rawMs > 4.0 * periodMs) { carryMs = 0.0; return rawMs; }     // a stall or a load: take it as measured
	double total = rawMs + carryMs;
	long n = lround(total / periodMs);
	// A real frame is never zero periods: the cap grid can hand out a short frame right
	// after a catch-up, and the camera must not stand still for one picture. Deltas under a
	// quarter period are the second Update() call of the same frame (loads, cutscene start)
	// and get zero, exactly like the stock minimum step.
	if (n < 1 && rawMs >= 0.25 * periodMs) n = 1;
	if (n < 0) n = 0;
	double used = (double)n * periodMs;
	carryMs = total - used;
	if (carryMs >  0.5 * periodMs) carryMs =  0.5 * periodMs;      // bounded: never grows across frames
	if (carryMs < -0.5 * periodMs) carryMs = -0.5 * periodMs;
	return used;
}

// DIAGNOSE: once per second, the smallest/largest per-frame delta in microseconds.
// Direct proof the clock is now sub-millisecond (multiples of 1000 us before the
// fix; ~11100 us with spread after). ns arg = raw 32-bit counter delta.
static void vcTimerLogDelta(long long deltaNs)
{
	static long long minNs = 0x7FFFFFFFFFFFFFFFLL, maxNs = 0;
	static int       count = 0;
	static double    lastLogMs = -1.0;
	if (deltaNs < minNs) minNs = deltaNs;
	if (deltaNs > maxNs) maxNs = deltaNs;
	count++;
	LARGE_INTEGER now; QueryPerformanceCounter(&now);
	double nowMs = (double)(unsigned long long)now.QuadPart / 1.0e6;
	if (lastLogMs < 0.0) lastLogMs = nowMs;
	if (nowMs - lastLogMs >= 1000.0) {
		printf("[vc-timer] frame delta min=%lld us max=%lld us over %d frames\n",
		       minNs / 1000, maxNs / 1000, count);
		minNs = 0x7FFFFFFFFFFFFFFFLL; maxNs = 0; count = 0; lastLogMs = nowMs;
	}
}
#endif

uint32 CTimer::m_snTimeInMilliseconds;
uint32 CTimer::m_snTimeInMillisecondsPauseMode = 1;

uint32 CTimer::m_snTimeInMillisecondsNonClipped;
uint32 CTimer::m_snPreviousTimeInMilliseconds;
uint32 CTimer::m_FrameCounter;
float CTimer::ms_fTimeScale;
float CTimer::ms_fTimeStep;
float CTimer::ms_fTimeStepNonClipped;
bool  CTimer::m_UserPause;
bool  CTimer::m_CodePause; 
#ifdef FIX_BUGS
uint32 CTimer::m_LogicalFrameCounter;
uint32 CTimer::m_LogicalFramesPassed;
#endif

uint32 _nCyclesPerMS = 1;

#ifdef RW_HIRES_TIMER
LARGE_INTEGER _oldPerfCounter;
LARGE_INTEGER perfSuspendCounter;
#define RsTimerType uint32
#else
#define RsTimerType double
#endif

RsTimerType oldPcTimer;

RsTimerType suspendPcTimer;

uint32 suspendDepth;

void CTimer::Initialise(void)
{
	debug("Initialising CTimer...\n");
	
	ms_fTimeScale = 1.0f;
	ms_fTimeStep = 1.0f;
	suspendDepth = 0;
	m_UserPause = false;
	m_CodePause = false;
	m_snTimeInMillisecondsNonClipped = 0;
	m_snPreviousTimeInMilliseconds = 0;
	m_snTimeInMilliseconds = 1;
#ifdef FIX_BUGS
	m_LogicalFrameCounter = 0;
	m_LogicalFramesPassed = 0;
#endif
	
#ifdef RW_HIRES_TIMER
	LARGE_INTEGER perfFreq;
	if ( QueryPerformanceFrequency(&perfFreq) )
	{
		OutputDebugString("Performance counter available (high-resolution timer active)\n");
		_nCyclesPerMS = uint32(perfFreq.QuadPart / 1000);
		QueryPerformanceCounter(&_oldPerfCounter);
	}
	else
#endif
	{
		OutputDebugString("Performance counter not available, using millesecond timer\n");
		_nCyclesPerMS = 0;
		oldPcTimer = RsTimer();
	}
	
	m_snTimeInMilliseconds = m_snPreviousTimeInMilliseconds;
	
	m_FrameCounter = 0;
	
	DMAudio.ResetTimers(m_snPreviousTimeInMilliseconds);
	
	debug("CTimer ready\n");
}

void CTimer::Shutdown(void)
{
	;
}

#ifdef FIX_BUGS
void CTimer::Update(void)
{ 
	static double frameTimeLogical = 0.0;
	static double frameTimeFraction = 0.0;
	static double frameTimeFractionScaled = 0.0;
	double frameTime;
	double dblUpdInMs;

	m_snPreviousTimeInMilliseconds = m_snTimeInMilliseconds;
	
#ifdef RW_HIRES_TIMER
	if ( (double)_nCyclesPerMS != 0.0 )
	{
		LARGE_INTEGER pc;
		QueryPerformanceCounter(&pc);
		
		// Full 64-bit delta. The old 32-bit LowPart subtraction wrapped every
		// ~4.29 s at our nanosecond QueryPerformanceCounter frequency, mangling
		// the timestep after a stall. Stays uint64 until the divide below.
		uint64 updInCycles = (uint64)(pc.QuadPart - _oldPerfCounter.QuadPart);

		_oldPerfCounter = pc;
#ifdef LIBRW_VISIONOS
		vcTimerLogDelta(updInCycles); // updInCycles is nanoseconds (freq = 1e9)

		// Quantized timestep: one value for the frame, the time scale applied after rounding,
		// and frameTime, dblUpdInMs and the pause-mode clock all derived from it.
		dblUpdInMs = vcQuantizeDelta((double)updInCycles / (double)_nCyclesPerMS);
		frameTime = GetIsPaused() ? dblUpdInMs : dblUpdInMs * ms_fTimeScale;
#if VC_DIAGNOSTICS
		if (vc_diag_on()) vc_diag_simdt(dblUpdInMs);
#endif
#else
		float updInCyclesScaled = GetIsPaused() ? updInCycles : updInCycles * ms_fTimeScale;

		frameTime = updInCyclesScaled / (double)_nCyclesPerMS;

		dblUpdInMs = (double)updInCycles / (double)_nCyclesPerMS;
#endif
	}
	else
#endif
	{
		RsTimerType timer = RsTimer();
		
		RsTimerType updInMs = timer - oldPcTimer;
		
		frameTime = (double)updInMs * ms_fTimeScale;
		
		oldPcTimer = timer;

		dblUpdInMs = (double)updInMs;
	}

	// count frames as if we're running at 30 fps
	m_LogicalFramesPassed = 0;
	frameTimeLogical += dblUpdInMs;
	while (frameTimeLogical >= 1000.0 / 30.0) {
		frameTimeLogical -= 1000.0 / 30.0;
		m_LogicalFramesPassed++;
	}
	m_LogicalFrameCounter += m_LogicalFramesPassed;

	frameTimeFraction += dblUpdInMs;
	frameTimeFractionScaled += frameTime;

	m_snTimeInMillisecondsPauseMode += uint32(frameTimeFraction);
		
	if ( GetIsPaused() )
		ms_fTimeStep = 0.0f;
	else
	{
		m_snTimeInMilliseconds += uint32(frameTimeFractionScaled);
		m_snTimeInMillisecondsNonClipped += uint32(frameTimeFractionScaled);
		ms_fTimeStep = frameTime / 1000.0f * 50.0f;
	}
	frameTimeFraction -= uint32(frameTimeFraction);
	frameTimeFractionScaled -= uint32(frameTimeFractionScaled);
	
	if ( ms_fTimeStep < 0.01f && !GetIsPaused() && !CSpecialFX::bSnapShotActive)
		ms_fTimeStep = 0.01f;

	ms_fTimeStepNonClipped = ms_fTimeStep;
	
	if ( !CRecordDataForGame::IsPlayingBack() )
	{
		ms_fTimeStep = Min(3.0f, ms_fTimeStep);

		if ( (m_snTimeInMilliseconds - m_snPreviousTimeInMilliseconds) > 60 )
			m_snTimeInMilliseconds = m_snPreviousTimeInMilliseconds + 60;
	}
  
	if ( CRecordDataForChase::IsRecording() )
	{
		ms_fTimeStep = 1.0f;
		m_snTimeInMilliseconds = m_snPreviousTimeInMilliseconds + 16;
	}
  
	m_FrameCounter++;
}
#else
void CTimer::Update(void)
{
	m_snPreviousTimeInMilliseconds = m_snTimeInMilliseconds;
	
#ifdef RW_HIRES_TIMER
	if ( (double)_nCyclesPerMS != 0.0 )
	{
		LARGE_INTEGER pc;
		QueryPerformanceCounter(&pc);
		
		// Full 64-bit delta. The old 32-bit LowPart subtraction wrapped every
		// ~4.29 s at our nanosecond QueryPerformanceCounter frequency, mangling
		// the timestep after a stall. Stays uint64 until the divide below.
		uint64 updInCycles = (uint64)(pc.QuadPart - _oldPerfCounter.QuadPart);
		
		_oldPerfCounter = pc;
		
		float updInCyclesScaled = GetIsPaused() ? updInCycles : updInCycles * ms_fTimeScale;
		
		double frameTime = updInCyclesScaled / (double)_nCyclesPerMS;

		m_snTimeInMillisecondsPauseMode = m_snTimeInMillisecondsPauseMode + frameTime;
		
		if ( GetIsPaused() )
			ms_fTimeStep = 0.0f;
		else
		{
			m_snTimeInMilliseconds = m_snTimeInMilliseconds + frameTime;
			m_snTimeInMillisecondsNonClipped = m_snTimeInMillisecondsNonClipped + frameTime;
			ms_fTimeStep = frameTime / 1000.0f * 50.0f;
		}
	}
	else
#endif
	{
		RsTimerType timer = RsTimer();
		
		RsTimerType updInMs = timer - oldPcTimer;
		
		double frameTime = (double)updInMs * ms_fTimeScale;
		
		oldPcTimer = timer;

		m_snTimeInMillisecondsPauseMode = m_snTimeInMillisecondsPauseMode + frameTime;

		if ( GetIsPaused() )
			ms_fTimeStep = 0.0f;
		else
		{
			m_snTimeInMilliseconds = m_snTimeInMilliseconds + frameTime;
			m_snTimeInMillisecondsNonClipped = m_snTimeInMillisecondsNonClipped + frameTime;
			ms_fTimeStep = frameTime / 1000.0f * 50.0f;
		}
	}
	
	if ( ms_fTimeStep < 0.01f && !GetIsPaused() && !CSpecialFX::bSnapShotActive)
		ms_fTimeStep = 0.01f;

	ms_fTimeStepNonClipped = ms_fTimeStep;
	
	if ( !CRecordDataForGame::IsPlayingBack() )
	{
		ms_fTimeStep = Min(3.0f, ms_fTimeStep);

		if ( (m_snTimeInMilliseconds - m_snPreviousTimeInMilliseconds) > 60 )
			m_snTimeInMilliseconds = m_snPreviousTimeInMilliseconds + 60;
	}
  
	if ( CRecordDataForChase::IsRecording() )
	{
		ms_fTimeStep = 1.0f;
		m_snTimeInMilliseconds = m_snPreviousTimeInMilliseconds + 16;
	}
  
	m_FrameCounter++;
}
#endif

void CTimer::Suspend(void)
{
	if ( ++suspendDepth > 1 )
		return;
	
#ifdef RW_HIRES_TIMER
	if ( (double)_nCyclesPerMS != 0.0 )
		QueryPerformanceCounter(&perfSuspendCounter);
	else
#endif
		suspendPcTimer = RsTimer();
}

void CTimer::Resume(void)
{
	if ( --suspendDepth != 0 )
		return;

#ifdef RW_HIRES_TIMER
	if ( (double)_nCyclesPerMS != 0.0 )
	{
		LARGE_INTEGER pc;
		QueryPerformanceCounter(&pc);

		_oldPerfCounter.QuadPart += pc.QuadPart - perfSuspendCounter.QuadPart;
	}
	else
#endif
		oldPcTimer += RsTimer() - suspendPcTimer;
}

uint32 CTimer::GetCyclesPerMillisecond(void)
{
#ifdef RW_HIRES_TIMER
	if (_nCyclesPerMS != 0)
		return _nCyclesPerMS;
	else 
#endif
		return 1;
}

// NOTE: the uint32 return is INTENTIONAL (reVC API), but at our nanosecond
// QueryPerformanceCounter frequency it holds only ~4.3 s before wrapping. This
// function is therefore valid ONLY for intra-frame differences (time since the
// last CTimer::Update, always << one frame). Do not use it to measure spans
// that can exceed a frame -- use a full 64-bit clock for that instead.
uint32 CTimer::GetCurrentTimeInCycles(void)
{
#ifdef RW_HIRES_TIMER
	if ( _nCyclesPerMS != 0 )
	{
		LARGE_INTEGER pc;
		QueryPerformanceCounter(&pc);
		// 64-bit delta (see Update). Intra-frame query, so it fits the uint32
		// return; computing from QuadPart avoids the 4.29 s LowPart wrap.
		return (uint32)(uint64)(pc.QuadPart - _oldPerfCounter.QuadPart);
	}
	else
#endif
		return RsTimer() - oldPcTimer;
}

bool CTimer::GetIsSlowMotionActive(void)
{
	return ms_fTimeScale < 1.0f;
}

void CTimer::Stop(void)
{
	m_snPreviousTimeInMilliseconds = m_snTimeInMilliseconds;
}

void CTimer::StartUserPause(void)
{
	m_UserPause = true;
}

void CTimer::EndUserPause(void)
{
	m_UserPause = false;
}

uint32 CTimer::GetCyclesPerFrame()
{
	return 20;
}

