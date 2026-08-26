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

// Game render-target size. THE single source: feeds the two vcrt MTLTextures,
// rsCAMERASIZE, RsGlobal and the librw open params. This is the OFFSCREEN size
// the game renders into for display on the world-anchored canvas -- a design
// choice (16:9), deliberately independent of the physical drawable dimensions;
// the canvas is a quad, not the drawable.
// TODO(visionos): later derive dynamically from the quad geometry and the
// angular resolution instead of a fixed 1920x1080.
#define VISIONOS_SCREEN_WIDTH  1920
#define VISIONOS_SCREEN_HEIGHT 1080

// Single source for the render-target size, queried by librw (gl3device) over
// the C seam -- same pattern as vc_external_framebuffer(). No duplicated
// constant, linker-checked; when the size becomes dynamic only this file changes.
extern "C" void vc_screen_size(int *w, int *h)
{
	if (w) *w = VISIONOS_SCREEN_WIDTH;
	if (h) *h = VISIONOS_SCREEN_HEIGHT;
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
// values MUST match. Read once from VC_RENDER_MODE (default cinema). Stereo is
// not implemented yet, so it logs a stub line and falls back to cinema; the
// EFFECTIVE mode returned here is therefore cinema. Kept in a mutable global (no
// compile-time bake-in) so a later runtime switch stays possible.
enum { VC_MODE_CINEMA = 0, VC_MODE_STEREO = 1 };
static int g_renderMode = -1;   // -1 = not yet resolved

extern "C" int vc_render_mode(void)
{
	if (g_renderMode < 0) {
		const char *v = getenv("VC_RENDER_MODE");
		int requested = (v && strcasecmp(v, "stereo") == 0) ? VC_MODE_STEREO : VC_MODE_CINEMA;
		if (requested == VC_MODE_STEREO) {
			printf("[vc-mode] VC_RENDER_MODE = stereo (requested)\n");
			printf("[vc-mode] stereo not implemented yet -> falling back to cinema\n");
			g_renderMode = VC_MODE_CINEMA;   // stub fallback
		} else {
			g_renderMode = VC_MODE_CINEMA;
		}
		printf("[vc-mode] active render mode = %s\n",
		       g_renderMode == VC_MODE_CINEMA ? "cinema" : "stereo");
	}
	return g_renderMode;
}

// --- Camera matrix override seam (stereo injection point) -------------------
// gl3device beginUpdate consumes these (getters below) after computing its own
// view/proj; when active it uploads ours instead. Buffered under a lock: the
// setters will later be driven from the compositor (main thread) while the
// getters run on the game thread. 16 floats each, column-major, librw convention.
static pthread_mutex_t g_mtxMutex = PTHREAD_MUTEX_INITIALIZER;
static float g_ovView[16];
static float g_ovProj[16];
static int   g_ovActive = 0;

extern "C" void vc_set_view_matrix(const float m[16])
{
	if (!m) return;
	pthread_mutex_lock(&g_mtxMutex); memcpy(g_ovView, m, 16 * sizeof(float)); pthread_mutex_unlock(&g_mtxMutex);
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
extern "C" void vc_get_view_matrix(float m[16])
{
	pthread_mutex_lock(&g_mtxMutex); memcpy(m, g_ovView, 16 * sizeof(float)); pthread_mutex_unlock(&g_mtxMutex);
}
extern "C" void vc_get_projection_matrix(float m[16])
{
	pthread_mutex_lock(&g_mtxMutex); memcpy(m, g_ovProj, 16 * sizeof(float)); pthread_mutex_unlock(&g_mtxMutex);
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
	if (s.field != prev.field) printf("[vc-input] %s %s\n", name, s.field ? "DOWN" : "UP");
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
	RsGlobal.maximumWidth  = VISIONOS_SCREEN_WIDTH;
	RsGlobal.maximumHeight = VISIONOS_SCREEN_HEIGHT;
	RsGlobal.width         = VISIONOS_SCREEN_WIDTH;
	RsGlobal.height        = VISIONOS_SCREEN_HEIGHT;

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

// Derived from the single source above so it never drifts (stringify the macro).
#define VC_STR2(x) #x
#define VC_STR(x)  VC_STR2(x)
static RwChar  _VMString[] = VC_STR(VISIONOS_SCREEN_WIDTH) " X " VC_STR(VISIONOS_SCREEN_HEIGHT) " X 32";
static RwChar *_VMList[1]  = { _VMString };

RwChar **
_psGetVideoModeList()
{
	// TODO(visionos): genau ein hartkodierter Modus.
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
	if (!vcrt_create(VISIONOS_SCREEN_WIDTH, VISIONOS_SCREEN_HEIGHT)) {
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
	openParams.width       = VISIONOS_SCREEN_WIDTH;
	openParams.height      = VISIONOS_SCREEN_HEIGHT;
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
	camRect.w = VISIONOS_SCREEN_WIDTH;
	camRect.h = VISIONOS_SCREEN_HEIGHT;
	RsEventHandler(rsCAMERASIZE, &camRect);
	printf("[vc-loop] rsCAMERASIZE %dx%d done\n", VISIONOS_SCREEN_WIDTH, VISIONOS_SCREEN_HEIGHT);

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

		if (!FrontEndMenuManager.m_bWantToRestart)
			break;
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
