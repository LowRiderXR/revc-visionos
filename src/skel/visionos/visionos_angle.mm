// visionos_angle.mm
//
// ANGLE bring-up for the visionOS skeleton. Kept in a separate ObjC++ unit so
// the EGL/GLES/Metal code does not have to include reVC's C++ headers (whose
// `nil`/`TRUE` macros clash with Foundation/ObjC). visionos.cpp calls
// vcgl_init_angle() from psInitialize().
//
// libEGL/libGLESv2 are ANGLE xcframeworks in the app bundle; they are NOT linked
// but loaded at runtime via dlopen. No ANGLE header is used: the few EGL
// constants are spelled out as integer literals (Khronos EGL registry / ANGLE).
//
// This step only brings up a surfaceless GLES 3.0 context and logs each step;
// no rendering, no framebuffer, no CompositorServices.

#ifdef LIBRW_VISIONOS

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <math.h>   // fabsf/lroundf (mv4 self-test)
#include <stdint.h>
#include <stdlib.h>   // getenv
#include <pthread.h>  // throttle mutex/cond
#include <time.h>     // clock_gettime for the wait deadline
#include <errno.h>    // ETIMEDOUT
#include <mach/mach_time.h> // mach_absolute_time for the rate-window measurement
#include <mach/mach.h>      // task_info / phys_footprint (memory probe)
#include <os/proc.h>        // os_proc_available_memory (headroom before jetsam)

#if __has_feature(objc_arc)
  #define VC_OBJ_TO_VOID(o) ((__bridge void *)(o))
#else
  #define VC_OBJ_TO_VOID(o) ((void *)(o))
#endif

// --- EGL constants as integer literals (no ANGLE headers in the project) ----
enum {
	VC_EGL_NONE                  = 0x3038,
	VC_EGL_OPENGL_ES_API         = 0x30A0,
	VC_EGL_OPENGL_ES3_BIT        = 0x0040, // EGL_OPENGL_ES3_BIT_KHR
	VC_EGL_RENDERABLE_TYPE       = 0x3040,
	VC_EGL_RED_SIZE              = 0x3024,
	VC_EGL_GREEN_SIZE            = 0x3023,
	VC_EGL_BLUE_SIZE             = 0x3022,
	VC_EGL_ALPHA_SIZE            = 0x3021,
	VC_EGL_DEPTH_SIZE            = 0x3025,
	VC_EGL_VENDOR                = 0x3053,
	VC_EGL_VERSION               = 0x3054,
	VC_EGL_CONTEXT_MAJOR_VERSION = 0x3098,
	VC_EGL_CONTEXT_MINOR_VERSION = 0x30FB,
	VC_EGL_DEVICE_EXT            = 0x322C, // eglQueryDisplayAttribEXT attribute
	VC_EGL_METAL_DEVICE_ANGLE    = 0x34A6, // eglQueryDeviceAttribEXT attribute
};

typedef unsigned int EGLBoolean;
typedef int          EGLint;
typedef unsigned int EGLenum;
typedef void        *EGLDisplay;
typedef void        *EGLConfig;
typedef void        *EGLContext;
typedef void        *EGLSurface;
typedef void        *EGLDeviceEXT;
typedef intptr_t     EGLAttrib;
typedef void        (*EGLProc)(void);

typedef EGLDisplay  (*PFN_eglGetDisplay)(void *displayId);
typedef EGLBoolean  (*PFN_eglInitialize)(EGLDisplay, EGLint *, EGLint *);
typedef const char *(*PFN_eglQueryString)(EGLDisplay, EGLint);
typedef EGLBoolean  (*PFN_eglBindAPI)(EGLenum);
typedef EGLBoolean  (*PFN_eglChooseConfig)(EGLDisplay, const EGLint *, EGLConfig *, EGLint, EGLint *);
typedef EGLContext  (*PFN_eglCreateContext)(EGLDisplay, EGLConfig, EGLContext, const EGLint *);
typedef EGLBoolean  (*PFN_eglMakeCurrent)(EGLDisplay, EGLSurface, EGLSurface, EGLContext);
typedef EGLint      (*PFN_eglGetError)(void);
typedef EGLProc     (*PFN_eglGetProcAddress)(const char *);
typedef EGLBoolean  (*PFN_eglQueryDisplayAttribEXT)(EGLDisplay, EGLint, EGLAttrib *);
typedef EGLBoolean  (*PFN_eglQueryDeviceAttribEXT)(EGLDeviceEXT, EGLint, EGLAttrib *);

typedef EGLBoolean  (*PFN_eglMakeCurrent)(EGLDisplay, EGLSurface, EGLSurface, EGLContext);

// EGL state owned here (the context is created here and handed to librw).
static void      *g_libEGL    = NULL;
static void      *g_libGLESv2 = NULL;
static EGLDisplay g_display   = NULL;
static EGLContext g_context   = NULL;
// Kept so the game thread can claim the context and the init thread release it.
static PFN_eglMakeCurrent   g_eglMakeCurrent   = NULL;
static PFN_eglGetProcAddress g_eglGetProcAddress = NULL;
static PFN_eglGetError      g_eglGetError = NULL;
// ANGLE's own MTLDevice. The render-target MTLTexture MUST live on THIS device
// or eglCreateImageKHR fails (the single most common, silent interop mistake).
static id<MTLDevice>        g_mtlDevice = nil;

#define VCLOG(...) NSLog(@"[vc-gl] " __VA_ARGS__)

// dlopen an ANGLE framework from <App>.app/Frameworks/<name>.framework/<name>.
// The path is resolved at runtime; the static library can't know it at build time.
static void *
vc_dlopen_framework(const char *name)
{
	NSString *frameworks = [[NSBundle mainBundle] privateFrameworksPath];
	NSString *path = [NSString stringWithFormat:@"%@/%s.framework/%s", frameworks, name, name];
	void *handle = dlopen(path.fileSystemRepresentation, RTLD_NOW | RTLD_GLOBAL);
	if (handle == NULL) // fall back to a bare name (flat layout / already loaded)
		handle = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
	return handle;
}

// Brings up ANGLE and a surfaceless GLES 3.0 context, current on this thread.
// On success returns true and, via outGetProcAddress, ANGLE's eglGetProcAddress
// (as a void*) so librw can load its GL entry points. Logs every step.
extern "C" bool
vcgl_init_angle(void **outGetProcAddress)
{
	if (outGetProcAddress) *outGetProcAddress = NULL;

	// 1) Load both ANGLE frameworks from the app bundle.
	g_libEGL = vc_dlopen_framework("libEGL");
	if (g_libEGL == NULL) { VCLOG(@"FAIL dlopen libEGL: %s", dlerror()); return false; }
	VCLOG(@"dlopen libEGL OK");

	g_libGLESv2 = vc_dlopen_framework("libGLESv2");
	if (g_libGLESv2 == NULL) { VCLOG(@"FAIL dlopen libGLESv2: %s", dlerror()); return false; }
	VCLOG(@"dlopen libGLESv2 OK");

	// 2) Resolve the core EGL entry points.
	#define VC_SYM(var, T, name) \
		T var = (T)dlsym(g_libEGL, name); \
		if (var == NULL) { VCLOG(@"FAIL dlsym %s", name); return false; }
	VC_SYM(eglGetDisplay,    PFN_eglGetDisplay,     "eglGetDisplay");
	VC_SYM(eglInitialize,    PFN_eglInitialize,     "eglInitialize");
	VC_SYM(eglQueryString,   PFN_eglQueryString,    "eglQueryString");
	VC_SYM(eglBindAPI,       PFN_eglBindAPI,        "eglBindAPI");
	VC_SYM(eglChooseConfig,  PFN_eglChooseConfig,   "eglChooseConfig");
	VC_SYM(eglCreateContext, PFN_eglCreateContext,  "eglCreateContext");
	VC_SYM(eglMakeCurrent,   PFN_eglMakeCurrent,    "eglMakeCurrent");
	VC_SYM(eglGetError,      PFN_eglGetError,       "eglGetError");
	VC_SYM(eglGetProcAddress,PFN_eglGetProcAddress, "eglGetProcAddress");
	#undef VC_SYM
	VCLOG(@"dlsym core EGL entry points OK");

	// 3) Display + initialize.
	g_display = eglGetDisplay((void *)0 /* EGL_DEFAULT_DISPLAY */);
	if (g_display == NULL) { VCLOG(@"FAIL eglGetDisplay"); return false; }

	EGLint major = 0, minor = 0;
	if (!eglInitialize(g_display, &major, &minor)) {
		VCLOG(@"FAIL eglInitialize (egl error 0x%x)", eglGetError());
		return false;
	}
	VCLOG(@"eglInitialize OK, EGL %d.%d", major, minor);

	const char *vendor  = eglQueryString(g_display, VC_EGL_VENDOR);
	const char *version = eglQueryString(g_display, VC_EGL_VERSION);
	VCLOG(@"EGL vendor='%s' version='%s'", vendor ? vendor : "?", version ? version : "?");

	// 4) Bind ES, choose an ES3-capable RGBA8/D24 config, create a 3.0 context.
	eglBindAPI(VC_EGL_OPENGL_ES_API);

	const EGLint cfgAttribs[] = {
		VC_EGL_RENDERABLE_TYPE, VC_EGL_OPENGL_ES3_BIT,
		VC_EGL_RED_SIZE,   8,
		VC_EGL_GREEN_SIZE, 8,
		VC_EGL_BLUE_SIZE,  8,
		VC_EGL_ALPHA_SIZE, 8,
		VC_EGL_DEPTH_SIZE, 24,
		VC_EGL_NONE
	};
	EGLConfig config = NULL;
	EGLint numConfigs = 0;
	if (!eglChooseConfig(g_display, cfgAttribs, &config, 1, &numConfigs) || numConfigs < 1) {
		VCLOG(@"FAIL eglChooseConfig (egl error 0x%x)", eglGetError());
		return false;
	}
	VCLOG(@"eglChooseConfig OK (%d config(s))", numConfigs);

	const EGLint ctxAttribs[] = {
		VC_EGL_CONTEXT_MAJOR_VERSION, 3,
		VC_EGL_CONTEXT_MINOR_VERSION, 0,
		VC_EGL_NONE
	};
	g_context = eglCreateContext(g_display, config, (EGLContext)0 /* EGL_NO_CONTEXT */, ctxAttribs);
	if (g_context == NULL) {
		VCLOG(@"FAIL eglCreateContext (egl error 0x%x)", eglGetError());
		return false;
	}
	VCLOG(@"eglCreateContext (GLES 3.0) OK");

	// NOTE: we deliberately do NOT eglMakeCurrent here. The game thread is the
	// single owner of the context (vcgl_make_current_on_this_thread), so the
	// context never migrates between threads.

	// ANGLE's backing MTLDevice. The render-target texture MUST be created on
	// exactly this device (see g_mtlDevice comment). Does not need a current ctx.
	PFN_eglQueryDisplayAttribEXT eglQueryDisplayAttribEXT =
		(PFN_eglQueryDisplayAttribEXT)eglGetProcAddress("eglQueryDisplayAttribEXT");
	PFN_eglQueryDeviceAttribEXT eglQueryDeviceAttribEXT =
		(PFN_eglQueryDeviceAttribEXT)eglGetProcAddress("eglQueryDeviceAttribEXT");
	if (eglQueryDisplayAttribEXT && eglQueryDeviceAttribEXT) {
		EGLAttrib eglDevice = 0, mtl = 0;
		if (eglQueryDisplayAttribEXT(g_display, VC_EGL_DEVICE_EXT, &eglDevice) &&
		    eglQueryDeviceAttribEXT((EGLDeviceEXT)eglDevice, VC_EGL_METAL_DEVICE_ANGLE, &mtl) && mtl) {
		#if __has_feature(objc_arc)
			g_mtlDevice = (__bridge id<MTLDevice>)(void *)mtl;
		#else
			g_mtlDevice = (id<MTLDevice>)(void *)mtl;
		#endif
			VCLOG(@"ANGLE MTLDevice = '%@'", g_mtlDevice.name);
		} else {
			VCLOG(@"FAIL: could not query ANGLE MTLDevice (egl error 0x%x)", eglGetError());
		}
	} else {
		VCLOG(@"FAIL: eglQuery*AttribEXT unavailable");
	}

	// Keep these for the game-thread make-current and GL fn resolution.
	g_eglMakeCurrent    = eglMakeCurrent;
	g_eglGetProcAddress = eglGetProcAddress;
	g_eglGetError       = eglGetError;

	if (outGetProcAddress) *outGetProcAddress = (void *)eglGetProcAddress;
	VCLOG(@"ANGLE ready; handing eglGetProcAddress to librw");
	return true;
}

// An EGL context is current on exactly one thread at a time. psInitialize()
// creates+uses it on the init thread; the game thread must claim it before it
// renders, and the init thread must release it first.

// Release the context from the calling (init) thread. EGL_NO_SURFACE/CONTEXT = 0.
extern "C" void
vcgl_release_current(void)
{
	if (g_eglMakeCurrent && g_display)
		g_eglMakeCurrent(g_display, (EGLSurface)0, (EGLSurface)0, (EGLContext)0);
	VCLOG(@"released GL context from init thread");
}

// Claim the context on the calling (game) thread, surfaceless.
extern "C" bool
vcgl_make_current_on_this_thread(void)
{
	if (!g_eglMakeCurrent || !g_display || !g_context) {
		VCLOG(@"FAIL make current: EGL not initialised");
		return false;
	}
	if (!g_eglMakeCurrent(g_display, (EGLSurface)0, (EGLSurface)0, g_context)) {
		VCLOG(@"FAIL eglMakeCurrent on game thread");
		return false;
	}
	VCLOG(@"GL context current on game thread");
	return true;
}

// ANGLE's eglGetProcAddress as a plain void*, for resolving GL fns without glad.
extern "C" void *
vcgl_get_proc_address(void)
{
	return (void *)g_eglGetProcAddress;
}

// ===========================================================================
// Render target: an MTLTexture on ANGLE's device, wrapped as an EGLImage and
// bound to a GL texture + FBO, so reVC (via librw) renders into a Metal texture
// instead of the non-existent surfaceless default framebuffer.
// ===========================================================================

// GL/EGL constants as integer literals (no ANGLE/GL headers in the project).
enum {
	VC_GL_TEXTURE_2D             = 0x0DE1,
	VC_GL_FRAMEBUFFER            = 0x8D40,
	VC_GL_RENDERBUFFER           = 0x8D41,
	VC_GL_COLOR_ATTACHMENT0      = 0x8CE0,
	VC_GL_DEPTH_STENCIL_ATTACHMENT = 0x821A,
	VC_GL_DEPTH_ATTACHMENT       = 0x8D00,
	VC_GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE = 0x8CD0,
	VC_GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME = 0x8CD1,
	VC_GL_FRAMEBUFFER_COMPLETE   = 0x8CD5,
	VC_GL_TEXTURE_MAG_FILTER     = 0x2800,
	VC_GL_TEXTURE_MIN_FILTER     = 0x2801,
	VC_GL_TEXTURE_WRAP_S         = 0x2802,
	VC_GL_TEXTURE_WRAP_T         = 0x2803,
	VC_GL_NEAREST                = 0x2600,
	VC_GL_CLAMP_TO_EDGE          = 0x812F,
	VC_EGL_METAL_TEXTURE_ANGLE   = 0x34A7,
	// ANGLE_metal_texture_client_buffer, added v2 (2024-02-12): selects the
	// Metal texture-array slice to wrap. From Prototypes/angle-src eglext_angle.h.
	VC_EGL_METAL_TEXTURE_ARRAY_SLICE_ANGLE = 0x34DD,
	VC_GL_COLOR_BUFFER_BIT       = 0x4000,

	// EGL_ANGLE_metal_shared_event_sync. Values confirmed from Klepton
	// (shinyquagsire23/Klepton, runtime/gfx/kl_glfb.c, MIT, (c) 2026 Max Thomas).
	VC_EGL_SYNC_METAL_SHARED_EVENT_ANGLE                 = 0x34D8,
	VC_EGL_SYNC_METAL_SHARED_EVENT_OBJECT_ANGLE          = 0x34D9,
	VC_EGL_SYNC_METAL_SHARED_EVENT_SIGNAL_VALUE_LO_ANGLE = 0x34DA,
	VC_EGL_SYNC_METAL_SHARED_EVENT_SIGNAL_VALUE_HI_ANGLE = 0x34DB,
	VC_EGL_EXTENSIONS = 0x3055,
};

typedef unsigned int GLenumVC;
typedef unsigned int GLuintVC;
typedef int          GLintVC;
typedef int          GLsizeiVC;

typedef void     (*PFN_glGenTextures)(GLsizeiVC, GLuintVC *);
typedef void     (*PFN_glBindTexture)(GLenumVC, GLuintVC);
typedef void     (*PFN_glTexParameteri)(GLenumVC, GLenumVC, GLintVC);
typedef void     (*PFN_glGenFramebuffers)(GLsizeiVC, GLuintVC *);
typedef void     (*PFN_glBindFramebuffer)(GLenumVC, GLuintVC);
typedef void     (*PFN_glFramebufferTexture2D)(GLenumVC, GLenumVC, GLenumVC, GLuintVC, GLintVC);
typedef void     (*PFN_glFramebufferRenderbuffer)(GLenumVC, GLenumVC, GLenumVC, GLuintVC);
typedef GLenumVC (*PFN_glCheckFramebufferStatus)(GLenumVC);
typedef void     (*PFN_glGetFramebufferAttachmentParameteriv)(GLenumVC, GLenumVC, GLenumVC, GLintVC *);
typedef GLenumVC (*PFN_glGetError)(void);
typedef void     (*PFN_glFinish)(void);
typedef void     (*PFN_glFlush)(void);
typedef void     (*PFN_glEGLImageTargetTexture2DOES)(GLenumVC, void *);
typedef void *   (*PFN_eglCreateImageKHR)(EGLDisplay, EGLContext, EGLenum, void *, const EGLint *);
typedef void *   (*PFN_eglCreateSync)(EGLDisplay, EGLenum, const EGLAttrib *);
typedef unsigned int (*PFN_eglDestroySync)(EGLDisplay, void *);
typedef const char * (*PFN_eglQueryString)(EGLDisplay, EGLint);

typedef void (*PFN_glClearColorVC)(float, float, float, float);
typedef void (*PFN_glClearVC)(GLenumVC);

static PFN_glClearColorVC             p_glClearColor = NULL;
static PFN_glClearVC                  p_glClear = NULL;
static PFN_glGenTextures              p_glGenTextures = NULL;
static PFN_glBindTexture              p_glBindTexture = NULL;
static PFN_glTexParameteri            p_glTexParameteri = NULL;
static PFN_glGenFramebuffers          p_glGenFramebuffers = NULL;
static PFN_glBindFramebuffer          p_glBindFramebuffer = NULL;
static PFN_glFramebufferTexture2D     p_glFramebufferTexture2D = NULL;
static PFN_glFramebufferRenderbuffer  p_glFramebufferRenderbuffer = NULL;
static PFN_glCheckFramebufferStatus   p_glCheckFramebufferStatus = NULL;
static PFN_glGetFramebufferAttachmentParameteriv p_glGetFramebufferAttachmentParameteriv = NULL;
static PFN_glGetError                 p_glGetError = NULL;
static PFN_glFinish                   p_glFinish = NULL;
static PFN_glFlush                    p_glFlush = NULL;
static PFN_glEGLImageTargetTexture2DOES p_glEGLImageTargetTexture2DOES = NULL;
static PFN_eglCreateImageKHR          p_eglCreateImageKHR = NULL;
static PFN_eglCreateSync              p_eglCreateSync = NULL;
static PFN_eglDestroySync             p_eglDestroySync = NULL;
static PFN_eglQueryString             p_eglQueryString = NULL;

// Render-target buffer pool. VC_NUM_BUFFERS is the ARRAY CAPACITY; g_numBuffers is
// the count actually used (env VC_NUM_BUFFERS, default 3). With only 2, begin_frame
// blocks ~1 display frame every frame (one buffer ACQUIRED, one READY, none FREE)
// -> self-sustaining 45 Hz. A 3rd buffer gives reVC the missing lead so it doesn't
// wait on the compositor's release. Tunable so M2 vs M5 can pick their sweet spot.
#define VC_NUM_BUFFERS 4
static int g_numBuffers = 3;
static int g_frameCapHz = 0;   // >0 = fixed cap Hz; 0 = uncapped; -1 = auto (= display rate)
static int g_displayHz = 0;    // measured compositor rate (acquire-call frequency), 0 until known

// One bit ("busy") could not tell "finished but not yet fetched" from "in use by
// the compositor" -- and the second must never be reclaimed while Metal reads it.
// Four explicit states make that distinction and drive the throttle:
//   FREE       the game may render into it (only FREE is ever selected)
//   IN_FLIGHT  reserved by the game at begin_frame; GL is (or will be) drawing
//   READY      published; waiting for the compositor to acquire it
//   ACQUIRED   compositor holds it; freed only by vc_release_frame()
typedef enum {
	VC_BUF_FREE = 0,
	VC_BUF_IN_FLIGHT,
	VC_BUF_READY,
	VC_BUF_ACQUIRED,
} VCBufState;

typedef struct {
	id<MTLTexture> mtlTexture;
	void          *eglImage;
	GLuintVC       glTex;
	GLuintVC       glFbo;
	VCBufState     state;
	uint64_t       waitValue;  // shared-event value to wait for (0 = no wait)
	uint64_t       poseSetTime; // mach time the head pose of THIS slice was pushed (latency probe)
} VCBuffer;

static VCBuffer g_buf[VC_NUM_BUFFERS];
static int      g_rtWidth = 0, g_rtHeight = 0;
static bool     g_extActive   = false;   // render target ready (redirect armed)
static int      g_currentBack = 0;       // buffer the game renders into this frame
static bool     g_haveBack    = false;   // this frame actually reserved a buffer
static int      g_readyIndex  = -1;      // latest published (for the compositor)
static uint64_t g_signalValue = 0;       // monotonic shared-event counter
static uint64_t g_frameCount  = 0;       // successful publishes
static uint64_t g_waitCount   = 0;       // how often the game thread parked
static uint64_t g_discardCount = 0;      // phantom publishes suppressed by the gate
// Buffer strategy (VC_BUFFER_MODE): the ONLY difference between the two modes is
// whether vc_acquire_ready_frame recycles a superseded READY buffer to FREE.
//   false = "wait"   (default): don't recycle; the game waits for a real release,
//                    giving exactly the display rate with no discarded work.
//   true  = "newest" : recycle the older READY at acquire -> game runs ahead
//                    (~2x display rate here), compositor always sees the freshest.
static bool     g_recycleOlderReady = false;
static uint64_t g_recycleCount      = 0;  // READY->FREE recycles (0 in "wait")
static id<MTLSharedEvent>   g_sharedEvent = nil;
static id<MTLCommandQueue>  g_cmdQueue    = nil;
static bool     g_useSharedEvent = false; // extension present and not disabled
static bool     g_noFence        = false; // VC_NOFENCE=1
static void    *g_prevSync       = NULL;  // destroyed one frame later (gotcha d)

static pthread_mutex_t g_bufMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_bufCond  = PTHREAD_COND_INITIALIZER;

// Mirror of vc_ready_frame_t in AvpViceCity/VCPlatform.h -- MUST match layout.
typedef struct {
	void    *texture;
	uint32_t index;
	uint64_t wait_value;
	uint32_t width, height;
	uint32_t eye_count;   // 1 = mono 2D texture; 2 = stereo 2D-array (slice per eye)
	void    *hud_texture; // stereo only: 2D transparent HUD/2D/menu overlay; NULL in cinema
	uint64_t pose_set_time; // mach time of the head pose this frame was RENDERED with (reproj fix)
} vc_ready_frame_t;

// Defined in visionos.cpp; lets the throttle wake up for vc_game_thread_stop().
extern "C" bool vc_should_stop(void);
// Defined in visionos.cpp: push time of the head pose reVC last rendered with.
extern "C" uint64_t vc_last_consumed_pose_time(void);
// Defined in visionos.cpp: 1 = verbose perf logs enabled (VC_PERF_LOG).
extern "C" int vc_perf_log(void);

// Stereo accessors (defined further down with the stereo target). Used by
// vc_acquire_ready_frame to hand out the array texture + eye_count in stereo.
extern "C" int   vc_render_mode(void);          // 1 == VC_MODE_STEREO
extern "C" bool  vc_stereo_ready(void);
extern "C" void *vc_stereo_array_texture(int idx);
// Throttled (1/s) publish-side probe: reads back both slices of the just-
// published stereo buffer and logs the centre pixel, to tell "producer black"
// (eye passes wrote nothing) from "handoff black" (slices fine, display broke).
extern "C" void  vcrt_stereo_publish_probe(int idx);

static const char *
vcrt_fbo_status_name(GLenumVC s)
{
	switch (s) {
		case 0x8CD5: return "GL_FRAMEBUFFER_COMPLETE";
		case 0x8CD6: return "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
		case 0x8CD7: return "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
		case 0x8CD9: return "GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS";
		case 0x8CDD: return "GL_FRAMEBUFFER_UNSUPPORTED";
		case 0x8D56: return "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
		default:     return "GL_FRAMEBUFFER_<unknown>";
	}
}

static bool
vcrt_resolve(void)
{
	if (!g_eglGetProcAddress) { VCLOG(@"[vc-rt] FAIL: no eglGetProcAddress"); return false; }
	#define VC_GL(v, T, n) v = (T)g_eglGetProcAddress(n); if (!v) { VCLOG(@"[vc-rt] FAIL resolve %s", n); return false; }
	VC_GL(p_glClearColor,             PFN_glClearColorVC,           "glClearColor");
	VC_GL(p_glClear,                  PFN_glClearVC,                "glClear");
	VC_GL(p_glGenTextures,            PFN_glGenTextures,            "glGenTextures");
	VC_GL(p_glBindTexture,            PFN_glBindTexture,            "glBindTexture");
	VC_GL(p_glTexParameteri,          PFN_glTexParameteri,          "glTexParameteri");
	VC_GL(p_glGenFramebuffers,        PFN_glGenFramebuffers,        "glGenFramebuffers");
	VC_GL(p_glBindFramebuffer,        PFN_glBindFramebuffer,        "glBindFramebuffer");
	VC_GL(p_glFramebufferTexture2D,   PFN_glFramebufferTexture2D,   "glFramebufferTexture2D");
	VC_GL(p_glFramebufferRenderbuffer,PFN_glFramebufferRenderbuffer,"glFramebufferRenderbuffer");
	VC_GL(p_glCheckFramebufferStatus, PFN_glCheckFramebufferStatus, "glCheckFramebufferStatus");
	VC_GL(p_glGetFramebufferAttachmentParameteriv, PFN_glGetFramebufferAttachmentParameteriv, "glGetFramebufferAttachmentParameteriv");
	VC_GL(p_glGetError,               PFN_glGetError,               "glGetError");
	VC_GL(p_glFinish,                 PFN_glFinish,                 "glFinish");
	// gotcha (c): resolve glFlush directly from ANGLE, not via a maybe-empty ptr.
	VC_GL(p_glFlush,                  PFN_glFlush,                  "glFlush");
	VC_GL(p_glEGLImageTargetTexture2DOES, PFN_glEGLImageTargetTexture2DOES, "glEGLImageTargetTexture2DOES");
	VC_GL(p_eglCreateImageKHR,        PFN_eglCreateImageKHR,        "eglCreateImageKHR");
	#undef VC_GL

	// Core EGL 1.5 entry points: resolve straight from libEGL. eglCreateSync
	// (NOT eglCreateSyncKHR: KHR takes EGLint attribs, too narrow for the 64-bit
	// MTLSharedEvent pointer -- gotcha a).
	p_eglCreateSync  = (PFN_eglCreateSync)dlsym(g_libEGL, "eglCreateSync");
	p_eglDestroySync = (PFN_eglDestroySync)dlsym(g_libEGL, "eglDestroySync");
	p_eglQueryString = (PFN_eglQueryString)dlsym(g_libEGL, "eglQueryString");
	return true;
}

extern "C" void vcrt_readback_log(void);   // defined below; called from publish

static double
vc_now_seconds(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

// Seconds from mach_absolute_time (independent clock for the rate-window
// measurement, so the reported window duration can't be biased by the same
// source the throttle uses).
static double
vc_mach_seconds(void)
{
	static mach_timebase_info_data_t tb = { 0, 0 };
	if (tb.denom == 0) mach_timebase_info(&tb);
	unsigned __int128 ns = (unsigned __int128)mach_absolute_time() * tb.numer / tb.denom;
	return (double)(uint64_t)ns / 1.0e9;
}

// Create one buffer: MTLTexture (ANGLE device) -> EGLImage -> GL texture ->
// colour-only FBO. Depth is the shared renderbuffer that librw attaches later.
static bool
vcrt_make_buffer(int i, int width, int height)
{
	VCBuffer *b = &g_buf[i];

	// Verified params: RGBA8Unorm, ShaderRead|RenderTarget, StorageModePrivate,
	// on ANGLE's OWN device (mandatory for EGL_METAL_TEXTURE_ANGLE).
	MTLTextureDescriptor *desc =
		[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                   width:width height:height mipmapped:NO];
	desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
	desc.storageMode = MTLStorageModePrivate;
	b->mtlTexture = [g_mtlDevice newTextureWithDescriptor:desc];
	if (b->mtlTexture == nil) { VCLOG(@"[vc-rt] buffer %d FAIL: MTLTexture create", i); return false; }

	const EGLint imgAttribs[] = { VC_EGL_NONE };
	b->eglImage = p_eglCreateImageKHR(g_display, (EGLContext)0 /* EGL_NO_CONTEXT */,
		VC_EGL_METAL_TEXTURE_ANGLE, VC_OBJ_TO_VOID(b->mtlTexture), imgAttribs);
	if (b->eglImage == NULL) {
		VCLOG(@"[vc-rt] buffer %d FAIL eglCreateImageKHR (egl 0x%x). MOST COMMON CAUSE: "
		      "MTLTexture not on ANGLE's own MTLDevice.", i, g_eglGetError ? g_eglGetError() : 0);
		return false;
	}

	p_glGenTextures(1, &b->glTex);
	p_glBindTexture(VC_GL_TEXTURE_2D, b->glTex);
	p_glEGLImageTargetTexture2DOES(VC_GL_TEXTURE_2D, b->eglImage);
	GLenumVC bindErr = p_glGetError();
	if (bindErr != 0) { VCLOG(@"[vc-rt] buffer %d FAIL glEGLImageTargetTexture2DOES (gl 0x%x)", i, bindErr); return false; }
	p_glTexParameteri(VC_GL_TEXTURE_2D, VC_GL_TEXTURE_MAG_FILTER, VC_GL_NEAREST);
	p_glTexParameteri(VC_GL_TEXTURE_2D, VC_GL_TEXTURE_MIN_FILTER, VC_GL_NEAREST);
	p_glTexParameteri(VC_GL_TEXTURE_2D, VC_GL_TEXTURE_WRAP_S, VC_GL_CLAMP_TO_EDGE);
	p_glTexParameteri(VC_GL_TEXTURE_2D, VC_GL_TEXTURE_WRAP_T, VC_GL_CLAMP_TO_EDGE);

	p_glGenFramebuffers(1, &b->glFbo);
	p_glBindFramebuffer(VC_GL_FRAMEBUFFER, b->glFbo);
	p_glFramebufferTexture2D(VC_GL_FRAMEBUFFER, VC_GL_COLOR_ATTACHMENT0, VC_GL_TEXTURE_2D, b->glTex, 0);
	GLenumVC status = p_glCheckFramebufferStatus(VC_GL_FRAMEBUFFER);
	p_glBindFramebuffer(VC_GL_FRAMEBUFFER, 0);
	b->state = VC_BUF_FREE;
	b->waitValue = 0;
	VCLOG(@"[vc-rt] buffer %d: %dx%d RGBA8 on '%@', gl tex %u, FBO %u, colour-only status = %s",
	      i, width, height, g_mtlDevice.name, b->glTex, b->glFbo, vcrt_fbo_status_name(status));
	return true;
}

// Create the double-buffered render target + shared event. Game thread, GL
// context current. // TODO(visionos): width/height hardcoded; later from the
// CompositorServices drawable.
extern "C" bool
vcrt_create(int width, int height)
{
	if (g_mtlDevice == nil) { VCLOG(@"[vc-rt] FAIL: no ANGLE MTLDevice"); return false; }
	if (!vcrt_resolve()) return false;

	// STEREO defaults (each env-overridable below): 4 buffers + newest recycle + auto
	// frame-cap (= measured display rate) together break BOTH the 45 Hz buffer
	// starvation AND the 145 Hz free-run beat, without a phase-lock. Cinema keeps
	// wait + 2 buffers + no cap (half the GPU work; camera isn't head-locked there).
	bool stereo = (vc_render_mode() == 1);
	// 3 buffers (was 4): the 4th was against the 145-vs-90 beat, which the phase/cap work
	// removed; 3 keeps enough run-ahead for the newest-recycle. At high VC_RES each buffer
	// is large (slice array + HUD), so a buffer saved is ~85 MB. VC_NUM_BUFFERS overrides.
	if (stereo) { g_numBuffers = 3; g_recycleOlderReady = true; g_frameCapHz = -1; }

	// Buffer count (env VC_NUM_BUFFERS, clamped to the array capacity).
	const char *nb = getenv("VC_NUM_BUFFERS");
	if (nb) { int v = atoi(nb); if (v >= 2 && v <= VC_NUM_BUFFERS) g_numBuffers = v; }

	// Frame-rate cap (env VC_FRAME_CAP_HZ): >0 fixed, 0 off, negative keeps auto.
	const char *fc = getenv("VC_FRAME_CAP_HZ");
	if (fc) { int v = atoi(fc); if (v >= 0 && v <= 240) g_frameCapHz = v; }

	g_rtWidth = width;
	g_rtHeight = height;
	for (int i = 0; i < g_numBuffers; i++)
		if (!vcrt_make_buffer(i, width, height)) return false;

	// VC_NOFENCE=1 disables the wait entirely (publishes wait_value 0). A command
	// buffer that waits on a never-signalled value is committed and never runs:
	// the drawable presents, all counters look healthy, the screen stays black --
	// invisible from the Metal side, hence an explicit A/B switch.
	g_noFence = (getenv("VC_NOFENCE") != NULL);

	// Buffer strategy. Stereo defaults to "newest" (above); cinema to "wait" (no
	// recycle, exactly the display rate, half the GPU work). VC_BUFFER_MODE overrides.
	const char *mode = getenv("VC_BUFFER_MODE");
	if (mode) g_recycleOlderReady = (strcmp(mode, "newest") == 0);
	VCLOG(@"[vc-rt] buffer mode: %s (%d buffers, cap %s)", g_recycleOlderReady
	      ? "newest (recycle older READY at acquire -> game runs ahead)"
	      : "wait (game paces to real releases, no recycle)",
	      g_numBuffers, g_frameCapHz < 0 ? "auto=display" : (g_frameCapHz > 0 ? "fixed" : "off"));

	const char *exts = p_eglQueryString ? p_eglQueryString(g_display, VC_EGL_EXTENSIONS) : NULL;
	bool haveExt = exts && strstr(exts, "EGL_ANGLE_metal_shared_event_sync") != NULL;
	if (!haveExt)
		VCLOG(@"[vc-rt] EGL_EXTENSIONS = %s", exts ? exts : "(null)");

	if (!g_noFence && haveExt && p_eglCreateSync) {
		g_sharedEvent = [g_mtlDevice newSharedEvent];
		g_useSharedEvent = (g_sharedEvent != nil);
	}
	VCLOG(@"[vc-rt] sync: %s", g_noFence ? "DISABLED (VC_NOFENCE)"
	      : (g_useSharedEvent ? "EGL_ANGLE_metal_shared_event_sync"
	         : "FALLBACK glFlush + ANGLE queue ordering  // TODO(visionos)"));

	if (g_cmdQueue == nil) g_cmdQueue = [g_mtlDevice newCommandQueue];

	g_currentBack = 0;
	g_readyIndex  = -1;
	g_extActive   = true;
	VCLOG(@"[vc-rt] render target ready (%d buffers, %dx%d)",
	      g_numBuffers, width, height);
	return true;
}

// The redirect points at the CURRENT back buffer's FBO (chosen by
// vcrt_begin_frame). Both are read on the game thread, so no lock needed.
extern "C" bool vc_use_external_framebuffer(void)   { return g_extActive; }
extern "C" unsigned int vc_external_framebuffer(void) { return g_buf[g_currentBack].glFbo; }

// Attach ONE shared depth renderbuffer to BOTH FBOs. librw calls this the single
// time it attaches the camera's Z buffer; afterwards its "all good" fast path is
// correct because both FBOs share this depth. Re-runs only if librw recreates Z.
extern "C" void
vc_attach_depth_renderbuffer(unsigned int rbo)
{
	// Attach and then, for BOTH FBOs, read back what is ACTUALLY bound as the
	// depth attachment (OBJECT_NAME) plus the framebuffer status. If OBJECT_NAME
	// is 0 the FBO has no depth. We query the attachment (which ANGLE supports)
	// rather than the renderbuffer's DEPTH_SIZE (which ANGLE rejects with
	// GL_INVALID_ENUM for packed DEPTH24_STENCIL8).
	for (int i = 0; i < g_numBuffers; i++) {
		p_glBindFramebuffer(VC_GL_FRAMEBUFFER, g_buf[i].glFbo);
		p_glFramebufferRenderbuffer(VC_GL_FRAMEBUFFER, VC_GL_DEPTH_STENCIL_ATTACHMENT, VC_GL_RENDERBUFFER, rbo);
		GLenumVC status = p_glCheckFramebufferStatus(VC_GL_FRAMEBUFFER);

		GLintVC boundName = 0;
		if (p_glGetFramebufferAttachmentParameteriv)
			p_glGetFramebufferAttachmentParameteriv(VC_GL_FRAMEBUFFER,
				VC_GL_DEPTH_ATTACHMENT, VC_GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &boundName);

		VCLOG(@"[vc-rt] buffer %d FBO %u: colour+depth attach rbo %u -> bound depth OBJECT_NAME=%d (0 = NO depth) status = %s",
		      i, g_buf[i].glFbo, rbo, (int)boundName, vcrt_fbo_status_name(status));
	}
	p_glBindFramebuffer(VC_GL_FRAMEBUFFER, g_buf[g_currentBack].glFbo);   // restore caller's fbo
}

// Throttle: block (100 ms) until a back buffer is free, then select it. Called
// from psCameraBeginUpdate. false => timeout (park -> reVC skips the frame) or
// stop request. Two buffers pace the game thread to the compositor's
// acquire/release rate -- no second clock.
// Emit the publish/park/discard rate at most once per second. Called from EVERY
// frame attempt (both the park and the proceed path of vcrt_begin_frame), not
// only on a successful publish -- otherwise the line vanishes the moment the
// game parks, which is exactly when it's needed. Game-thread only: the counters
// and the static timer are all owned by this thread, so no lock is required.
static void
vcrt_log_rate(void)
{
	if (!vc_perf_log()) return;   // [vc-pub] is a verbose probe; gated behind VC_PERF_LOG
	static double   startT = 0.0, lastLog = 0.0;
	static uint64_t startN = 0, lastN = 0, lastW = 0, lastD = 0, lastR = 0;
	double now = vc_mach_seconds();
	if (lastLog == 0.0) {
		startT = lastLog = now;
		startN = lastN = g_frameCount; lastW = g_waitCount; lastD = g_discardCount; lastR = g_recycleCount;
		return;
	}
	double window = now - lastLog;
	if (window < 1.0) return;

	uint64_t pubs = g_frameCount - lastN;          // publishes in this window
	double   runtime = now - startT;               // since first rate log
	double   instRate = pubs / window;             // window-normalised (kills case a)
	double   longRate = runtime > 0.0 ? (double)(g_frameCount - startN) / runtime : 0.0;

	// window: actual duration of THIS counting window (should be ~1 s; if it's
	// e.g. 1.06 s while pubs=95, instRate is still ~90 -> case a).
	// total/runtime -> longRate: the true long-term rate. ~90 = case a, 95+ = case b.
	VCLOG(@"[vc-pub] readyIndex=%d pubs=%llu window=%.4fs instRate=%.1f/s parks=%llu discards=%llu recycles=%llu | total=%llu runtime=%.2fs longRate=%.2f/s",
	      g_readyIndex,
	      (unsigned long long)pubs, window, instRate,
	      (unsigned long long)(g_waitCount    - lastW),
	      (unsigned long long)(g_discardCount - lastD),
	      (unsigned long long)(g_recycleCount - lastR),
	      (unsigned long long)g_frameCount, runtime, longRate);

	lastLog = now; lastN = g_frameCount; lastW = g_waitCount; lastD = g_discardCount; lastR = g_recycleCount;
}

// ---- Eye-pass GPU timer (VC_EYE_GPU=1, default OFF) ------------------------
// The scene's GL work (both eye passes into the slices, ~2720x2624 x2 + MSAA) is
// submitted through ANGLE's own Metal command buffer, so we have no MTLCommandBuffer
// handle to time it -- and `eyes` only measures CPU submit, never the GPU fragment
// cost. ANGLE's Metal backend DOES implement GL_TIME_ELAPSED (QueryMtl), so we wrap
// the frame's GL draws (begin_frame proceed -> publish_frame) in a time-elapsed query.
// A small ring reads results a few frames later (async, no glFinish stall). This is
// the pre-foveation baseline the foveation win is measured against.
typedef unsigned long long GLuint64VC;
typedef void (*PFN_glGenQueries)(GLsizeiVC, GLuintVC *);
typedef void (*PFN_glBeginQuery)(GLenumVC, GLuintVC);
typedef void (*PFN_glEndQuery)(GLenumVC);
typedef void (*PFN_glGetQueryObjectuiv)(GLuintVC, GLenumVC, GLuintVC *);
typedef void (*PFN_glGetQueryObjectui64v)(GLuintVC, GLenumVC, GLuint64VC *);
static PFN_glGenQueries         p_glGenQueries = NULL;
static PFN_glBeginQuery         p_glBeginQuery = NULL;
static PFN_glEndQuery           p_glEndQuery = NULL;
static PFN_glGetQueryObjectuiv  p_glGetQueryObjectuiv = NULL;
static PFN_glGetQueryObjectui64v p_glGetQueryObjectui64v = NULL;
enum { VC_GL_TIME_ELAPSED = 0x88BF, VC_GL_QUERY_RESULT = 0x8866, VC_GL_QUERY_RESULT_AVAILABLE = 0x8867 };
#define VC_TQ_RING 4
static GLuintVC g_tq[VC_TQ_RING];
static bool     g_tqPending[VC_TQ_RING] = { false };
static int      g_tqWrite = 0, g_tqRead = 0, g_tqActiveSlot = -1;
static bool     g_tqOpen = false;
static bool     g_eyeGpuOn = false, g_eyeGpuReady = false, g_eyeGpuChecked = false;
static double   g_egSum = 0, g_egMin = 0, g_egMax = 0, g_egLast = 0, g_egLogT = 0;
static int      g_egN = 0;

static void
vcrt_eyegpu_ensure(void)
{
	if (g_eyeGpuChecked) return;
	g_eyeGpuChecked = true;
	const char *e = getenv("VC_EYE_GPU");
	g_eyeGpuOn = (e && e[0] == '1');
	if (!g_eyeGpuOn || !g_eglGetProcAddress) return;
	p_glGenQueries          = (PFN_glGenQueries)g_eglGetProcAddress("glGenQueries");
	p_glBeginQuery          = (PFN_glBeginQuery)g_eglGetProcAddress("glBeginQuery");
	p_glEndQuery            = (PFN_glEndQuery)g_eglGetProcAddress("glEndQuery");
	p_glGetQueryObjectuiv   = (PFN_glGetQueryObjectuiv)g_eglGetProcAddress("glGetQueryObjectuiv");
	p_glGetQueryObjectui64v = (PFN_glGetQueryObjectui64v)g_eglGetProcAddress("glGetQueryObjectui64vEXT");
	if (!p_glGetQueryObjectui64v)
		p_glGetQueryObjectui64v = (PFN_glGetQueryObjectui64v)g_eglGetProcAddress("glGetQueryObjectui64v");
	if (p_glGenQueries && p_glBeginQuery && p_glEndQuery && p_glGetQueryObjectuiv && p_glGetQueryObjectui64v) {
		p_glGenQueries(VC_TQ_RING, g_tq);
		g_eyeGpuReady = true;
		VCLOG(@"[vc-eyegpu] GL_TIME_ELAPSED timer armed (%d-deep ring)", VC_TQ_RING);
	} else {
		VCLOG(@"[vc-eyegpu] timer-query entry points unavailable -> disabled");
	}
}

static void
vcrt_eyegpu_poll(void)
{
	if (!g_eyeGpuReady) return;
	while (g_tqPending[g_tqRead]) {
		GLuintVC avail = 0;
		p_glGetQueryObjectuiv(g_tq[g_tqRead], VC_GL_QUERY_RESULT_AVAILABLE, &avail);
		if (!avail) break;
		GLuint64VC ns = 0;
		p_glGetQueryObjectui64v(g_tq[g_tqRead], VC_GL_QUERY_RESULT, &ns);
		g_tqPending[g_tqRead] = false;
		g_tqRead = (g_tqRead + 1) % VC_TQ_RING;
		double ms = (double)ns / 1.0e6;
		g_egSum += ms; g_egN++; g_egLast = ms;
		if (g_egN == 1 || ms < g_egMin) g_egMin = ms;
		if (ms > g_egMax) g_egMax = ms;
	}
	double now = vc_now_seconds();
	if (g_egN > 0 && now - g_egLogT >= 1.0) {
		g_egLogT = now;
		VCLOG(@"[vc-eyegpu] eye-pass GPU: avg=%.1f min=%.1f max=%.1f ms last=%.1f (n=%d)",
		      g_egSum / g_egN, g_egMin, g_egMax, g_egLast, g_egN);
		g_egSum = 0; g_egN = 0; g_egMin = 0; g_egMax = 0;
	}
}

static void
vcrt_eyegpu_close(void)  // end the open query, mark its slot for async readback
{
	if (!g_tqOpen) return;
	p_glEndQuery(VC_GL_TIME_ELAPSED);
	g_tqOpen = false;
	g_tqPending[g_tqActiveSlot] = true;
	g_tqWrite = (g_tqWrite + 1) % VC_TQ_RING;
}

static void
vcrt_eyegpu_begin(void)  // called on the proceed path, before any of the frame's draws
{
	vcrt_eyegpu_ensure();
	if (!g_eyeGpuReady) return;
	if (g_tqOpen) vcrt_eyegpu_close();   // stale (a reserved frame never published): balance it
	vcrt_eyegpu_poll();
	if (g_tqPending[g_tqWrite]) return;  // ring saturated -> skip timing this frame
	p_glBeginQuery(VC_GL_TIME_ELAPSED, g_tq[g_tqWrite]);
	g_tqOpen = true; g_tqActiveSlot = g_tqWrite;
}

static void vcrt_foveation_poll_dirty(void);   // defined with the stereo target below

extern "C" bool
vcrt_begin_frame(void)
{
	if (!g_extActive) return true;   // not set up yet; don't block

	// Frame-rate cap: sleep until the next frame slot so the game thread doesn't free-run
	// (~135 Hz) and waste ~1/3 of the GPU on discarded frames. capHz: fixed (>0), off
	// (0 = VC_FRAME_CAP_HZ=0), or auto (-1 = measured display rate, never hardcode 90).
	// This is a simple RATE cap on reVC's own clock. It does NOT phase-lock to the
	// compositor slots, so reVC's publish still drifts slowly through the slot window and a
	// frame occasionally sits one slot longer -- but that no longer shows, because the
	// reprojection fix (VC_REPROJ_FIX) reports each slice's true render pose, so the
	// compositor reprojects the slightly-older frame correctly instead of doubling it. (The
	// phase-locked cap "A" -- g_lastAcqSec grid, VC_CAP_LEAD_MS/GUARD_MS -- was removed as
	// symptom treatment once the fix addressed the cause; on device: no visible difference,
	// and it dropped the recycles the free-run wastes.)
	int capHz = (g_frameCapHz > 0) ? g_frameCapHz
	          : (g_frameCapHz < 0) ? (g_displayHz > 0 ? g_displayHz : 90)
	          : 0;
	if (capHz > 0) {
		static double nextSlot = 0.0;
		double period = 1.0 / (double)capHz;
		double now = vc_now_seconds();
		if (nextSlot == 0.0) nextSlot = now;
		if (now < nextSlot) { usleep((useconds_t)((nextSlot - now) * 1.0e6)); now = vc_now_seconds(); }
		nextSlot = (now > nextSlot + period) ? now + period : nextSlot + period;
	}

	vcrt_log_rate();   // once per frame ATTEMPT (park or proceed), see above

	// Latency probe: how long this call BLOCKS waiting for a FREE back buffer. With
	// 2 buffers + a compositor that shows each one twice (reVC 45 Hz vs 90 Hz), reVC
	// stalls here for a whole display frame -- a self-sustaining 45 Hz pacing. High
	// here == buffer starvation, not render cost. Throttled, stereo only.
	double vcBeginT0 = vc_now_seconds();

	struct timespec deadline;
	clock_gettime(CLOCK_REALTIME, &deadline);
	long ns = deadline.tv_nsec + 100L * 1000 * 1000;   // 100 ms
	deadline.tv_sec += ns / 1000000000L;
	deadline.tv_nsec = ns % 1000000000L;

	pthread_mutex_lock(&g_bufMutex);

	// Roll back a reservation that never got published (e.g. RwCameraBeginUpdate
	// failed after we reserved, or a gated caller skipped ShowRaster): return that
	// still-IN_FLIGHT buffer to FREE so it isn't leaked. If it WAS published its
	// state is READY/ACQUIRED and g_haveBack is already false, so this is a no-op.
	if (g_haveBack) {
		g_buf[g_currentBack].state = VC_BUF_FREE;
		g_haveBack = false;
	}

	int idx = -1;
	for (;;) {
		for (int i = 0; i < g_numBuffers; i++) if (g_buf[i].state == VC_BUF_FREE) { idx = i; break; }
		if (idx >= 0) break;
		if (vc_should_stop()) { pthread_mutex_unlock(&g_bufMutex); return false; }
		int rc = pthread_cond_timedwait(&g_bufCond, &g_bufMutex, &deadline);
		if (rc == ETIMEDOUT) {
			uint64_t parks = ++g_waitCount;
			pthread_mutex_unlock(&g_bufMutex);
			// First four parks individually so the publish->park transition is
			// visible frame by frame; the per-second rate above covers the rest.
			if (parks <= 4)
				VCLOG(@"[vc-pub] PARK %llu: no FREE back buffer (both READY/ACQUIRED) -- skip frame",
				      (unsigned long long)parks);
			return false;   // park: skip this frame, retry next iteration
		}
	}
	// Reserve at selection: FREE -> IN_FLIGHT. Only FREE is ever picked, so a
	// READY-but-unfetched or an ACQUIRED (compositor-owned) buffer is never taken.
	g_buf[idx].state = VC_BUF_IN_FLIGHT;
	g_currentBack = idx;
	g_haveBack = true;
	pthread_mutex_unlock(&g_bufMutex);
	if (vc_perf_log() && vc_render_mode() == 1) {
		static double lastBeginLog = 0.0;
		double nowS = vc_now_seconds();
		if (nowS - lastBeginLog >= 1.0) {
			lastBeginLog = nowS;
			VCLOG(@"[vc-begin] wait for FREE buffer = %.2f ms (high = buffer starvation -> 45 Hz pacing)",
			      (nowS - vcBeginT0) * 1000.0);
		}
	}
	// Apply a freshly pushed optical curve (from Swift) on THIS game thread, so the map
	// build + slice registration never race the render thread. Normally the curve arrives
	// long before stereo_ensure, which already built from it; this covers a late push.
	vcrt_foveation_poll_dirty();

	vcrt_eyegpu_begin();   // time this frame's GL (eye) passes (VC_EYE_GPU=1)
	return true;
}

// Clear the current back buffer's cinema/HUD colour to fully transparent. Stereo only.
// In stereo the world lives in the eye slices; this g_buf texture is published as the
// transparent HUD/2D overlay (hud_texture). librw's CLEARMODE clears only Z/stencil, so
// the colour is never wiped -- normally fine (gameplay only draws small HUD sprites, the
// rest stays transparent from allocation). But a load draws MessageScreen("FELD_WR") with
// an OPAQUE full-screen black background into one ring buffer; with no colour clear it
// stays baked there forever and flickers back in as the ring cycles. Clearing to (0,0,0,0)
// once per frame, right after vc_stereo_restore_main() and before the 2D/HUD pass, wipes
// any such stale content while leaving the freshly-drawn HUD intact.
extern "C" void
vc_hud_clear_transparent(void)
{
	if (!g_extActive || vc_render_mode() != 1) return;
	if (!p_glBindFramebuffer || !p_glClearColor || !p_glClear) return;
	p_glBindFramebuffer(VC_GL_FRAMEBUFFER, g_buf[g_currentBack].glFbo);
	p_glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	p_glClear((GLenumVC)VC_GL_COLOR_BUFFER_BIT);
}

// Enqueue a GPU signal of the shared event to `value` after the frame's GL work.
// Returns the EGLSync (destroyed one frame later) or NULL. This one seam carries
// the EGL_ANGLE_metal_shared_event_sync details.
static void *
vcrt_enqueue_signal(uint64_t value)
{
	if (g_useSharedEvent && p_eglCreateSync && g_sharedEvent) {
		// Mirrors Klepton kl_glfb.c::klfb_gpu_frame_now(): attach the MTLSharedEvent
		// and split the 64-bit signal value into LO/HI halves (EGLAttrib is a signed
		// intptr_t; passing the whole value would misbehave on the HI dword).
		uintptr_t evPtr = (uintptr_t)VC_OBJ_TO_VOID(g_sharedEvent);
		const EGLAttrib attribs[] = {
			VC_EGL_SYNC_METAL_SHARED_EVENT_OBJECT_ANGLE,          (EGLAttrib)evPtr,
			VC_EGL_SYNC_METAL_SHARED_EVENT_SIGNAL_VALUE_LO_ANGLE, (EGLAttrib)(value & 0xFFFFFFFFu),
			VC_EGL_SYNC_METAL_SHARED_EVENT_SIGNAL_VALUE_HI_ANGLE, (EGLAttrib)(value >> 32),
			VC_EGL_NONE
		};
		// gotcha (a): eglCreateSync, NOT eglCreateSyncKHR (EGLAttrib, not EGLint).
		void *sync = p_eglCreateSync(g_display, VC_EGL_SYNC_METAL_SHARED_EVENT_ANGLE, attribs);
		// gotcha (b): flush so the GPU actually reaches and signals the event.
		// A glFinish would be the very stall this path avoids.
		if (p_glFlush) p_glFlush();
		return sync;
	}

	// Fallback: no shared event -- submit the work, rely on ANGLE queue ordering.
	if (p_glFlush) p_glFlush();   // TODO(visionos): real cross-API sync needs the ext
	return NULL;
}

// Publish the buffer just rendered. Called from psCameraShowRaster (game thread,
// right after the frame's last GL draw).
extern "C" void
vcrt_publish_frame(void)
{
	if (!g_extActive) return;

	// Gate: only publish if THIS frame actually reserved a back buffer via
	// vcrt_begin_frame. reVC has many paths that call ShowRaster without checking
	// whether BeginUpdate succeeded (e.g. main.cpp:1964); on a park those would
	// otherwise re-publish the stale current-back. Suppress and count them.
	pthread_mutex_lock(&g_bufMutex);
	bool have = g_haveBack;
	int  back = g_currentBack;
	if (!have) {
		uint64_t d = ++g_discardCount;
		pthread_mutex_unlock(&g_bufMutex);
		if (d <= 4)
			VCLOG(@"[vc-pub] DISCARD %llu: ShowRaster with no reserved back buffer (ungated reVC path)",
			      (unsigned long long)d);
		return;
	}
	pthread_mutex_unlock(&g_bufMutex);

	vcrt_eyegpu_close();   // end the eye-pass timer opened in vcrt_begin_frame
	vcrt_eyegpu_poll();

	// The frame's GL work is recorded into g_buf[back]'s FBO. Enqueue the GPU
	// signal (or flush) BEFORE taking the lock -- EGL/GL work must not hold it.
	// Latency probe: time the signal/flush. glFlush submits the queued GL commands;
	// if the GPU is behind (backpressure) this BLOCKS until the driver's command
	// queue drains -- the "finish" cost that appears only under head-look (dynamic
	// camera => heavier/steadier GPU work). Throttled, stereo only.
	double vcPubT0 = vc_now_seconds();
	uint64_t value = 0;
	void *sync = NULL;
	if (!g_noFence) {
		value = ++g_signalValue;
		sync = vcrt_enqueue_signal(value);
	} else {
		if (p_glFlush) p_glFlush();
	}
	if (vc_perf_log() && vc_render_mode() == 1) {
		static double lastPubLog = 0.0;
		double nowS = vc_now_seconds();
		if (nowS - lastPubLog >= 1.0) {
			lastPubLog = nowS;
			VCLOG(@"[vc-publish] signal/flush = %.2f ms (GPU submit; high = GPU backpressure)",
			      (nowS - vcPubT0) * 1000.0);
		}
	}

	pthread_mutex_lock(&g_bufMutex);
	// gotcha (d): destroy the PREVIOUS frame's sync now, one frame later.
	if (g_prevSync && p_eglDestroySync) p_eglDestroySync(g_display, g_prevSync);
	g_prevSync = sync;

	g_buf[back].state = VC_BUF_READY;         // IN_FLIGHT -> READY
	g_buf[back].waitValue = value;            // 0 under VC_NOFENCE -> compositor won't wait
	g_buf[back].poseSetTime = vc_last_consumed_pose_time();  // latency probe: pose age of this slice
	g_readyIndex = back;
	g_haveBack = false;                        // reservation consumed
	uint64_t n = ++g_frameCount;
	pthread_mutex_unlock(&g_bufMutex);

	// First four publishes individually so the index rotation 0,1,0,1 is
	// provable even when only two frames ever publish (no consumer yet).
	if (n <= 4)
		VCLOG(@"[vc-pub] PUBLISH %llu: readyIndex=%d waitValue=%llu",
		      (unsigned long long)n, g_readyIndex, (unsigned long long)value);

	if (n == 2 && vc_perf_log()) vcrt_readback_log();   // one-time stereo self-test (GL/Metal cost) -- diagnostic only

	// Stereo diagnostic: read back the buffer we just published (throttled) so we
	// see whether the eye passes actually wrote content into what the compositor
	// will acquire. Gated behind VC_PERF_LOG: it allocates two render-target-sized
	// staging textures per call, and with no draining autorelease pool on the game
	// thread that was the render-target-sized ~2*W*H*4 B/s memory growth to jetsam.
	if (vc_perf_log() && vc_render_mode() == 1 && vc_stereo_ready())
		vcrt_stereo_publish_probe(back);
}

// Memory probe: phys_footprint = what this process currently uses; os_proc_available_memory
// = headroom left before jetsam kills us ("Terminated due to memory issue"). Throttled 2 s.
static void vc_log_memory(void)
{
	if (!vc_perf_log()) return;   // diagnostic; off by default (leak is fixed, flip VC_PERF_LOG to watch)
	static double last = 0.0; double now = vc_now_seconds();
	if (now - last < 2.0) return; last = now;
	uint64_t foot = 0;
	task_vm_info_data_t info; mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
	if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &cnt) == KERN_SUCCESS)
		foot = info.phys_footprint;
	size_t avail = os_proc_available_memory();
	// Split the growth: currentAllocatedSize = the ANGLE Metal device's live texture/buffer
	// bytes. If THIS climbs with footprint -> a GPU resource leak (textures/FBOs/EGLImages);
	// if it stays flat while footprint climbs -> a CPU/EGL-side leak (sync listeners etc.).
	unsigned long long metalMB = g_mtlDevice ? (unsigned long long)(g_mtlDevice.currentAllocatedSize / 1000000u) : 0;
	VCLOG(@"[vc-mem] footprint=%llu MB  metalAlloc=%llu MB  available-before-jetsam=%zu MB",
	      (unsigned long long)(foot / 1000000u), metalMB, avail / 1000000u);
}

// --- C seam consumed by the compositor (next step) -------------------------
extern "C" bool
vc_acquire_ready_frame(vc_ready_frame_t *out)
{
	if (!out) return false;
	vc_log_memory();

	// Measure the DISPLAY RATE from the acquire-call frequency: the compositor calls
	// this once per display frame, so its period is the refresh period. Feeds the
	// auto frame-cap (g_frameCapHz == -1) so we never hardcode 90 (M2 vs M5 differ).
	{
		static uint64_t sNum = 0, sDen = 0;
		if (sDen == 0) { mach_timebase_info_data_t tb; mach_timebase_info(&tb); sNum = tb.numer; sDen = tb.denom; }
		static uint64_t sLastAcq = 0;
		uint64_t nowT = mach_absolute_time();
		if (sLastAcq != 0) {
			double periodMs = (double)(nowT - sLastAcq) * (double)sNum / (double)sDen / 1.0e6;
			if (periodMs > 4.0 && periodMs < 40.0) {   // 25-250 Hz sane
				static double ewma = 0.0;
				double hz = 1000.0 / periodMs;
				ewma = (ewma == 0.0) ? hz : ewma * 0.9 + hz * 0.1;
				g_displayHz = (int)(ewma + 0.5);
			}
		}
		sLastAcq = nowT;
	}

	pthread_mutex_lock(&g_bufMutex);
	int idx = g_readyIndex;
	// Nothing new to hand out: never published, or the latest was already acquired
	// (and not yet republished). The compositor keeps its last texture -- a MISSED
	// SLOT (the same frame is shown twice). Kept on always (throttled 10 s): first
	// signal of a future perf regression.
	static uint64_t sAcqTotal = 0, sAcqMiss = 0;
	sAcqTotal++;
	if (idx < 0 || g_buf[idx].state != VC_BUF_READY) {
		sAcqMiss++;
		pthread_mutex_unlock(&g_bufMutex);
		if (vc_render_mode() == 1) {
			static double lastMissLog = 0.0;
			double nowS = vc_now_seconds();
			if (nowS - lastMissLog >= 10.0) {
				lastMissLog = nowS;
				VCLOG(@"[vc-miss] missed display slots: %llu / %llu acquires (%.1f%%) -- compositor re-showed a frame",
				      (unsigned long long)sAcqMiss, (unsigned long long)sAcqTotal,
				      100.0 * (double)sAcqMiss / (double)(sAcqTotal ? sAcqTotal : 1));
			}
		}
		return false;
	}
	// "newest" ONLY: if the OTHER buffer is still sitting in READY (a superseded
	// frame the compositor never fetched), recycle it straight to FREE so the
	// game runs ahead. In "wait" (default) we skip this, so the game only gets a
	// buffer back on a real vc_release_frame -> it paces to the display rate.
	// Never touch an ACQUIRED buffer -- Metal may still be reading it.
	if (g_recycleOlderReady)
		for (int i = 0; i < g_numBuffers; i++)
			if (i != idx && g_buf[i].state == VC_BUF_READY) {
				g_buf[i].state = VC_BUF_FREE;
				g_recycleCount++;
				pthread_cond_signal(&g_bufCond);
			}
	g_buf[idx].state = VC_BUF_ACQUIRED;        // READY -> ACQUIRED (won't be reused)
	// Stereo: hand out the 2-slice array texture for this index and eye_count=2;
	// cinema: the plain 2D texture and eye_count=1. The index/state/wait_value
	// machinery is shared, so the shared-event sync covers whichever texture the
	// eye/cinema passes wrote into this buffer.
	bool stereo = (vc_render_mode() == 1) && vc_stereo_ready();
	out->texture    = stereo ? vc_stereo_array_texture(idx) : VC_OBJ_TO_VOID(g_buf[idx].mtlTexture);
	out->index      = (uint32_t)idx;
	out->wait_value = g_buf[idx].waitValue;
	out->width      = (uint32_t)g_rtWidth;
	out->height     = (uint32_t)g_rtHeight;
	out->eye_count  = stereo ? 2u : 1u;
	// Stereo: also hand out the cinema 2D texture as the transparent HUD overlay
	// (same index, written by the same GL stream, so the one shared-event covers it).
	// Cinema: no separate HUD layer.
	out->hud_texture = stereo ? VC_OBJ_TO_VOID(g_buf[idx].mtlTexture) : NULL;
	out->pose_set_time = g_buf[idx].poseSetTime;   // render pose of THIS buffer (reproj fix)
	void    *outTex = out->texture;
	uint64_t outWait = out->wait_value;
	uint64_t poseSetT = g_buf[idx].poseSetTime;
	pthread_mutex_unlock(&g_bufMutex);

	// Latency probe: full push -> acquire (~display) age of THIS slice's head pose.
	// Push->consume is measured on the game side ([vc-pose-age]); this adds the
	// consume->publish->acquire legs. Large here vs small there => the delay sits in
	// the render/publish/buffering, not the pose hand-off. Throttled, stereo only.
	if (vc_perf_log() && stereo && poseSetT != 0) {
		static uint64_t sNum = 0, sDen = 0;
		if (sDen == 0) { mach_timebase_info_data_t tb; mach_timebase_info(&tb); sNum = tb.numer; sDen = tb.denom; }
		double ageMs = (double)(mach_absolute_time() - poseSetT) * (double)sNum / (double)sDen / 1.0e6;
		// The judder is push->acquire JITTER (occasional +1 frame ~= 2x period), invisible
		// in a mean. Bucket EVERY acquire over an interval and report the distribution:
		// median (p50), p95, max, and the fraction of +1-frame outliers (> 1.5*period).
		// That is the number A must drive down -- steady latency beats low-but-jittery.
		static uint32_t hist[64];   // 1 ms buckets, 0..63 (clamped)
		static uint64_t hn = 0, hout = 0;
		static double hmax = 0.0, lastLatLog = 0.0;
		double nowS = vc_now_seconds();
		int b = (int)ageMs; if (b < 0) b = 0; if (b > 63) b = 63;
		hist[b]++; hn++;
		if (ageMs > hmax) hmax = ageMs;
		double periodMs = 1000.0 / (double)(g_displayHz > 0 ? g_displayHz : 90);
		if (ageMs > 1.5 * periodMs) hout++;
		if (lastLatLog == 0.0) lastLatLog = nowS;
		if (nowS - lastLatLog >= 2.0 && hn > 0) {
			lastLatLog = nowS;
			uint64_t acc = 0; int p50 = 0, p95 = 0; bool g50 = false, g95 = false;
			for (int i = 0; i < 64; i++) {
				acc += hist[i];
				if (!g50 && acc * 100 >= hn * 50) { p50 = i; g50 = true; }
				if (!g95 && acc * 100 >= hn * 95) { p95 = i; g95 = true; }
			}
			VCLOG(@"[vc-pose-jitter] n=%llu p50=%d ms p95=%d ms max=%.1f ms  +1frame(>%.0fms)=%llu (%.1f%%)",
			      (unsigned long long)hn, p50, p95, hmax,
			      1.5 * periodMs, (unsigned long long)hout, 100.0 * (double)hout / (double)hn);
			for (int i = 0; i < 64; i++) hist[i] = 0;
			hn = 0; hout = 0; hmax = 0.0;
		}
	}

	// Throttled handoff proof: which index / texture / wait the compositor is
	// handed. Compare against [vc-stereo] PUBLISH -- idx and tex MUST match, and
	// wait must be a sane increasing value.
	if (stereo) {
		static double lastAcqLog = 0.0;
		double now = vc_now_seconds();
		if (now - lastAcqLog >= 1.0) {
			lastAcqLog = now;
			VCLOG(@"[vc-stereo] ACQUIRE idx=%d tex=%p wait=%llu eye_count=%u",
			      idx, outTex, (unsigned long long)outWait, out->eye_count);
		}
	}
	return true;
}

extern "C" void
vc_release_frame(uint32_t index)
{
	if (index >= g_numBuffers) return;
	pthread_mutex_lock(&g_bufMutex);
	// Only an ACQUIRED buffer returns to the pool; ignore stray/duplicate
	// releases so we can't accidentally free a buffer the game is rendering into.
	if (g_buf[index].state == VC_BUF_ACQUIRED) {
		g_buf[index].state = VC_BUF_FREE;
		pthread_cond_signal(&g_bufCond);   // wake the game thread waiting for a buffer
	}
	pthread_mutex_unlock(&g_bufMutex);
}

extern "C" void *
vc_get_shared_event(void)
{
	return VC_OBJ_TO_VOID(g_sharedEvent);   // NULL if the fallback path is in use
}

// One-time proof that pixels really land in the shared MTLTexture: Metal-blit
// the render target into a shared staging texture and log a few pixels. NOT
// glReadPixels -- that would only confirm GL's own view.
extern "C" void
vcrt_readback_log(void)
{
	static bool done = false;
	if (done) return;
	done = true;

	id<MTLTexture> src = g_buf[0].mtlTexture;
	if (src == nil || g_mtlDevice == nil) { VCLOG(@"[vc-rt] readback: no texture"); return; }

	// Make sure the GL commands that produced the frame are submitted to ANGLE's
	// Metal queue before the blit reads the texture.
	if (p_glFinish) p_glFinish();

	MTLTextureDescriptor *sd =
		[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                   width:g_rtWidth
		                                                  height:g_rtHeight
		                                               mipmapped:NO];
	sd.usage = MTLTextureUsageShaderRead;
	sd.storageMode = MTLStorageModeShared;
	id<MTLTexture> staging = [g_mtlDevice newTextureWithDescriptor:sd];
	if (staging == nil) { VCLOG(@"[vc-rt] readback: staging alloc failed"); return; }

	if (g_cmdQueue == nil) g_cmdQueue = [g_mtlDevice newCommandQueue];
	id<MTLCommandBuffer> cb = [g_cmdQueue commandBuffer];
	id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
	[blit copyFromTexture:src
	          sourceSlice:0 sourceLevel:0
	         sourceOrigin:MTLOriginMake(0, 0, 0)
	           sourceSize:MTLSizeMake(g_rtWidth, g_rtHeight, 1)
	            toTexture:staging
	     destinationSlice:0 destinationLevel:0
	    destinationOrigin:MTLOriginMake(0, 0, 0)];
	[blit endEncoding];
	[cb commit];
	[cb waitUntilCompleted];

	// Sample the centre and one off-centre point.
	uint8_t c[4] = {0,0,0,0}, q[4] = {0,0,0,0};
	[staging getBytes:c bytesPerRow:4
	       fromRegion:MTLRegionMake2D(g_rtWidth/2, g_rtHeight/2, 1, 1) mipmapLevel:0];
	[staging getBytes:q bytesPerRow:4
	       fromRegion:MTLRegionMake2D(g_rtWidth/4, g_rtHeight/4, 1, 1) mipmapLevel:0];
	VCLOG(@"[vc-rt] readback via Metal blit: centre RGBA=%d,%d,%d,%d  quarter RGBA=%d,%d,%d,%d",
	      c[0], c[1], c[2], c[3], q[0], q[1], q[2], q[3]);
}

// VC_SLICE_TEST: prove (via read-back, not a return value) whether ANGLE can
// wrap a single slice of a Metal 2D-array texture as an EGLImage that GL renders
// into. Four cases; each slice cleared to a unique colour (slice 0 red, slice 1
// green), then Metal-blitted back and the real pixel logged. Two identical
// colours back for an array = both wrote slice 0 = silent failure. Test textures
// are created on ANGLE's device and freed afterwards; the normal path is
// untouched.
extern "C" void
vcrt_slice_test(void)
{
	if (g_mtlDevice == nil || !p_eglCreateImageKHR || !g_eglGetProcAddress) {
		VCLOG(@"[vc-slice] SKIP: prerequisites missing"); return;
	}
	// Test-only entry points, resolved locally so the main table stays untouched.
	typedef void (*PFN_glClearColor)(float,float,float,float);
	typedef void (*PFN_glClear)(GLenumVC);
	typedef void (*PFN_glViewport)(GLintVC,GLintVC,GLsizeiVC,GLsizeiVC);
	typedef void (*PFN_glDeleteTextures)(GLsizeiVC, const GLuintVC*);
	typedef void (*PFN_glDeleteFramebuffers)(GLsizeiVC, const GLuintVC*);
	typedef unsigned int (*PFN_eglDestroyImageKHR)(EGLDisplay, void*);
	PFN_glClearColor         glClearColor_         = (PFN_glClearColor)g_eglGetProcAddress("glClearColor");
	PFN_glClear              glClear_              = (PFN_glClear)g_eglGetProcAddress("glClear");
	PFN_glViewport           glViewport_           = (PFN_glViewport)g_eglGetProcAddress("glViewport");
	PFN_glDeleteTextures     glDeleteTextures_     = (PFN_glDeleteTextures)g_eglGetProcAddress("glDeleteTextures");
	PFN_glDeleteFramebuffers glDeleteFramebuffers_ = (PFN_glDeleteFramebuffers)g_eglGetProcAddress("glDeleteFramebuffers");
	PFN_eglDestroyImageKHR   eglDestroyImageKHR_   = (PFN_eglDestroyImageKHR)g_eglGetProcAddress("eglDestroyImageKHR");
	if (!glClearColor_ || !glClear_ || !glViewport_) {
		VCLOG(@"[vc-slice] SKIP: could not resolve gl clear/viewport"); return;
	}

	const int W = 4, H = 4;
	if (g_cmdQueue == nil) g_cmdQueue = [g_mtlDevice newCommandQueue];

	struct { const char *name; MTLPixelFormat fmt; bool isArray; } cases[4] = {
		{"a 2D      RGBA8Unorm ", MTLPixelFormatRGBA8Unorm,  false},
		{"b 2DArray RGBA8Unorm ", MTLPixelFormatRGBA8Unorm,  true },
		{"c 2D      RGBA16Float", MTLPixelFormatRGBA16Float, false},
		{"d 2DArray RGBA16Float", MTLPixelFormatRGBA16Float, true },
	};

	for (int ci = 0; ci < 4; ci++) {
		int slices = cases[ci].isArray ? 2 : 1;

		MTLTextureDescriptor *td = [[MTLTextureDescriptor alloc] init];
		td.pixelFormat = cases[ci].fmt;
		td.width = W; td.height = H;
		td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
		td.storageMode = MTLStorageModePrivate;
		if (cases[ci].isArray) { td.textureType = MTLTextureType2DArray; td.arrayLength = 2; }
		else                   { td.textureType = MTLTextureType2D; }
		id<MTLTexture> tex = [g_mtlDevice newTextureWithDescriptor:td];
		if (tex == nil) { VCLOG(@"[vc-slice] %s: MTLTexture alloc FAILED", cases[ci].name); continue; }

		for (int s = 0; s < slices; s++) {
			const EGLint attrsArr[] = { (EGLint)VC_EGL_METAL_TEXTURE_ARRAY_SLICE_ANGLE, (EGLint)s, (EGLint)VC_EGL_NONE };
			const EGLint attrs2D[]  = { (EGLint)VC_EGL_NONE };
			void *img = p_eglCreateImageKHR(g_display, (EGLContext)0, VC_EGL_METAL_TEXTURE_ANGLE,
			                                VC_OBJ_TO_VOID(tex), cases[ci].isArray ? attrsArr : attrs2D);
			if (img == NULL) {
				VCLOG(@"[vc-slice] %s slice %d: eglCreateImageKHR=NULL eglErr=0x%x",
				      cases[ci].name, s, g_eglGetError ? g_eglGetError() : 0);
				continue;
			}

			GLuintVC gltex = 0, glfbo = 0;
			p_glGenTextures(1, &gltex);
			p_glBindTexture(VC_GL_TEXTURE_2D, gltex);
			p_glEGLImageTargetTexture2DOES(VC_GL_TEXTURE_2D, img);
			GLenumVC bindErr = p_glGetError();
			p_glGenFramebuffers(1, &glfbo);
			p_glBindFramebuffer(VC_GL_FRAMEBUFFER, glfbo);
			p_glFramebufferTexture2D(VC_GL_FRAMEBUFFER, VC_GL_COLOR_ATTACHMENT0, VC_GL_TEXTURE_2D, gltex, 0);
			GLenumVC status = p_glCheckFramebufferStatus(VC_GL_FRAMEBUFFER);

			// slice 0 = red, slice 1 = green
			glViewport_(0, 0, W, H);
			glClearColor_((s == 0) ? 1.0f : 0.0f, (s == 1) ? 1.0f : 0.0f, 0.0f, 1.0f);
			glClear_((GLenumVC)VC_GL_COLOR_BUFFER_BIT);
			if (p_glFinish) p_glFinish();

			// Read back THIS slice via a Metal blit into a shared 2D staging tex.
			MTLTextureDescriptor *sd = [[MTLTextureDescriptor alloc] init];
			sd.pixelFormat = cases[ci].fmt; sd.width = W; sd.height = H;
			sd.usage = MTLTextureUsageShaderRead; sd.storageMode = MTLStorageModeShared;
			sd.textureType = MTLTextureType2D;
			id<MTLTexture> staging = [g_mtlDevice newTextureWithDescriptor:sd];
			id<MTLCommandBuffer> cb = [g_cmdQueue commandBuffer];
			id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
			[blit copyFromTexture:tex sourceSlice:s sourceLevel:0
			         sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(W,H,1)
			            toTexture:staging destinationSlice:0 destinationLevel:0
			    destinationOrigin:MTLOriginMake(0,0,0)];
			[blit endEncoding]; [cb commit]; [cb waitUntilCompleted];

			char colorStr[80];
			if (cases[ci].fmt == MTLPixelFormatRGBA8Unorm) {
				uint8_t px[4] = {0,0,0,0};
				[staging getBytes:px bytesPerRow:4 fromRegion:MTLRegionMake2D(W/2,H/2,1,1) mipmapLevel:0];
				snprintf(colorStr, sizeof(colorStr), "RGBA8=%d,%d,%d,%d", px[0],px[1],px[2],px[3]);
			} else {
				__fp16 px[4] = {0,0,0,0};
				[staging getBytes:px bytesPerRow:8 fromRegion:MTLRegionMake2D(W/2,H/2,1,1) mipmapLevel:0];
				snprintf(colorStr, sizeof(colorStr), "RGBA16F=%.2f,%.2f,%.2f,%.2f",
				         (float)px[0],(float)px[1],(float)px[2],(float)px[3]);
			}

			VCLOG(@"[vc-slice] %s slice %d: eglImage=OK fbo=%s bindErr=0x%x -> %s",
			      cases[ci].name, s, vcrt_fbo_status_name(status), (unsigned)bindErr, colorStr);

			p_glBindFramebuffer(VC_GL_FRAMEBUFFER, 0);
			if (glDeleteFramebuffers_) glDeleteFramebuffers_(1, &glfbo);
			if (glDeleteTextures_)     glDeleteTextures_(1, &gltex);
			if (eglDestroyImageKHR_)   eglDestroyImageKHR_(g_display, img);
			staging = nil;
		}
		tex = nil; // ARC releases the test MTLTexture
	}
	VCLOG(@"[vc-slice] done");
}

// ===========================================================================
// ===========================================================================
// GL capability dump, once, with a current context. The question it has to answer:
// does this ANGLE build expose OVR_multiview2? The measured pass-boundary wait
// (eyeset+eyeend = 74-78 % of the render time) has exactly one structural fix --
// render both eyes in ONE pass -- and without the extension that plan does not
// exist, so every estimate built on it is void. Logged in every run, not asserted:
// it is a fact about the build, and it is the first thing to check when the
// multiview question comes up again.
// ===========================================================================
typedef const unsigned char *(*PFN_glGetStringVC)(GLenumVC);
typedef const unsigned char *(*PFN_glGetStringiVC)(GLenumVC, GLuintVC);
typedef void (*PFN_glGetIntegervVC)(GLenumVC, GLintVC *);

// ===========================================================================
// Stufe 4a (multiview-plan.md): draw-and-read-back self-test THROUGH ANGLE.
// Draws colored bars into a 512x512x2 multiview FBO with a real
// GL_OVR_multiview2 program (one draw call, ANGLE doubles the instances and
// routes layers), reads both layers back, and verifies every bar center
// against its expected position -- the spike's method, but through the whole
// GL stack. Runs twice: without foveation, then with a self-built two-layer
// rate map (shared V falloff per the measured rule; H uniform on layer 0,
// falloff on layer 1) registered BY SIZE for the test resolution, where the
// expected positions come from the map itself. Named failure modes as in the
// spike: DOUBLE-WARP and UNSHIFTED. Log filter: vc-mv4.
static const int   kMv4Size  = 512;
static const float kMv4Shift = 24.0f;   // logical px added per view
static const float kMv4Half  = 12.0f;
static const float kMv4Centers[4] = {96.0f, 192.0f, 288.0f, 384.0f};

static id<MTLRasterizationRateMap>
vcrt_mv4_build_map(void)
{
	// Gate-test shape: shared vertical falloff (the rule), horizontal uniform
	// on layer 0 and falloff on layer 1 so the layers are distinguishable.
	const int zones = 8;
	float uni[8], fall[8];
	for (int i = 0; i < zones; i++) {
		float d = fabsf((float)i - (zones - 1) / 2.0f) / ((zones - 1) / 2.0f);
		uni[i]  = 1.0f;
		fall[i] = 1.0f - 0.75f * d;
	}
	MTLRasterizationRateLayerDescriptor *l0 =
		[[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:MTLSizeMake(zones, zones, 1)
		                                                      horizontal:uni
		                                                        vertical:fall];
	MTLRasterizationRateLayerDescriptor *l1 =
		[[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:MTLSizeMake(zones, zones, 1)
		                                                      horizontal:fall
		                                                        vertical:fall];
	MTLRasterizationRateMapDescriptor *desc = [[MTLRasterizationRateMapDescriptor alloc] init];
	desc.screenSize = MTLSizeMake(kMv4Size, kMv4Size, 0);
	[desc setLayer:l0 atIndex:0];
	[desc setLayer:l1 atIndex:1];
	return [g_mtlDevice newRasterizationRateMapWithDescriptor:desc];
}

// Scans one row (horizontal=true) or column of an RGBA8 readback for the given
// color; returns the run center in scan units, or -1. Tolerance matches the
// spike (per-channel 40/255).
static float
vcrt_mv4_scan(const unsigned char *px, int stride, bool horizontal, int fixed, int limit,
              float r, float g, float b)
{
	int first = -1, last = -1;
	for (int i = 0; i < limit; i++) {
		const unsigned char *p = horizontal ? px + (fixed * stride + i) * 4
		                                    : px + (i * stride + fixed) * 4;
		if (abs((int)p[0] - (int)(r * 255)) < 40 && abs((int)p[1] - (int)(g * 255)) < 40 &&
		    abs((int)p[2] - (int)(b * 255)) < 40) {
			if (first < 0) first = i;
			last = i;
		}
	}
	return first >= 0 ? (first + last) / 2.0f : -1.0f;
}

static void
vcrt_mv4_draw_selftest(void)
{
	typedef GLuintVC (*PFN_glCreateShaderVC)(GLenumVC);
	typedef void (*PFN_glShaderSourceVC)(GLuintVC, GLsizeiVC, const char **, const GLintVC *);
	typedef void (*PFN_glCompileShaderVC)(GLuintVC);
	typedef void (*PFN_glGetShaderivVC)(GLuintVC, GLenumVC, GLintVC *);
	typedef GLuintVC (*PFN_glCreateProgramVC)(void);
	typedef void (*PFN_glAttachShaderVC)(GLuintVC, GLuintVC);
	typedef void (*PFN_glLinkProgramVC)(GLuintVC);
	typedef void (*PFN_glGetProgramivVC)(GLuintVC, GLenumVC, GLintVC *);
	typedef void (*PFN_glUseProgramVC)(GLuintVC);
	typedef GLintVC (*PFN_glGetUniformLocationVC)(GLuintVC, const char *);
	typedef void (*PFN_glUniform2fVC)(GLintVC, float, float);
	typedef void (*PFN_glUniform1fVC)(GLintVC, float);
	typedef void (*PFN_glGenBuffersVC)(GLsizeiVC, GLuintVC *);
	typedef void (*PFN_glDeleteBuffersVC)(GLsizeiVC, const GLuintVC *);
	typedef void (*PFN_glBufferDataVC)(GLenumVC, long, const void *, GLenumVC);
	typedef void (*PFN_glEnableVertexAttribArrayVC)(GLuintVC);
	typedef void (*PFN_glDisableVertexAttribArrayVC)(GLuintVC);
	typedef void (*PFN_glVertexAttribPointerVC)(GLuintVC, GLintVC, GLenumVC, unsigned char, GLsizeiVC, const void *);
	typedef void (*PFN_glDrawArraysVC)(GLenumVC, GLintVC, GLsizeiVC);
	typedef void (*PFN_glViewportVC)(GLintVC, GLintVC, GLsizeiVC, GLsizeiVC);
	typedef void (*PFN_glClearColorVC2)(float, float, float, float);
	typedef void (*PFN_glClearVC2)(GLenumVC);
	typedef void (*PFN_glReadPixelsVC)(GLintVC, GLintVC, GLsizeiVC, GLsizeiVC, GLenumVC, GLenumVC, void *);
	typedef void (*PFN_glFramebufferTextureLayerVC)(GLenumVC, GLenumVC, GLuintVC, GLintVC, GLintVC);
	typedef void (*PFN_glDeleteProgramVC)(GLuintVC);
	typedef void (*PFN_glDeleteShaderVC)(GLuintVC);
	typedef void (*PFN_glTexStorage3DVC2)(GLenumVC, GLsizeiVC, GLenumVC, GLsizeiVC, GLsizeiVC, GLsizeiVC);
	typedef void (*PFN_glDeleteTexturesVC2)(GLsizeiVC, const GLuintVC *);
	typedef void (*PFN_glDeleteFramebuffersVC2)(GLsizeiVC, const GLuintVC *);
	typedef void (*PFN_glFBTexMultiviewOVR2)(GLenumVC, GLenumVC, GLuintVC, GLintVC, GLintVC, GLsizeiVC);
	typedef void (*PFN_glGetIntegervVC2)(GLenumVC, GLintVC *);
	typedef void (*PFN_ANGLESetRateMapForSize2)(unsigned, unsigned, unsigned, void *);

#define MV4(fn, ty) ty fn = (ty)g_eglGetProcAddress(#fn); if (!fn) { VCLOG(@"[vc-mv4] SKIPPED (%s unresolved)", #fn); return; }
	MV4(glCreateShader, PFN_glCreateShaderVC) MV4(glShaderSource, PFN_glShaderSourceVC)
	MV4(glCompileShader, PFN_glCompileShaderVC) MV4(glGetShaderiv, PFN_glGetShaderivVC)
	MV4(glCreateProgram, PFN_glCreateProgramVC) MV4(glAttachShader, PFN_glAttachShaderVC)
	MV4(glLinkProgram, PFN_glLinkProgramVC) MV4(glGetProgramiv, PFN_glGetProgramivVC)
	MV4(glUseProgram, PFN_glUseProgramVC) MV4(glGetUniformLocation, PFN_glGetUniformLocationVC)
	MV4(glUniform2f, PFN_glUniform2fVC) MV4(glUniform1f, PFN_glUniform1fVC)
	MV4(glGenBuffers, PFN_glGenBuffersVC) MV4(glDeleteBuffers, PFN_glDeleteBuffersVC)
	MV4(glBufferData, PFN_glBufferDataVC)
	typedef void (*PFN_glBindBufferVC)(GLenumVC, GLuintVC);
	MV4(glBindBuffer, PFN_glBindBufferVC)
	MV4(glEnableVertexAttribArray, PFN_glEnableVertexAttribArrayVC)
	MV4(glDisableVertexAttribArray, PFN_glDisableVertexAttribArrayVC)
	MV4(glVertexAttribPointer, PFN_glVertexAttribPointerVC)
	MV4(glDrawArrays, PFN_glDrawArraysVC) MV4(glViewport, PFN_glViewportVC)
	MV4(glClearColor, PFN_glClearColorVC2) MV4(glClear, PFN_glClearVC2)
	MV4(glReadPixels, PFN_glReadPixelsVC)
	MV4(glFramebufferTextureLayer, PFN_glFramebufferTextureLayerVC)
	MV4(glDeleteProgram, PFN_glDeleteProgramVC) MV4(glDeleteShader, PFN_glDeleteShaderVC)
	MV4(glTexStorage3D, PFN_glTexStorage3DVC2) MV4(glDeleteTextures, PFN_glDeleteTexturesVC2)
	MV4(glDeleteFramebuffers, PFN_glDeleteFramebuffersVC2)
	MV4(glFramebufferTextureMultiviewOVR, PFN_glFBTexMultiviewOVR2)
	MV4(glGetIntegerv, PFN_glGetIntegervVC2)
#undef MV4
	typedef void (*PFN_glFBTexMsMvVC)(GLenumVC, GLenumVC, GLuintVC, GLintVC, GLsizeiVC, GLintVC, GLsizeiVC);
	PFN_glFBTexMsMvVC glFBTexMsMv =
		(PFN_glFBTexMsMvVC)g_eglGetProcAddress("glFramebufferTextureMultisampleMultiviewOVR");

	// --- Program (real OVR_multiview2 shader through ANGLE's translator).
	const char *vsSrc =
		"#version 300 es\n"
		"#extension GL_OVR_multiview2 : require\n"
		"layout(num_views = 2) in;\n"
		"layout(location = 0) in vec2 in_pos;\n"
		"layout(location = 1) in vec4 in_color;\n"
		"uniform vec2 u_screen;\n"
		"uniform float u_shift;\n"
		"out vec4 v_color;\n"
		"void main() {\n"
		"  vec2 p = in_pos + float(gl_ViewID_OVR) * vec2(u_shift, u_shift);\n"
		"  gl_Position = vec4(p.x / u_screen.x * 2.0 - 1.0, 1.0 - p.y / u_screen.y * 2.0, 0.0, 1.0);\n"
		"  v_color = in_color;\n"
		"}\n";
	const char *fsSrc =
		"#version 300 es\n"
		"precision highp float;\n"
		"in vec4 v_color;\n"
		"out vec4 fragColor;\n"
		"void main() { fragColor = v_color; }\n";

	GLuintVC vs = glCreateShader(0x8B31), fs = glCreateShader(0x8B30);
	glShaderSource(vs, 1, &vsSrc, NULL); glCompileShader(vs);
	glShaderSource(fs, 1, &fsSrc, NULL); glCompileShader(fs);
	GLintVC ok = 0;
	glGetShaderiv(vs, 0x8B81 /* COMPILE_STATUS */, &ok);
	if (!ok) { VCLOG(@"[vc-mv4] RESULT=FAIL stage=vs-compile"); return; }
	glGetShaderiv(fs, 0x8B81, &ok);
	if (!ok) { VCLOG(@"[vc-mv4] RESULT=FAIL stage=fs-compile"); return; }
	GLuintVC prog = glCreateProgram();
	glAttachShader(prog, vs); glAttachShader(prog, fs);
	glLinkProgram(prog);
	glGetProgramiv(prog, 0x8B82 /* LINK_STATUS */, &ok);
	if (!ok) { VCLOG(@"[vc-mv4] RESULT=FAIL stage=link"); glDeleteProgram(prog); return; }

	// --- Geometry: 4 vertical bars (full height) + 4 horizontal bars in the
	// left strip, colors as in the spike. 6 verts/quad, [x y r g b a].
	float verts[8 * 6 * 6];
	int vi = 0;
	const float xcol[4][3] = {{1,0,0},{0,1,0},{0,0,1},{1,1,0}};
	const float ycol[4][3] = {{0,1,1},{1,0,1},{1,1,1},{1,0.5f,0}};
	for (int b = 0; b < 4; b++) {
		float c = kMv4Centers[b];
		float q0x = c - kMv4Half, q1x = c + kMv4Half, q0y = 0, q1y = (float)kMv4Size;
		float corn[6][2] = {{q0x,q0y},{q1x,q0y},{q0x,q1y},{q1x,q0y},{q1x,q1y},{q0x,q1y}};
		for (int v = 0; v < 6; v++) {
			verts[vi++] = corn[v][0]; verts[vi++] = corn[v][1];
			verts[vi++] = xcol[b][0]; verts[vi++] = xcol[b][1]; verts[vi++] = xcol[b][2]; verts[vi++] = 1;
		}
	}
	for (int b = 0; b < 4; b++) {
		float c = kMv4Centers[b];
		float q0x = 8, q1x = 72, q0y = c - kMv4Half, q1y = c + kMv4Half;
		float corn[6][2] = {{q0x,q0y},{q1x,q0y},{q0x,q1y},{q1x,q0y},{q1x,q1y},{q0x,q1y}};
		for (int v = 0; v < 6; v++) {
			verts[vi++] = corn[v][0]; verts[vi++] = corn[v][1];
			verts[vi++] = ycol[b][0]; verts[vi++] = ycol[b][1]; verts[vi++] = ycol[b][2]; verts[vi++] = 1;
		}
	}

	// --- Saved state.
	GLintVC prevDrawFbo = 0, prevReadFbo = 0, prevArrayBuf = 0, prevProg = 0, prevVp[4];
	glGetIntegerv(0x8CA6, &prevDrawFbo); glGetIntegerv(0x8CAA, &prevReadFbo);
	glGetIntegerv(0x8894, &prevArrayBuf); glGetIntegerv(0x8B8D, &prevProg);
	glGetIntegerv(0x0BA2 /* GL_VIEWPORT */, prevVp);

	GLuintVC tex = 0, fbo = 0, readFbo = 0, vbo = 0, tex2 = 0, fbo2 = 0;
	p_glGenTextures(1, &tex);
	p_glBindTexture(0x8C1A, tex);
	glTexStorage3D(0x8C1A, 1, 0x8058, kMv4Size, kMv4Size, 2);
	p_glGenFramebuffers(1, &fbo);
	p_glBindFramebuffer(0x8CA9 /* DRAW */, fbo);
	glFramebufferTextureMultiviewOVR(0x8CA9, 0x8CE0, tex, 0, 0, 2);
	// Stufe 4b: same again with the implicit-MSAA variant (2x, like the
	// production eye passes). The texture is the resolve target.
	if (glFBTexMsMv) {
		p_glGenTextures(1, &tex2);
		p_glBindTexture(0x8C1A, tex2);
		glTexStorage3D(0x8C1A, 1, 0x8058, kMv4Size, kMv4Size, 2);
		p_glGenFramebuffers(1, &fbo2);
		p_glBindFramebuffer(0x8CA9, fbo2);
		glFBTexMsMv(0x8CA9, 0x8CE0, tex2, 0, 2, 0, 2);
		VCLOG(@"[vc-mv4] msaa fbo status=0x%X err=0x%X",
		      p_glCheckFramebufferStatus(0x8CA9), p_glGetError());
	} else {
		VCLOG(@"[vc-mv4] glFramebufferTextureMultisampleMultiviewOVR unresolved -> msaa variants skipped");
	}
	p_glGenFramebuffers(1, &readFbo);
	glGenBuffers(1, &vbo);
	glBindBuffer(0x8892 /* ARRAY_BUFFER */, vbo);
	glBufferData(0x8892, sizeof(verts), verts, 0x88E4 /* STATIC_DRAW */);
	glUseProgram(prog);
	glUniform2f(glGetUniformLocation(prog, "u_screen"), (float)kMv4Size, (float)kMv4Size);
	glUniform1f(glGetUniformLocation(prog, "u_shift"), kMv4Shift);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(0, 2, 0x1406 /* FLOAT */, 0, 24, (const void *)0);
	glVertexAttribPointer(1, 4, 0x1406, 0, 24, (const void *)8);
	glViewport(0, 0, kMv4Size, kMv4Size);

	static unsigned char *readback = (unsigned char *)malloc((size_t)kMv4Size * kMv4Size * 4);

	// Two variants: unfoveated, then foveated with a registered two-layer map.
	typedef void (*PFN_SetRateMapForSize)(unsigned, unsigned, unsigned, void *);
	PFN_SetRateMapForSize setRateMapForSize =
		(PFN_SetRateMapForSize)dlsym(RTLD_DEFAULT, "ANGLEMetalSetRasterizationRateMapForSize");
	int totalFail = 0;
	for (int variant = 0; variant < 4; variant++) {
		const bool fove = (variant % 2) == 1;
		const bool msaa = variant >= 2;
		if (msaa && !glFBTexMsMv) break;
		GLuintVC curFbo = msaa ? fbo2 : fbo;
		GLuintVC curTex = msaa ? tex2 : tex;
		const char *vname = msaa ? (fove ? "fove-msaa" : "plain-msaa")
		                         : (fove ? "foveated" : "plain");
		id<MTLRasterizationRateMap> map = nil;
		if (fove) {
			if (!setRateMapForSize || !g_mtlDevice) {
				VCLOG(@"[vc-mv4] foveated variant SKIPPED (registry entry point or device missing)");
				break;
			}
			map = vcrt_mv4_build_map();
			if (!map) { VCLOG(@"[vc-mv4] foveated variant SKIPPED (map build failed)"); break; }
			setRateMapForSize(kMv4Size, kMv4Size, 1, VC_OBJ_TO_VOID(map));
			// The render pass desc is cached per framebuffer state; re-attach
			// so the pass rebuilds and picks the freshly registered map up
			// (found via the host repro: without this the map is ignored).
			p_glBindFramebuffer(0x8CA9, curFbo);
			if (msaa) glFBTexMsMv(0x8CA9, 0x8CE0, curTex, 0, 2, 0, 2);
			else      glFramebufferTextureMultiviewOVR(0x8CA9, 0x8CE0, curTex, 0, 0, 2);
		}
		p_glBindFramebuffer(0x8CA9, curFbo);
		glClearColor(0, 0, 0, 1);
		glClear(0x4000 /* COLOR_BUFFER_BIT */);
		glDrawArrays(0x0004 /* TRIANGLES */, 0, 48);
		GLenumVC drawErr = p_glGetError();

		int fails = 0, checks = 0;
		for (int layer = 0; layer < 2; layer++) {
			float shift = (float)layer * kMv4Shift;
			p_glBindFramebuffer(0x8CA8 /* READ */, readFbo);
			glFramebufferTextureLayer(0x8CA8, 0x8CE0, curTex, 0, layer);
			glReadPixels(0, 0, kMv4Size, kMv4Size, 0x1908 /* RGBA */, 0x1401 /* UNSIGNED_BYTE */, readback);

			// Expected mapping: identity when unfoveated, the map's own answer
			// when foveated. ANGLE stores FBO content bottom-up and the rate
			// map warps in Metal row space, so y expectations run through
			// phys(SIZE - y); readback rows ARE metal rows (host-verified).
			int physW = kMv4Size, physH = kMv4Size;
			if (fove) {
				MTLSize ps = [map physicalSizeForLayer:layer];
				physW = (int)ps.width;
				physH = (int)ps.height;
			}
			for (int axis = 0; axis < 2; axis++) {
				// Scan line position (logical), clear of the other bar set.
				float sx = 40.0f + shift;
				for (int b = 0; b < 4; b++) {
					float logical = kMv4Centers[b] + shift;
					float expected, doubleWarp, unshifted;
					if (axis == 0) {
						if (fove) {
							MTLCoordinate2D e = [map mapScreenToPhysicalCoordinates:MTLCoordinate2DMake(logical, sx) forLayer:layer];
							MTLCoordinate2D d = [map mapScreenToPhysicalCoordinates:e forLayer:layer];
							MTLCoordinate2D u = [map mapScreenToPhysicalCoordinates:MTLCoordinate2DMake(kMv4Centers[b], sx) forLayer:layer];
							expected = e.x; doubleWarp = d.x; unshifted = u.x;
						} else {
							expected = logical; doubleWarp = logical; unshifted = kMv4Centers[b];
						}
					} else {
						if (fove) {
							MTLCoordinate2D e = [map mapScreenToPhysicalCoordinates:MTLCoordinate2DMake(sx, kMv4Size - logical) forLayer:layer];
							MTLCoordinate2D d = [map mapScreenToPhysicalCoordinates:e forLayer:layer];
							MTLCoordinate2D u = [map mapScreenToPhysicalCoordinates:MTLCoordinate2DMake(sx, kMv4Size - kMv4Centers[b]) forLayer:layer];
							expected = e.y; doubleWarp = d.y; unshifted = u.y;
						} else {
							expected = kMv4Size - logical; doubleWarp = expected; unshifted = kMv4Size - kMv4Centers[b];
						}
					}
					// Scan row/column in readback (= Metal row) space.
					int fixed;
					float measured;
					if (axis == 0) {
						float metalRow = fove
							? [map mapScreenToPhysicalCoordinates:MTLCoordinate2DMake(256, kMv4Size - sx) forLayer:layer].y
							: (kMv4Size - sx);
						fixed = (int)lroundf(metalRow);
						const float *col = xcol[b];
						measured = vcrt_mv4_scan(readback, kMv4Size, true, fixed, physW, col[0], col[1], col[2]);
					} else {
						float metalCol = fove
							? [map mapScreenToPhysicalCoordinates:MTLCoordinate2DMake(sx, 256) forLayer:layer].x
							: sx;
						fixed = (int)lroundf(metalCol);
						const float *col = ycol[b];
						measured = vcrt_mv4_scan(readback, kMv4Size, false, fixed, physH, col[0], col[1], col[2]);
					}
					checks++;
					float delta = measured - expected;
					bool pass = measured >= 0 && fabsf(delta) <= 4.0f;
					if (!pass) fails++;
					const char *diag = "";
					if (!pass && measured >= 0) {
						if (fabsf(measured - doubleWarp) <= 4.0f) diag = " <- DOUBLE-WARP";
						else if (fabsf(measured - unshifted) <= 4.0f) diag = " <- UNSHIFTED (layer routing broken?)";
					}
					VCLOG(@"[vc-mv4] %s layer=%d %c-bar[%d] expected=%.1f measured=%.1f delta=%.1f %s%s",
					      vname, layer, axis == 0 ? 'x' : 'y', b,
					      expected, measured, measured >= 0 ? delta : -999.0f,
					      pass ? "OK" : "FAIL", diag);
				}
			}
		}
		VCLOG(@"[vc-mv4] variant=%s drawErr=0x%X RESULT=%s checks=%d fails=%d",
		      vname, drawErr, fails == 0 && drawErr == 0 ? "PASS" : "FAIL",
		      checks, fails);
		totalFail += fails + (drawErr != 0 ? 1 : 0);
		if (fove) {
			setRateMapForSize(kMv4Size, kMv4Size, 1, NULL);   // unbind BEFORE releasing (registry rule)
			map = nil;
		}
	}
	VCLOG(@"[vc-mv4] OVERALL=%s", totalFail == 0 ? "PASS" : "FAIL");

	// --- Restore and tear down.
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glUseProgram((GLuintVC)prevProg);
	glBindBuffer(0x8892, (GLuintVC)prevArrayBuf);
	p_glBindFramebuffer(0x8CA9, (GLuintVC)prevDrawFbo);
	p_glBindFramebuffer(0x8CA8, (GLuintVC)prevReadFbo);
	glViewport(prevVp[0], prevVp[1], prevVp[2], prevVp[3]);
	glDeleteBuffers(1, &vbo);
	glDeleteFramebuffers(1, &readFbo);
	glDeleteFramebuffers(1, &fbo);
	glDeleteTextures(1, &tex);
	if (fbo2) glDeleteFramebuffers(1, &fbo2);
	if (tex2) glDeleteTextures(1, &tex2);
	glDeleteProgram(prog);
	glDeleteShader(vs);
	glDeleteShader(fs);
}

// Word-boundary match: "GL_OVR_multiview" is a PREFIX of "GL_OVR_multiview2", so a
// plain strstr reports the wrong one as present.
static bool
vcrt_has_ext(const char *haystack, const char *name)
{
	size_t len = strlen(name);
	for (const char *p = strstr(haystack, name); p; p = strstr(p + 1, name)) {
		char before = (p == haystack) ? ' ' : p[-1];
		char after  = p[len];
		if ((before == ' ' || before == '\0') && (after == ' ' || after == '\0')) return true;
	}
	return false;
}

static void
vcrt_log_gl_extensions(void)
{
	static bool done = false;
	if (done) return;
	done = true;

	PFN_glGetStringVC   getString   = (PFN_glGetStringVC)g_eglGetProcAddress("glGetString");
	PFN_glGetStringiVC  getStringi  = (PFN_glGetStringiVC)g_eglGetProcAddress("glGetStringi");
	PFN_glGetIntegervVC getIntegerv = (PFN_glGetIntegervVC)g_eglGetProcAddress("glGetIntegerv");
	if (!getString) { VCLOG(@"[vc-caps] glGetString unresolved -- cannot query extensions"); return; }

	const unsigned char *ver = getString(0x1F02 /* GL_VERSION */);
	const unsigned char *ren = getString(0x1F01 /* GL_RENDERER */);
	VCLOG(@"[vc-caps] GL_VERSION = %s | GL_RENDERER = %s",
	      ver ? (const char*)ver : "?", ren ? (const char*)ren : "?");

	// ES 3 reports extensions one at a time; the monolithic GL_EXTENSIONS string is the
	// ES 2 fallback. Collect into one buffer either way so the search is the same.
	static char all[16384];
	size_t used = 0;
	all[0] = '\0';
	GLintVC n = 0;
	if (getStringi && getIntegerv) {
		getIntegerv(0x821D /* GL_NUM_EXTENSIONS */, &n);
		for (GLintVC i = 0; i < n; i++) {
			const unsigned char *e = getStringi(0x1F03 /* GL_EXTENSIONS */, (GLuintVC)i);
			if (!e) continue;
			size_t l = strlen((const char*)e);
			if (used + l + 2 >= sizeof(all)) break;
			memcpy(all + used, e, l); used += l;
			all[used++] = ' '; all[used] = '\0';
		}
	}
	if (used == 0) {
		const unsigned char *e = getString(0x1F03 /* GL_EXTENSIONS */);
		if (e) { strncpy(all, (const char*)e, sizeof(all) - 1); all[sizeof(all) - 1] = '\0'; used = strlen(all); }
	}

	static const char *kWanted[] = {
		"GL_OVR_multiview",
		"GL_OVR_multiview2",
		"GL_OVR_multiview_multisampled_render_to_texture",
		"GL_ANGLE_multiview_multisample",
		"GL_ANGLE_texture_multisample",
	};
	bool haveMV2 = false;
	for (unsigned i = 0; i < sizeof(kWanted)/sizeof(kWanted[0]); i++) {
		bool have = vcrt_has_ext(all, kWanted[i]);
		if (strcmp(kWanted[i], "GL_OVR_multiview2") == 0) haveMV2 = have;
		VCLOG(@"[vc-caps]   %-50s %s", kWanted[i], have ? "YES" : "no");
	}

	// GL_ANGLE_request_extension is present on this build, and that changes what "absent"
	// means: ANGLE keeps some extensions IMPLEMENTED BUT DISABLED, listed separately under
	// GL_REQUESTABLE_EXTENSIONS_ANGLE and switched on with glRequestExtensionANGLE. So
	// missing from GL_EXTENSIONS does NOT yet mean missing from the build -- checking only
	// the active list would have answered a different question than the one asked.
	static char req[8192];
	size_t reqUsed = 0;
	req[0] = '\0';
	GLintVC nReq = 0;
	if (vcrt_has_ext(all, "GL_ANGLE_request_extension") && getStringi && getIntegerv) {
		getIntegerv(0x93A9 /* GL_NUM_REQUESTABLE_EXTENSIONS_ANGLE */, &nReq);
		for (GLintVC i = 0; i < nReq; i++) {
			const unsigned char *e = getStringi(0x93A8 /* GL_REQUESTABLE_EXTENSIONS_ANGLE */, (GLuintVC)i);
			if (!e) continue;
			size_t l = strlen((const char*)e);
			if (reqUsed + l + 2 >= sizeof(req)) break;
			memcpy(req + reqUsed, e, l); reqUsed += l;
			req[reqUsed++] = ' '; req[reqUsed] = '\0';
		}
	}
	bool reqMV2 = vcrt_has_ext(req, "GL_OVR_multiview2");
	bool reqMV  = vcrt_has_ext(req, "GL_OVR_multiview");
	VCLOG(@"[vc-caps] requestable (GL_ANGLE_request_extension): %d entries | GL_OVR_multiview=%s GL_OVR_multiview2=%s",
	      (int)nReq, reqMV ? "YES" : "no", reqMV2 ? "YES" : "no");
	if (reqUsed && (reqMV || reqMV2 || nReq <= 24)) {
		for (size_t off = 0; off < reqUsed; off += 280) {
			char chunk[288];
			size_t l = (reqUsed - off < 280) ? (reqUsed - off) : 280;
			memcpy(chunk, req + off, l); chunk[l] = '\0';
			VCLOG(@"[vc-caps]   req: %s", chunk);
		}
	}

	if (haveMV2 && getIntegerv) {
		GLintVC maxViews = 0;
		getIntegerv(0x9631 /* GL_MAX_VIEWS_OVR */, &maxViews);
		VCLOG(@"[vc-caps]   GL_MAX_VIEWS_OVR = %d (need >= 2)", maxViews);
	}
	VCLOG(@"[vc-caps] VERDICT: one-pass stereo (multiview) is %s",
	      haveMV2 ? "AVAILABLE NOW (active extension)"
	      : (reqMV2 ? "AVAILABLE VIA glRequestExtensionANGLE -- implemented but off by default"
	                : "NOT IN THIS BUILD -- neither active nor requestable; the ANGLE Metal backend would have to implement it"));

	// Stufe 2 Teil 2 (multiview-plan.md): does the backend ACCEPT a multiview
	// attachment? One self-test FBO -- 64x64x2 array texture, two views --
	// checked for completeness and error-free acceptance, then torn down and
	// all bindings restored. Success criteria were fixed in the plan BEFORE
	// this code existed: COMPLETE (0x8CD5) + all glGetError sweeps 0 + the
	// game running on unchanged. Runs only when the extension is active,
	// i.e. behind KL_GL_MULTIVIEW=1.
	if (haveMV2) {
		typedef void (*PFN_glTexStorage3DVC)(GLenumVC, GLsizeiVC, GLenumVC, GLsizeiVC, GLsizeiVC, GLsizeiVC);
		typedef void (*PFN_glDeleteTexturesVC)(GLsizeiVC, const GLuintVC *);
		typedef void (*PFN_glDeleteFramebuffersVC)(GLsizeiVC, const GLuintVC *);
		typedef void (*PFN_glFBTexMultiviewOVR)(GLenumVC, GLenumVC, GLuintVC, GLintVC, GLintVC, GLsizeiVC);
		PFN_glTexStorage3DVC       texStorage3D = (PFN_glTexStorage3DVC)g_eglGetProcAddress("glTexStorage3D");
		PFN_glDeleteTexturesVC     deleteTex    = (PFN_glDeleteTexturesVC)g_eglGetProcAddress("glDeleteTextures");
		PFN_glDeleteFramebuffersVC deleteFbo    = (PFN_glDeleteFramebuffersVC)g_eglGetProcAddress("glDeleteFramebuffers");
		PFN_glFBTexMultiviewOVR    fbTexMV      = (PFN_glFBTexMultiviewOVR)g_eglGetProcAddress("glFramebufferTextureMultiviewOVR");
		if (!texStorage3D || !deleteTex || !deleteFbo || !fbTexMV || !p_glGenTextures ||
		    !p_glBindTexture || !p_glGenFramebuffers || !p_glBindFramebuffer ||
		    !p_glCheckFramebufferStatus || !p_glGetError || !getIntegerv) {
			VCLOG(@"[vc-caps] multiview FBO self-test: SKIPPED (entry point unresolved; glFramebufferTextureMultiviewOVR=%p)",
			      (void *)fbTexMV);
		} else {
			GLintVC prevDrawFbo = 0, prevTex2DArray = 0;
			getIntegerv(0x8CA6 /* GL_DRAW_FRAMEBUFFER_BINDING */, &prevDrawFbo);
			getIntegerv(0x8C1D /* GL_TEXTURE_BINDING_2D_ARRAY */, &prevTex2DArray);
			(void)p_glGetError();   // clear any stale error so the sweeps below are ours
			GLuintVC tex = 0, fbo = 0;
			p_glGenTextures(1, &tex);
			p_glBindTexture(0x8C1A /* GL_TEXTURE_2D_ARRAY */, tex);
			texStorage3D(0x8C1A, 1, 0x8058 /* GL_RGBA8 */, 64, 64, 2);
			p_glGenFramebuffers(1, &fbo);
			p_glBindFramebuffer(0x8CA9 /* GL_DRAW_FRAMEBUFFER */, fbo);
			GLenumVC errSetup = p_glGetError();
			fbTexMV(0x8CA9, 0x8CE0 /* GL_COLOR_ATTACHMENT0 */, tex, 0, /*baseViewIndex*/ 0, /*numViews*/ 2);
			GLenumVC errAttach = p_glGetError();
			GLenumVC status    = p_glCheckFramebufferStatus(0x8CA9);
			GLenumVC errAfter  = p_glGetError();
			bool pass = (status == 0x8CD5 /* GL_FRAMEBUFFER_COMPLETE */) &&
			            errSetup == 0 && errAttach == 0 && errAfter == 0;
			VCLOG(@"[vc-caps] multiview FBO self-test (2 views on 64x64x2 array): status=0x%X err setup/attach/check=0x%X/0x%X/0x%X -> %s",
			      status, errSetup, errAttach, errAfter, pass ? "PASS" : "FAIL");
			p_glBindFramebuffer(0x8CA9, (GLuintVC)prevDrawFbo);
			p_glBindTexture(0x8C1A, (GLuintVC)prevTex2DArray);
			deleteFbo(1, &fbo);
			deleteTex(1, &tex);
			// Stufe 4a: only once the plain attach is proven COMPLETE does the
			// draw-and-read-back test add meaning.
			if (pass) {
				vcrt_mv4_draw_selftest();
			}
		}
	}

	// Independent of multiview: EXT_multisampled_render_to_texture would let the eye FBO
	// carry MSAA in tile memory and resolve on store, instead of our separate 228 MB
	// multisample FBO plus a full-screen blit per eye. Different question, same log.
	VCLOG(@"[vc-caps] GL_EXT_multisampled_render_to_texture = %s (implicit MSAA resolve -> no separate MSAA FBO, no resolve blit)",
	      vcrt_has_ext(all, "GL_EXT_multisampled_render_to_texture") ? "YES" : "no");

	// When it is missing, the full list IS the useful part: it says what the Metal
	// backend does expose, which is the starting point for judging the port effort.
	if (!haveMV2) {
		VCLOG(@"[vc-caps] %d extensions, %zu bytes; full list follows", (int)n, used);
		for (size_t off = 0; off < used; off += 280) {
			char chunk[288];
			size_t l = (used - off < 280) ? (used - off) : 280;
			memcpy(chunk, all + off, l); chunk[l] = '\0';
			VCLOG(@"[vc-caps]   %s", chunk);
		}
	}
}

// Phase 5.5 stereo target: DOUBLE-BUFFERED to share the cinema state machine.
// Per back buffer i: ONE MTLTexture (2D array, arrayLength=2) whose slices are
// the two eyes, plus a GL FBO per slice. Depth is a DEDICATED renderbuffer, never
// the cinema depth, cleared per eye pass. It used to be ONE for all FBOs -- see
// VC_STEREO_DEPTH_SPLIT below for why that is now per eye. The eye textures are tied to the SAME index as the
// cinema g_buf[i], so vcrt_begin_frame / vcrt_publish_frame / the shared-event
// sync all apply unchanged -- only which texture vc_acquire_ready_frame hands out
// (and eye_count) differs by mode.
// ===========================================================================
typedef struct {
	id<MTLTexture> arrayTex;    // 2D array, arrayLength 2 (slice 0 = left, 1 = right)
	void          *img[2];      // per-slice EGLImage
	GLuintVC       glTex[2];
	GLuintVC       fbo[2];
} VCStereoBuffer;

static VCStereoBuffer g_stereoBuf[VC_NUM_BUFFERS];
// VC_STEREO_DEPTH_SPLIT (default 1): one depth renderbuffer PER EYE instead of one
// shared by every eye FBO of every ring buffer.
// WHY, measured: with VC_MSAA=0 the per-eye split of the boundary timers showed
// eyClr = 0.0 / 3.2-4.9 ms -- eye 0's clear is free, eye 1's costs 3-5 ms. Eye 0's clear
// writes an EGLImage-imported slice too and costs nothing, so it is NOT a per-touch
// interop tax; the CPU waits for the PREVIOUS EYE. The shared depth renderbuffer is the
// coupling: eye 1's glClear(COLOR|DEPTH) writes the very buffer eye 0 has just been
// writing -- a write-after-write hazard on one resource. Eye 0 hits the same hazard
// against the previous FRAME's eye 1, but a whole logic phase and a publish sit in
// between, so that work is long finished -- which is exactly the 0.0 / 3-5 asymmetry.
// Not caught by the MSAA=2 test because there each MSAA FBO already had its own depth;
// the wait came from the resolve blit's read-after-write on eye 0's colour instead.
// Cost: one extra D24S8 renderbuffer, ~28 MB at 7.1 MP. Index 1 unused when off.
static GLuintVC       g_stereoDepthRbo[2] = {0, 0};
static int            g_stereoDepthSplit  = -1;   // -1 = unread
static bool           g_stereoReady    = false;
static bool           g_stereoFailed   = false;

static int
vcStereoDepthSplit(void)
{
	if (g_stereoDepthSplit < 0) {
		const char *e = getenv("VC_STEREO_DEPTH_SPLIT");
		g_stereoDepthSplit = e ? (atoi(e) ? 1 : 0) : 1;   // default: one depth per eye
	}
	return g_stereoDepthSplit;
}

extern "C" bool  vc_stereo_ready(void) { return g_stereoReady; }
extern "C" void *vc_stereo_array_texture(int idx)
{
	if (!g_stereoReady || idx < 0 || idx >= g_numBuffers) return NULL;
	return VC_OBJ_TO_VOID(g_stereoBuf[idx].arrayTex);
}

// ---- Fixed foveation (default ON; VC_FOVEATE=0 disables) ------------------
// Render the eye passes at a variable rate: sharp centre (rate 1.0), coarse edges
// (VC_FOVEATE_EDGE, default 0.2) over VC_FOVEATE_ZONES (default 16) zones/axis. A
// centred, PARAMETRIC curve (not the compositor's gaze-driven map): the slice pipeline
// is decoupled/buffered, so a gaze-following map would sit stale; and parametric lets
// us tune the reduction to the frame budget. We build our own MTLRasterizationRateMap at
// the SLICE size and bind it BY IDENTITY to each stereo array texture via the patched
// ANGLE registry -- so ANGLE foveates ONLY the world eye passes, never the HUD/cinema.
//
// MSAA is forced OFF under foveation (see vcMsaaSamples): the eye passes then render
// DIRECTLY into the array slice we own (by-identity works), not an ANGLE-owned
// multisample renderbuffer whose glBlitFramebuffer resolve is not rate-map aware.
//
// Phase 4a: the slice is left in the rate map's WARPED physical layout and the display
// quad still samples it with plain UV -> the IMAGE looks distorted until the unwarp grid
// (Phase 6). The eye-pass GPU time (VC_EYE_GPU) is already valid -> measure the raw win.
typedef void (*PFN_ANGLESetRateMap)(void *, void *);
typedef void (*PFN_ANGLESetRateMapForSize)(unsigned int, unsigned int, unsigned int, void *);
static PFN_ANGLESetRateMap         p_ANGLESetRateMap = NULL;
static PFN_ANGLESetRateMapForSize  p_ANGLESetRateMapForSize = NULL;
static int vcMsaaSamples(void);   // forward: foveation registration depends on the MSAA path
// IMPLICIT MSAA (GL_EXT_multisampled_render_to_texture, confirmed present on this build).
// Instead of our own multisample FBO plus a full-screen glBlitFramebuffer resolve per eye,
// the slice FBO itself carries the sample count and Metal resolves as the pass's STORE
// ACTION. Verified in the ANGLE source (Prototypes/angle-src) rather than assumed:
//   RenderBufferMtl.mm / TextureMtl.mm allocate the multisample surface with
//   MakeMemoryLess2DMSTexture -> MTLStorageModeMemoryless, i.e. tile memory only, and
//   RenderTargetMtl::setWithImplicitMSTexture makes the base texture the resolve target.
// So it is a REAL implicit resolve, not an emulated blit -- the 228 MB and the blit both
// go away. VC_MSAA_IMPLICIT=0 falls back to the explicit path for the A/B.
static int vcMsaaImplicit(void);
static bool g_msaaImplicitActive = false;   // set only once the FBO is verified COMPLETE
static bool g_msaaImplicitFailed = false;   // set if any slice rejected it -> explicit path
static id<MTLRasterizationRateMap> g_foveMap = nil;

// Sampled optical curve pushed from Swift (vc_set_foveation_curve): the compositor's own
// per-eye rate profile resampled to our slice. Preferred over the parametric bell -- it
// puts the dense zone on the true (off-centre) optical axis and removes only fragments the
// compositor discards anyway. Guarded by g_bufMutex; built into g_foveMap on the game
// thread. NX/NY 0 => none yet (parametric fallback).
static float g_foveCurveH[64], g_foveCurveV[64];
static int   g_foveCurveNX = 0, g_foveCurveNY = 0;
static bool  g_foveCurveSet = false, g_foveCurveDirty = false;

static bool
vcFoveateOn(void)
{
	static int v = -1;
	if (v < 0) { const char *e = getenv("VC_FOVEATE"); v = (e && e[0] == '0') ? 0 : 1; }  // default ON; VC_FOVEATE=0 disables
	return v != 0;
}

static id<MTLRasterizationRateMap>
vcrt_build_fove_map(int W, int H)
{
	if (![g_mtlDevice supportsRasterizationRateMapWithLayerCount:1]) {
		VCLOG(@"[vc-fove] device has no rasterization rate map support -> disabled");
		return nil;
	}
	float hq[64], vq[64];
	int nx, ny;
	// Prefer the sampled optical curve from Swift; else the parametric bell.
	pthread_mutex_lock(&g_bufMutex);
	bool haveCurve = g_foveCurveSet;
	if (haveCurve) {
		nx = g_foveCurveNX; ny = g_foveCurveNY;
		memcpy(hq, g_foveCurveH, (size_t)nx * sizeof(float));
		memcpy(vq, g_foveCurveV, (size_t)ny * sizeof(float));
	}
	pthread_mutex_unlock(&g_bufMutex);
	if (!haveCurve) {
		int zones = 16; { const char *e = getenv("VC_FOVEATE_ZONES"); if (e) { int z = atoi(e); if (z >= 2 && z <= 64) zones = z; } }
		float edge = 0.2f; { const char *e = getenv("VC_FOVEATE_EDGE"); if (e) { float f = (float)atof(e); if (f > 0.05f && f <= 1.0f) edge = f; } }
		for (int i = 0; i < zones; i++) {
			float f = ((float)i + 0.5f) / (float)zones;
			float w = 0.5f * (1.0f + cosf((float)M_PI * (2.0f * f - 1.0f)));
			float r = edge + (1.0f - edge) * w;
			hq[i] = vq[i] = (r < 0.01f) ? 0.01f : r;
		}
		nx = ny = zones;
	}
	VCLOG(@"[vc-fove] build: source=%s nx=%d ny=%d", haveCurve ? "compositor-curve(Swift)" : "parametric-bell", nx, ny);
	// Any bad ObjC selector / Metal exception here must DISABLE foveation, not abort the
	// whole process. Step logs so the last one printed pinpoints the failing call.
	id<MTLRasterizationRateMap> map = nil;
	@try {
		VCLOG(@"[vc-fove] build step 1: layer descriptor (nx=%d ny=%d)", nx, ny);
		MTLRasterizationRateLayerDescriptor *layer =
			[[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:MTLSizeMake(nx, ny, 1)
			                                                     horizontal:hq
			                                                       vertical:vq];
		VCLOG(@"[vc-fove] build step 2: map descriptor (screen=%dx%d)", W, H);
		MTLRasterizationRateMapDescriptor *desc = [[MTLRasterizationRateMapDescriptor alloc] init];
		desc.screenSize = MTLSizeMake(W, H, 0);
		[desc setLayer:layer atIndex:0];
		VCLOG(@"[vc-fove] build step 3: newRasterizationRateMapWithDescriptor");
		map = [g_mtlDevice newRasterizationRateMapWithDescriptor:desc];
	} @catch (NSException *ex) {
		VCLOG(@"[vc-fove] EXCEPTION building rate map: %@ (%@) -> foveation DISABLED",
		      ex.name, ex.reason);
		return nil;
	}
	if (map) {
		MTLSize ph = [map physicalSizeForLayer:0];
		VCLOG(@"[vc-fove] rate map built: screen=%dx%d physical=%dx%d (~%.0f%% of fragments)",
		      W, H, (int)ph.width, (int)ph.height,
		      100.0 * (double)ph.width * (double)ph.height / ((double)W * (double)H));
	} else {
		VCLOG(@"[vc-fove] newRasterizationRateMapWithDescriptor returned nil -> foveation off");
	}
	return map;
}

// Build once, then bind by identity to every stereo array texture. Called at the end of
// vcrt_stereo_ensure (textures exist). No-op unless VC_FOVEATE=1.
static void
vcrt_foveation_apply(int W, int H)
{
	if (!vcFoveateOn()) return;
	VCLOG(@"[vc-fove] apply: entering (VC_FOVEATE=1, %dx%d)", W, H);
	g_foveCurveDirty = false;                      // consume any pending curve
	g_foveMap = vcrt_build_fove_map(W, H);         // always (re)build; ARC releases the old map
	if (g_foveMap == nil) return;
	if (!p_ANGLESetRateMap) {
		p_ANGLESetRateMap = (PFN_ANGLESetRateMap)dlsym(RTLD_DEFAULT, "ANGLEMetalSetRasterizationRateMap");
		if (!p_ANGLESetRateMap) { VCLOG(@"[vc-fove] ANGLEMetalSetRasterizationRateMap symbol missing -> unpatched ANGLE?"); return; }
	}
	if (!p_ANGLESetRateMapForSize)
		p_ANGLESetRateMapForSize = (PFN_ANGLESetRateMapForSize)dlsym(RTLD_DEFAULT, "ANGLEMetalSetRasterizationRateMapForSize");

	int samples = vcMsaaSamples();
	if (samples > 0 && g_msaaImplicitActive) {
		// IMPLICIT MSAA: register BY IDENTITY, like the no-MSAA path. The by-size hack
		// below exists only because the EXPLICIT path's multisample renderbuffer is
		// ANGLE-owned and has no MTLTexture anyone outside can name. With the implicit
		// path the render target's BASE texture is our own slice again, and klepton's
		// lookup (RateMapForRenderTarget -> GetRasterizationRateMapForTarget) tries
		// identity FIRST, walking up parentTexture -- so registering the array texture
		// matches, and the resolve store action preserves the physical layout the same
		// way the blit did. Checked against the patch source, not assumed.
		int n = 0;
		for (int i = 0; i < g_numBuffers; i++) {
			id<MTLTexture> t = g_stereoBuf[i].arrayTex;
			if (t) { p_ANGLESetRateMap(VC_OBJ_TO_VOID(t), VC_OBJ_TO_VOID(g_foveMap)); n++; }
		}
		VCLOG(@"[vc-fove] rate map bound BY IDENTITY to %d stereo slices (implicit MSAA %dx: the pass resolves into the slice, so the slice IS the render target)", n, samples);
	} else if (samples > 0 && p_ANGLESetRateMapForSize) {
		// MSAA path: the eye passes render into the multisample target(s) (W x H, N
		// samples), which ANGLE owns -> register the map BY SIZE so it attaches to that
		// pass. By SIZE, so VC_MSAA_SPLIT's second per-eye target is covered by the same
		// registration. The position-preserving resolve blit then carries the warped layout into
		// the slice. Do NOT register the slice by identity here: the resolve writes into
		// the slice and must stay a clean 1:1 copy (the by-size registry only matches the
		// N-sample target, so the 1-sample slice/HUD are untouched either way).
		p_ANGLESetRateMapForSize((unsigned)W, (unsigned)H, (unsigned)samples, VC_OBJ_TO_VOID(g_foveMap));
		VCLOG(@"[vc-fove] rate map bound BY SIZE to the %dx%d %dx-MSAA target (resolve carries it to the slices)", W, H, samples);
	} else {
		// No MSAA: the eye passes render straight into the slices we own -> by identity.
		int n = 0;
		for (int i = 0; i < g_numBuffers; i++) {
			id<MTLTexture> t = g_stereoBuf[i].arrayTex;
			if (t) { p_ANGLESetRateMap(VC_OBJ_TO_VOID(t), VC_OBJ_TO_VOID(g_foveMap)); n++; }
		}
		VCLOG(@"[vc-fove] rate map bound BY IDENTITY to %d stereo slices (world eye passes only)", n);
	}
}

// Game-thread poll: rebuild+register when Swift has pushed a new optical curve.
static void
vcrt_foveation_poll_dirty(void)
{
	if (g_stereoReady && vcFoveateOn() && g_foveCurveDirty)
		vcrt_foveation_apply(g_rtWidth, g_rtHeight);
}

// The rate map the slices were rendered with (for the display quad's unwarp), or NULL.
extern "C" void *
vc_foveation_rate_map(void)
{
	return (vcFoveateOn() && g_foveMap != nil) ? VC_OBJ_TO_VOID(g_foveMap) : NULL;
}

// Is foveation requested? (Swift gate for sampling + pushing the optical curve.)
extern "C" int
vc_foveation_wanted(void)
{
	return vcFoveateOn() ? 1 : 0;
}

// Swift pushes the sampled compositor rate curve (per-axis rates, enveloped over both
// eyes, peak-normalized, floored). Stored; the game thread rebuilds g_foveMap from it and
// re-registers on the slices (see the dirty check in vcrt_begin_frame + vcrt_stereo_ensure).
extern "C" void
vc_set_foveation_curve(const float *h, int nx, const float *v, int ny)
{
	if (!h || !v || nx < 2 || ny < 2 || nx > 64 || ny > 64) return;
	pthread_mutex_lock(&g_bufMutex);
	memcpy(g_foveCurveH, h, (size_t)nx * sizeof(float));
	memcpy(g_foveCurveV, v, (size_t)ny * sizeof(float));
	g_foveCurveNX = nx; g_foveCurveNY = ny;
	g_foveCurveSet = true; g_foveCurveDirty = true;
	pthread_mutex_unlock(&g_bufMutex);
	VCLOG(@"[vc-fove] curve received from Swift: %d x %d zones", nx, ny);
}

// Lazily create BOTH stereo back buffers. Called from the game thread (GL context
// current) on the first eye pass, so g_rtWidth/Height and the GL entry points are
// resolved and g_buf already exists. Idempotent; latches failure, logs once.
extern "C" bool
vcrt_stereo_ensure(void)
{
	if (g_stereoReady)  return true;
	if (g_stereoFailed) return false;
	if (g_mtlDevice == nil || !p_eglCreateImageKHR || !g_eglGetProcAddress) {
		g_stereoFailed = true; VCLOG(@"[vc-stereo] SKIP: prerequisites missing"); return false;
	}

	// Renderbuffer entry points for the dedicated depth buffer (resolved locally;
	// the main table doesn't carry them).
	typedef void (*PFN_glGenRenderbuffers)(GLsizeiVC, GLuintVC *);
	typedef void (*PFN_glBindRenderbuffer)(GLenumVC, GLuintVC);
	typedef void (*PFN_glRenderbufferStorage)(GLenumVC, GLenumVC, GLsizeiVC, GLsizeiVC);
	PFN_glGenRenderbuffers    glGenRenderbuffers_    = (PFN_glGenRenderbuffers)g_eglGetProcAddress("glGenRenderbuffers");
	PFN_glBindRenderbuffer    glBindRenderbuffer_    = (PFN_glBindRenderbuffer)g_eglGetProcAddress("glBindRenderbuffer");
	PFN_glRenderbufferStorage glRenderbufferStorage_ = (PFN_glRenderbufferStorage)g_eglGetProcAddress("glRenderbufferStorage");
	if (!glGenRenderbuffers_ || !glBindRenderbuffer_ || !glRenderbufferStorage_) {
		g_stereoFailed = true; VCLOG(@"[vc-stereo] SKIP: renderbuffer funcs unresolved"); return false;
	}

	const int W = g_rtWidth, H = g_rtHeight;

	// Implicit-MSAA entry points (EXT_multisampled_render_to_texture). Resolved locally;
	// absence is not fatal, it just means the explicit path stays.
	typedef void (*PFN_glFramebufferTexture2DMultisampleEXT)(GLenumVC, GLenumVC, GLenumVC, GLuintVC, GLintVC, GLsizeiVC);
	typedef void (*PFN_glRenderbufferStorageMultisampleEXT)(GLenumVC, GLsizeiVC, GLenumVC, GLsizeiVC, GLsizeiVC);
	PFN_glFramebufferTexture2DMultisampleEXT fbTex2DMS =
		(PFN_glFramebufferTexture2DMultisampleEXT)g_eglGetProcAddress("glFramebufferTexture2DMultisampleEXT");
	PFN_glRenderbufferStorageMultisampleEXT rbStorageMSExt =
		(PFN_glRenderbufferStorageMultisampleEXT)g_eglGetProcAddress("glRenderbufferStorageMultisampleEXT");
	const int implicitSamples = (vcMsaaImplicit() && vcMsaaSamples() > 0 && fbTex2DMS && rbStorageMSExt)
	                            ? vcMsaaSamples() : 0;
	if (vcMsaaImplicit() && vcMsaaSamples() > 0 && !(fbTex2DMS && rbStorageMSExt))
		VCLOG(@"[vc-msaa] implicit MSAA requested but entry points unresolved (fbTex2DMS=%p rbStorageMSExt=%p) -- falling back to the EXPLICIT resolve path",
		      (void*)fbTex2DMS, (void*)rbStorageMSExt);

	// Depth renderbuffer(s), cleared per pass. One PER EYE by default so eye 1 does not
	// write the buffer eye 0 is still writing (see VC_STEREO_DEPTH_SPLIT above).
	// With implicit MSAA the depth must carry the SAME sample count as the colour or the
	// FBO is incomplete -- and glRenderbufferStorageMultisampleEXT is what makes ANGLE
	// allocate it memoryless too (RenderBufferMtl.mm), so the depth buffer also stops
	// costing real memory.
	const int nDepth = vcStereoDepthSplit() ? 2 : 1;
	for (int d = 0; d < nDepth; d++) {
		glGenRenderbuffers_(1, &g_stereoDepthRbo[d]);
		glBindRenderbuffer_(VC_GL_RENDERBUFFER, g_stereoDepthRbo[d]);
		if (implicitSamples > 0)
			rbStorageMSExt(VC_GL_RENDERBUFFER, implicitSamples, 0x88F0 /* GL_DEPTH24_STENCIL8 */, W, H);
		else
			glRenderbufferStorage_(VC_GL_RENDERBUFFER, 0x88F0 /* GL_DEPTH24_STENCIL8 */, W, H);
	}
	if (nDepth == 1) g_stereoDepthRbo[1] = g_stereoDepthRbo[0];

	for (int i = 0; i < g_numBuffers; i++) {
		VCStereoBuffer *b = &g_stereoBuf[i];

		// Verified params (same as the cinema buffers / slice test) + array type.
		MTLTextureDescriptor *td = [[MTLTextureDescriptor alloc] init];
		td.pixelFormat = MTLPixelFormatRGBA8Unorm;
		td.width = W; td.height = H;
		td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
		td.storageMode = MTLStorageModePrivate;
		td.textureType = MTLTextureType2DArray;
		td.arrayLength = 2;
		b->arrayTex = [g_mtlDevice newTextureWithDescriptor:td];
		if (b->arrayTex == nil) { g_stereoFailed = true; VCLOG(@"[vc-stereo] buffer %d FAIL: array MTLTexture create", i); return false; }

		for (int s = 0; s < 2; s++) {
			const EGLint attrs[] = { (EGLint)VC_EGL_METAL_TEXTURE_ARRAY_SLICE_ANGLE, (EGLint)s, (EGLint)VC_EGL_NONE };
			b->img[s] = p_eglCreateImageKHR(g_display, (EGLContext)0, VC_EGL_METAL_TEXTURE_ANGLE,
			                                VC_OBJ_TO_VOID(b->arrayTex), attrs);
			if (b->img[s] == NULL) {
				g_stereoFailed = true;
				VCLOG(@"[vc-stereo] buffer %d slice %d FAIL eglCreateImageKHR (egl 0x%x)", i, s, g_eglGetError ? g_eglGetError() : 0);
				return false;
			}
			p_glGenTextures(1, &b->glTex[s]);
			p_glBindTexture(VC_GL_TEXTURE_2D, b->glTex[s]);
			p_glEGLImageTargetTexture2DOES(VC_GL_TEXTURE_2D, b->img[s]);
			GLenumVC bindErr = p_glGetError();
			p_glGenFramebuffers(1, &b->fbo[s]);
			p_glBindFramebuffer(VC_GL_FRAMEBUFFER, b->fbo[s]);
			if (implicitSamples > 0)
				fbTex2DMS(VC_GL_FRAMEBUFFER, VC_GL_COLOR_ATTACHMENT0, VC_GL_TEXTURE_2D, b->glTex[s], 0, implicitSamples);
			else
				p_glFramebufferTexture2D(VC_GL_FRAMEBUFFER, VC_GL_COLOR_ATTACHMENT0, VC_GL_TEXTURE_2D, b->glTex[s], 0);
			GLenumVC msErr = implicitSamples > 0 ? p_glGetError() : 0;
			p_glFramebufferRenderbuffer(VC_GL_FRAMEBUFFER, VC_GL_DEPTH_STENCIL_ATTACHMENT, VC_GL_RENDERBUFFER, g_stereoDepthRbo[s]);
			GLenumVC status = p_glCheckFramebufferStatus(VC_GL_FRAMEBUFFER);
			// LOUD on failure and FALL BACK: the slices are EGLImage-imported Metal
			// textures, and ANGLE-on-Metal has silently failed on EGLImage combinations
			// before. An incomplete FBO here must not ship as a black screen.
			if (implicitSamples > 0 && (status != VC_GL_FRAMEBUFFER_COMPLETE || msErr != 0)) {
				VCLOG(@"[vc-msaa] IMPLICIT MSAA %dx REJECTED on buffer %d slice %d: status=%s glErr=0x%x (EGLImage slice + implicit resolve not supported) -- reverting to the EXPLICIT resolve path",
				      implicitSamples, i, s, vcrt_fbo_status_name(status), (unsigned)msErr);
				p_glFramebufferTexture2D(VC_GL_FRAMEBUFFER, VC_GL_COLOR_ATTACHMENT0, VC_GL_TEXTURE_2D, b->glTex[s], 0);
				glBindRenderbuffer_(VC_GL_RENDERBUFFER, g_stereoDepthRbo[s]);
				glRenderbufferStorage_(VC_GL_RENDERBUFFER, 0x88F0 /* GL_DEPTH24_STENCIL8 */, W, H);
				p_glFramebufferRenderbuffer(VC_GL_FRAMEBUFFER, VC_GL_DEPTH_STENCIL_ATTACHMENT, VC_GL_RENDERBUFFER, g_stereoDepthRbo[s]);
				status = p_glCheckFramebufferStatus(VC_GL_FRAMEBUFFER);
				g_msaaImplicitFailed = true;
			}
			VCLOG(@"[vc-stereo] buffer %d slice %d: %dx%d RGBA8 gltex %u fbo %u depthRbo %u samples=%d(%s) bindErr=0x%x status=%s",
			      i, s, W, H, b->glTex[s], b->fbo[s], g_stereoDepthRbo[s],
			      g_msaaImplicitFailed ? 1 : implicitSamples,
			      implicitSamples > 0 && !g_msaaImplicitFailed ? "implicit resolve" : "explicit path",
			      (unsigned)bindErr, vcrt_fbo_status_name(status));
		}
	}
	p_glBindFramebuffer(VC_GL_FRAMEBUFFER, 0);
	g_stereoReady = true;
	// Must be settled BEFORE vcrt_foveation_apply below: it picks identity vs by-size
	// registration off this flag.
	g_msaaImplicitActive = (implicitSamples > 0 && !g_msaaImplicitFailed);
	if (vcMsaaSamples() > 0)
		VCLOG(@"[vc-msaa] MSAA %dx path = %s", vcMsaaSamples(), g_msaaImplicitActive
		      ? "IMPLICIT (EXT_multisampled_render_to_texture: memoryless tile-memory samples, resolve as store action -- no separate MSAA FBO, no resolve blit)"
		      : "EXPLICIT (own multisample FBO + glBlitFramebuffer resolve per eye)");
	vcrt_log_gl_extensions();   // multiview support -- decides whether the one-pass plan exists at all
	VCLOG(@"[vc-stereo] array render target ready (%d buffers x 2 slices, %dx%d, %d dedicated depth rbo%s -- VC_STEREO_DEPTH_SPLIT=%d: %s)",
	      g_numBuffers, W, H, nDepth, nDepth == 1 ? "" : "s", vcStereoDepthSplit(),
	      vcStereoDepthSplit() ? "one per eye -- no write-after-write between the eye passes"
	                           : "SHARED by both eyes (old behaviour)");
	vcrt_foveation_apply(W, H);   // bind the rate map to the slices when VC_FOVEATE=1
	// Render-target VRAM per resolution step, and the bytes WRITTEN per frame (both eye
	// slices; that is the fragment/bandwidth cost the eye passes pay). Answers area-vs-
	// bandwidth: eyes-ms should track slice bytes-written if bandwidth-bound, pixels if
	// fragment-bound. RGBA8=4B/px colour, D24S8=4B/px depth (shared, written once/frame).
	{
		double mp        = (double)W * (double)H / 1.0e6;
		uint64_t sliceBuf = (uint64_t)W * H * 4ull * 2ull;              // 2 slices RGBA8, per buffer
		uint64_t sliceAll = sliceBuf * (uint64_t)g_numBuffers;         // x N buffers (resident)
		uint64_t cinema   = (uint64_t)W * H * 4ull * (uint64_t)g_numBuffers; // HUD/cinema 2D tex x N
		uint64_t depth    = (uint64_t)W * H * 4ull * (uint64_t)nDepth; // D24S8 renderbuffer(s)
		uint64_t resident = sliceAll + cinema + depth;
		uint64_t writtenPerFrame = (uint64_t)W * H * 4ull * 2ull       // 2 eye slice colours
		                         + (uint64_t)W * H * 4ull * 2ull;      // + depth, once per eye pass
		VCLOG(@"[vc-vram] %dx%d (%.2f MP/eye)  resident: slices=%llu MB cinema=%llu MB depth=%llu MB total=%llu MB  |  written/frame(2 slices+depth)=%llu MB",
		      W, H, mp,
		      (unsigned long long)(sliceAll / 1000000ull), (unsigned long long)(cinema / 1000000ull),
		      (unsigned long long)(depth / 1000000ull),   (unsigned long long)(resident / 1000000ull),
		      (unsigned long long)(writtenPerFrame / 1000000ull));
	}
	return true;
}

// ===========================================================================
// MSAA (VC_MSAA=2/4/8, default 0=off). The eye passes render into ONE shared
// multisample FBO (colour + depth+stencil multisample renderbuffers at the slice
// size) and are RESOLVED via glBlitFramebuffer into the per-slice single-sample
// EGLImage FBO the compositor samples. Passes are sequential on this GL thread, so
// we resolve the PREVIOUS eye at the start of the next eye's binding, and the LAST
// eye from vc_stereo_restore_main. Any failure is LOUD (status name + glGetError,
// logged once) -- ANGLE-on-Metal has silently failed on EGLImage combinations
// before, so we never assume the blit worked; on failure we fall back to 1x but
// SAY SO in the log.
// ===========================================================================
typedef void (*PFN_glGenRenderbuffers)(GLsizeiVC, GLuintVC *);
typedef void (*PFN_glBindRenderbuffer)(GLenumVC, GLuintVC);
typedef void (*PFN_glRenderbufferStorageMultisample)(GLenumVC, GLsizeiVC, GLenumVC, GLsizeiVC, GLsizeiVC);
typedef void (*PFN_glBlitFramebuffer)(GLintVC, GLintVC, GLintVC, GLintVC, GLintVC, GLintVC, GLintVC, GLintVC, GLenumVC, GLenumVC);

static int      g_msaaSamples  = -1;   // -1 = unread; 0 = off; 2/4/8 = on
// VC_MSAA_SPLIT (default 0 = OFF, shared target): one multisample target PER EYE.
// MEASURED AND REFUTED on device. The idea was that the shared target is a write-after-
// write hazard -- eye 1 re-renders the exact renderbuffer eye 0 just wrote. Two targets
// changed NOTHING (eyPre 1.9/2.0/3.9/4.2 ms split vs 1.9/2.0/3.4/4.3 shared, at matching
// draw counts; refutation threshold was 20 %).
// WHY IT COULD NOT WORK: the per-eye split of the boundary timers, added in the same
// build, showed the wait sits ONLY at eye 1's boundary (e0/e1 = 0.0/1.9 etc.) -- i.e. in
// the resolve blit of eye 0, which READS eye 0's multisample buffer. That is a
// READ-after-write dependency on eye 0's rendering, not a write-after-write one on the
// target, so giving eye 1 its own target cannot help. Kept behind the switch (off, no
// memory cost) because it is the cheap A/B if the resolve scheme is ever restructured.
// Cost when on: one extra colour + one extra depth multisample renderbuffer (~114 MB at
// 7.1 MP / 2x), logged below. Index 1 is only used when split is on; else both use [0].
static GLuintVC g_msaaFbo[2]      = {0, 0};
static GLuintVC g_msaaColorRbo[2] = {0, 0};
static GLuintVC g_msaaDepthRbo[2] = {0, 0};
static int      g_msaaSplit    = -1;   // -1 = unread
static bool     g_msaaReady    = false;
static bool     g_msaaFailed   = false;
static int      g_msaaPending  = -1;    // eye whose content sits in its MSAA FBO awaiting resolve
static PFN_glGenRenderbuffers               p_msGenRenderbuffers  = NULL;
static PFN_glBindRenderbuffer               p_msBindRenderbuffer  = NULL;
static PFN_glRenderbufferStorageMultisample p_msRenderbufferStorageMultisample = NULL;
static PFN_glBlitFramebuffer                p_msBlitFramebuffer   = NULL;

extern "C" void vc_stereo_msaa_resolve_pending(void);

static int
vcMsaaSamples(void)
{
	// MSAA and foveation now COEXIST: with foveation on, the eye passes render into the
	// shared multisample target (registered by SIZE so ANGLE attaches the rate map there);
	// the position-preserving glBlitFramebuffer resolve carries the warped layout into the
	// slice, which the display unwarp then reads. So no MSAA override here.
	if (g_msaaSamples < 0) {
		int s = 2;   // default 2x (VC_MSAA=0 disables, 4/8 also accepted)
		const char *e = getenv("VC_MSAA");
		if (e) s = atoi(e);
		if (s != 0 && s != 2 && s != 4 && s != 8) s = 2;   // clamp to 0/2/4/8
		g_msaaSamples = s;
	}
	return g_msaaSamples;
}

static int
vcMsaaImplicit(void)
{
	static int v = -1;
	if (v < 0) {
		const char *e = getenv("VC_MSAA_IMPLICIT");
		v = e ? (atoi(e) ? 1 : 0) : 1;   // default: implicit resolve
	}
	return v;
}

static int
vcMsaaSplit(void)
{
	if (g_msaaSplit < 0) {
		const char *e = getenv("VC_MSAA_SPLIT");
		g_msaaSplit = e ? (atoi(e) ? 1 : 0) : 0;   // default: shared (split measured and refuted)
	}
	return g_msaaSplit;
}

// Which multisample FBO an eye renders into. Without split both share index 0.
static inline int vcMsaaIdx(int eye) { return (vcMsaaSplit() && eye == 1) ? 1 : 0; }

// Lazily create the multisample FBO(s). Latches failure; logs LOUD once.
static bool
vcrt_msaa_ensure(void)
{
	if (g_msaaReady)  return true;
	if (g_msaaFailed) return false;
	int samples = vcMsaaSamples();
	if (samples == 0) { g_msaaFailed = true; return false; }
	// The implicit path needs none of this: the slice FBO carries the samples itself.
	if (g_msaaImplicitActive) { g_msaaFailed = true; return false; }

	if (!p_msGenRenderbuffers)  p_msGenRenderbuffers  = (PFN_glGenRenderbuffers)g_eglGetProcAddress("glGenRenderbuffers");
	if (!p_msBindRenderbuffer)  p_msBindRenderbuffer  = (PFN_glBindRenderbuffer)g_eglGetProcAddress("glBindRenderbuffer");
	if (!p_msRenderbufferStorageMultisample) p_msRenderbufferStorageMultisample = (PFN_glRenderbufferStorageMultisample)g_eglGetProcAddress("glRenderbufferStorageMultisample");
	if (!p_msBlitFramebuffer)   p_msBlitFramebuffer   = (PFN_glBlitFramebuffer)g_eglGetProcAddress("glBlitFramebuffer");
	if (!p_msGenRenderbuffers || !p_msBindRenderbuffer || !p_msRenderbufferStorageMultisample || !p_msBlitFramebuffer) {
		g_msaaFailed = true;
		VCLOG(@"[vc-msaa] ERROR: entry points unresolved (rbStorageMS=%p blit=%p) -- MSAA OFF, FALLING BACK TO 1x",
		      (void*)p_msRenderbufferStorageMultisample, (void*)p_msBlitFramebuffer);
		return false;
	}

	const int W = g_rtWidth, H = g_rtHeight;
	const int nTargets = vcMsaaSplit() ? 2 : 1;
	GLenumVC eColor = 0, eDepth = 0;

	for (int i = 0; i < nTargets; i++) {
		p_msGenRenderbuffers(1, &g_msaaColorRbo[i]);
		p_msBindRenderbuffer(VC_GL_RENDERBUFFER, g_msaaColorRbo[i]);
		p_msRenderbufferStorageMultisample(VC_GL_RENDERBUFFER, samples, 0x8058 /* GL_RGBA8 */, W, H);
		if (p_glGetError() != 0) eColor = 1;

		p_msGenRenderbuffers(1, &g_msaaDepthRbo[i]);
		p_msBindRenderbuffer(VC_GL_RENDERBUFFER, g_msaaDepthRbo[i]);
		p_msRenderbufferStorageMultisample(VC_GL_RENDERBUFFER, samples, 0x88F0 /* GL_DEPTH24_STENCIL8 */, W, H);
		if (p_glGetError() != 0) eDepth = 1;

		p_glGenFramebuffers(1, &g_msaaFbo[i]);
		p_glBindFramebuffer(VC_GL_FRAMEBUFFER, g_msaaFbo[i]);
		p_glFramebufferRenderbuffer(VC_GL_FRAMEBUFFER, VC_GL_COLOR_ATTACHMENT0,        VC_GL_RENDERBUFFER, g_msaaColorRbo[i]);
		p_glFramebufferRenderbuffer(VC_GL_FRAMEBUFFER, VC_GL_DEPTH_STENCIL_ATTACHMENT, VC_GL_RENDERBUFFER, g_msaaDepthRbo[i]);
		GLenumVC status = p_glCheckFramebufferStatus(VC_GL_FRAMEBUFFER);
		p_glBindFramebuffer(VC_GL_FRAMEBUFFER, 0);

		if (status != VC_GL_FRAMEBUFFER_COMPLETE) {
			g_msaaFailed = true;
			VCLOG(@"[vc-msaa] ERROR: MSAA %dx FBO[%d] INCOMPLETE %dx%d status=%s (colourStorageErr=%u depthStorageErr=%u) -- MSAA OFF, FALLING BACK TO 1x",
			      samples, i, W, H, vcrt_fbo_status_name(status), (unsigned)eColor, (unsigned)eDepth);
			return false;
		}
	}
	if (!vcMsaaSplit()) { g_msaaFbo[1] = g_msaaFbo[0]; g_msaaColorRbo[1] = g_msaaColorRbo[0]; g_msaaDepthRbo[1] = g_msaaDepthRbo[0]; }

	g_msaaReady = true;
	// (colour RGBA8 + depth24stencil8) x samples x W x H, per target.
	unsigned long long perTarget = (unsigned long long)W * H * (4ull + 4ull) * (unsigned)samples;
	VCLOG(@"[vc-msaa] MSAA %dx render target READY %dx%d -- %d target%s (VC_MSAA_SPLIT=%d: %s), %llu MB total; resolve via glBlitFramebuffer (colourStorageErr=%u depthStorageErr=%u)",
	      samples, W, H, nTargets, nTargets == 1 ? "" : "s", vcMsaaSplit(),
	      vcMsaaSplit() ? "one per eye -- no write-after-write between the eye passes"
	                    : "SHARED by both eyes (old behaviour)",
	      perTarget * (unsigned)nTargets / 1000000ull, (unsigned)eColor, (unsigned)eDepth);
	return true;
}

// Resolve the pending eye's multisample content into its single-sample slice FBO.
// Called between eyes (from vc_stereo_eye_fbo) and for the last eye (from
// vc_stereo_restore_main).
extern "C" void
vc_stereo_msaa_resolve_pending(void)
{
	if (g_msaaImplicitActive) return;   // resolved by the pass's store action, nothing to do
	if (!g_msaaReady || g_msaaPending < 0) return;
	int eye = g_msaaPending;
	g_msaaPending = -1;

	GLuintVC srcFbo = g_msaaFbo[vcMsaaIdx(eye)];
	GLuintVC dstFbo = g_stereoBuf[g_currentBack].fbo[eye];
	const int W = g_rtWidth, H = g_rtHeight;

	p_glBindFramebuffer(0x8CA8 /* GL_READ_FRAMEBUFFER */, srcFbo);
	p_glBindFramebuffer(0x8CA9 /* GL_DRAW_FRAMEBUFFER */, dstFbo);
	p_msBlitFramebuffer(0, 0, W, H, 0, 0, W, H, VC_GL_COLOR_BUFFER_BIT, VC_GL_NEAREST);
	GLenumVC err = p_glGetError();
	// Leave both READ+DRAW on the MSAA FBO we just read, so librw's currentFramebuffer
	// cache matches the real binding. With split targets the next eye binds the OTHER
	// FBO, which differs from librw's cached value too -- so the bind is not skipped.
	p_glBindFramebuffer(VC_GL_FRAMEBUFFER, srcFbo);

	static bool loggedOnce = false;
	if (!loggedOnce) {
		loggedOnce = true;
		VCLOG(@"[vc-msaa] first resolve: blit %dx%d MSAA->slice eye=%d srcFbo=%u dstFbo=%u glErr=0x%x %s",
		      W, H, eye, srcFbo, dstFbo, (unsigned)err, err == 0 ? "OK" : "*** BLIT FAILED ***");
	} else if (err != 0) {
		VCLOG(@"[vc-msaa] ERROR: resolve blit eye=%d glErr=0x%x", eye, (unsigned)err);
	}
}

// The eye FBO of the CURRENT back buffer (chosen by vcrt_begin_frame), so the two
// eye passes render into the buffer that will be published this frame. With
// VC_MSAA>0 the eyes render into the shared multisample FBO instead, and the
// previous eye is resolved into its slice here (its RenderScene has finished).
extern "C" unsigned int
vc_stereo_eye_fbo(int eye)
{
	if (!(g_stereoReady && (eye == 0 || eye == 1))) return 0;
	int slot = eye;
	// Implicit MSAA: the slice FBO IS the multisample render target, and Metal resolves
	// into the slice when the pass ends. No shared target, no pending resolve, and -- the
	// point of the change -- no full-screen blit at the eye boundary.
	if (g_msaaImplicitActive) return g_stereoBuf[g_currentBack].fbo[slot];
	if (vcMsaaSamples() > 0 && vcrt_msaa_ensure()) {
		// Resolve the previous eye. With a SHARED target this must happen before we
		// overwrite it; with split targets it is only "the previous eye's content is
		// due" -- the ordering constraint on the render target itself is gone.
		vc_stereo_msaa_resolve_pending();
		g_msaaPending = slot;               // resolve target = slice
		return g_msaaFbo[vcMsaaIdx(slot)];
	}
	return g_stereoBuf[g_currentBack].fbo[slot];
}

// One-time proof that the two eye passes wrote DIFFERENT content into the two
// slices of the current back buffer: Metal-blit each slice into a shared 2D
// staging texture and log sampled pixels. Identical => both passes rendered the
// same => per-eye matrices did not take.
extern "C" void
vc_stereo_readback_log(void)
{
	static bool done = false;
	if (done || !g_stereoReady) return;
	done = true;

	id<MTLTexture> arr = g_stereoBuf[g_currentBack].arrayTex;
	if (arr == nil) return;
	if (p_glFinish) p_glFinish();   // submit the eye passes to ANGLE's queue first
	if (g_cmdQueue == nil) g_cmdQueue = [g_mtlDevice newCommandQueue];

	MTLTextureDescriptor *sd =
		[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                   width:g_rtWidth height:g_rtHeight mipmapped:NO];
	sd.usage = MTLTextureUsageShaderRead;
	sd.storageMode = MTLStorageModeShared;

	// Sample three points (centre + two off-centre); a horizontal eye offset only
	// changes a pixel where there's parallax, so one sky-flat point could read
	// equal by chance. DIFFER if ANY point differs.
	const int pts[3][2] = { { g_rtWidth/2, g_rtHeight/2 },
	                        { g_rtWidth/4, g_rtHeight/2 },
	                        { g_rtWidth/2, g_rtHeight/4 } };
	uint8_t px[2][3][4];
	memset(px, 0, sizeof(px));
	for (int s = 0; s < 2; s++) @autoreleasepool {
		id<MTLTexture> staging = [g_mtlDevice newTextureWithDescriptor:sd];
		id<MTLCommandBuffer> cb = [g_cmdQueue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit copyFromTexture:arr sourceSlice:s sourceLevel:0
		         sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(g_rtWidth, g_rtHeight, 1)
		            toTexture:staging destinationSlice:0 destinationLevel:0
		    destinationOrigin:MTLOriginMake(0,0,0)];
		[blit endEncoding]; [cb commit]; [cb waitUntilCompleted];
		for (int p = 0; p < 3; p++)
			[staging getBytes:px[s][p] bytesPerRow:4
			       fromRegion:MTLRegionMake2D(pts[p][0], pts[p][1], 1, 1) mipmapLevel:0];
	}
	bool differ = memcmp(px[0], px[1], sizeof(px[0])) != 0;
	VCLOG(@"[vc-stereo] readback (centre)  slice0 RGBA=%d,%d,%d,%d  slice1 RGBA=%d,%d,%d,%d  -> %s",
	      px[0][0][0],px[0][0][1],px[0][0][2],px[0][0][3], px[1][0][0],px[1][0][1],px[1][0][2],px[1][0][3],
	      differ ? "DIFFER (eyes rendered separately)" : "IDENTICAL (per-eye matrices did NOT take)");
	VCLOG(@"[vc-stereo] readback (offc.)   slice0 RGBA=%d,%d,%d,%d / %d,%d,%d,%d  slice1 RGBA=%d,%d,%d,%d / %d,%d,%d,%d",
	      px[0][1][0],px[0][1][1],px[0][1][2],px[0][1][3], px[0][2][0],px[0][2][1],px[0][2][2],px[0][2][3],
	      px[1][1][0],px[1][1][1],px[1][1][2],px[1][1][3], px[1][2][0],px[1][2][1],px[1][2][2],px[1][2][3]);
}

// Throttled (1/s) probe of the JUST-PUBLISHED stereo buffer, on the game thread.
// Reads back the centre pixel of both slices of g_stereoBuf[idx] -- the exact
// buffer/index the compositor will acquire this cycle. Non-black => the eye
// passes produced content and the black is in the handoff/display; black =>
// the eye passes themselves wrote nothing (producer). glFinish first so the GL
// work is complete before the Metal blit reads the texture.
extern "C" void
vcrt_stereo_publish_probe(int idx)
{
	if (!g_stereoReady || idx < 0 || idx >= g_numBuffers) return;
	static double lastProbe = 0.0;
	double now = vc_now_seconds();
	if (now - lastProbe < 1.0) return;
	lastProbe = now;

	id<MTLTexture> arr = g_stereoBuf[idx].arrayTex;
	if (arr == nil || g_mtlDevice == nil) return;
	if (p_glFinish) p_glFinish();
	if (g_cmdQueue == nil) g_cmdQueue = [g_mtlDevice newCommandQueue];

	MTLTextureDescriptor *sd =
		[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                   width:g_rtWidth height:g_rtHeight mipmapped:NO];
	sd.usage = MTLTextureUsageShaderRead;
	sd.storageMode = MTLStorageModeShared;

	uint8_t px[2][4] = {{0,0,0,0},{0,0,0,0}};
	// Per-iteration pool: the command buffer is autoreleased and retains its
	// staging texture; without a draining pool on the game thread they accumulate.
	for (int s = 0; s < 2; s++) @autoreleasepool {
		id<MTLTexture> staging = [g_mtlDevice newTextureWithDescriptor:sd];
		id<MTLCommandBuffer> cb = [g_cmdQueue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit copyFromTexture:arr sourceSlice:s sourceLevel:0
		         sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(g_rtWidth, g_rtHeight, 1)
		            toTexture:staging destinationSlice:0 destinationLevel:0
		    destinationOrigin:MTLOriginMake(0,0,0)];
		[blit endEncoding]; [cb commit]; [cb waitUntilCompleted];
		[staging getBytes:px[s] bytesPerRow:4
		       fromRegion:MTLRegionMake2D(g_rtWidth/2, g_rtHeight/2, 1, 1) mipmapLevel:0];
	}
	VCLOG(@"[vc-stereo] PUBLISH buf=%d tex=%p wait=%llu eye0 RGBA=%d,%d,%d,%d  eye1 RGBA=%d,%d,%d,%d (throttled 1/s)",
	      idx, VC_OBJ_TO_VOID(arr), (unsigned long long)g_buf[idx].waitValue,
	      px[0][0],px[0][1],px[0][2],px[0][3], px[1][0],px[1][1],px[1][2],px[1][3]);
}

#endif // LIBRW_VISIONOS
