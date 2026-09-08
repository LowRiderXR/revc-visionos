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
	return true;
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
// Phase 5.5 stereo target: DOUBLE-BUFFERED to share the cinema state machine.
// Per back buffer i: ONE MTLTexture (2D array, arrayLength=2) whose slices are
// the two eyes, plus a GL FBO per slice. One DEDICATED depth renderbuffer shared
// by all FBOs and cleared per eye pass (passes are sequential on this GL thread;
// never the cinema depth). The eye textures are tied to the SAME index as the
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
static GLuintVC       g_stereoDepthRbo = 0;
static bool           g_stereoReady    = false;
static bool           g_stereoFailed   = false;

extern "C" bool  vc_stereo_ready(void) { return g_stereoReady; }
extern "C" void *vc_stereo_array_texture(int idx)
{
	if (!g_stereoReady || idx < 0 || idx >= g_numBuffers) return NULL;
	return VC_OBJ_TO_VOID(g_stereoBuf[idx].arrayTex);
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

	// One dedicated depth renderbuffer, shared by all eye FBOs, cleared per pass.
	glGenRenderbuffers_(1, &g_stereoDepthRbo);
	glBindRenderbuffer_(VC_GL_RENDERBUFFER, g_stereoDepthRbo);
	glRenderbufferStorage_(VC_GL_RENDERBUFFER, 0x88F0 /* GL_DEPTH24_STENCIL8 */, W, H);

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
			p_glFramebufferTexture2D(VC_GL_FRAMEBUFFER, VC_GL_COLOR_ATTACHMENT0, VC_GL_TEXTURE_2D, b->glTex[s], 0);
			p_glFramebufferRenderbuffer(VC_GL_FRAMEBUFFER, VC_GL_DEPTH_STENCIL_ATTACHMENT, VC_GL_RENDERBUFFER, g_stereoDepthRbo);
			GLenumVC status = p_glCheckFramebufferStatus(VC_GL_FRAMEBUFFER);
			VCLOG(@"[vc-stereo] buffer %d slice %d: %dx%d RGBA8 gltex %u fbo %u bindErr=0x%x status=%s",
			      i, s, W, H, b->glTex[s], b->fbo[s], (unsigned)bindErr, vcrt_fbo_status_name(status));
		}
	}
	p_glBindFramebuffer(VC_GL_FRAMEBUFFER, 0);
	g_stereoReady = true;
	VCLOG(@"[vc-stereo] array render target ready (%d buffers x 2 slices, %dx%d, dedicated depth)",
	      g_numBuffers, W, H);
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
static GLuintVC g_msaaFbo      = 0;
static GLuintVC g_msaaColorRbo = 0;
static GLuintVC g_msaaDepthRbo = 0;
static bool     g_msaaReady    = false;
static bool     g_msaaFailed   = false;
static int      g_msaaPending  = -1;    // eye whose content sits in g_msaaFbo awaiting resolve
static PFN_glGenRenderbuffers               p_msGenRenderbuffers  = NULL;
static PFN_glBindRenderbuffer               p_msBindRenderbuffer  = NULL;
static PFN_glRenderbufferStorageMultisample p_msRenderbufferStorageMultisample = NULL;
static PFN_glBlitFramebuffer                p_msBlitFramebuffer   = NULL;

extern "C" void vc_stereo_msaa_resolve_pending(void);

static int
vcMsaaSamples(void)
{
	if (g_msaaSamples < 0) {
		int s = 2;   // default 2x (VC_MSAA=0 disables, 4/8 also accepted)
		const char *e = getenv("VC_MSAA");
		if (e) s = atoi(e);
		if (s != 0 && s != 2 && s != 4 && s != 8) s = 2;   // clamp to 0/2/4/8
		g_msaaSamples = s;
	}
	return g_msaaSamples;
}

// Lazily create the shared multisample FBO. Latches failure; logs LOUD once.
static bool
vcrt_msaa_ensure(void)
{
	if (g_msaaReady)  return true;
	if (g_msaaFailed) return false;
	int samples = vcMsaaSamples();
	if (samples == 0) { g_msaaFailed = true; return false; }

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

	p_msGenRenderbuffers(1, &g_msaaColorRbo);
	p_msBindRenderbuffer(VC_GL_RENDERBUFFER, g_msaaColorRbo);
	p_msRenderbufferStorageMultisample(VC_GL_RENDERBUFFER, samples, 0x8058 /* GL_RGBA8 */, W, H);
	GLenumVC eColor = p_glGetError();

	p_msGenRenderbuffers(1, &g_msaaDepthRbo);
	p_msBindRenderbuffer(VC_GL_RENDERBUFFER, g_msaaDepthRbo);
	p_msRenderbufferStorageMultisample(VC_GL_RENDERBUFFER, samples, 0x88F0 /* GL_DEPTH24_STENCIL8 */, W, H);
	GLenumVC eDepth = p_glGetError();

	p_glGenFramebuffers(1, &g_msaaFbo);
	p_glBindFramebuffer(VC_GL_FRAMEBUFFER, g_msaaFbo);
	p_glFramebufferRenderbuffer(VC_GL_FRAMEBUFFER, VC_GL_COLOR_ATTACHMENT0,        VC_GL_RENDERBUFFER, g_msaaColorRbo);
	p_glFramebufferRenderbuffer(VC_GL_FRAMEBUFFER, VC_GL_DEPTH_STENCIL_ATTACHMENT, VC_GL_RENDERBUFFER, g_msaaDepthRbo);
	GLenumVC status = p_glCheckFramebufferStatus(VC_GL_FRAMEBUFFER);
	p_glBindFramebuffer(VC_GL_FRAMEBUFFER, 0);

	if (status != VC_GL_FRAMEBUFFER_COMPLETE) {
		g_msaaFailed = true;
		VCLOG(@"[vc-msaa] ERROR: MSAA %dx FBO INCOMPLETE %dx%d status=%s (colourStorageErr=0x%x depthStorageErr=0x%x) -- MSAA OFF, FALLING BACK TO 1x",
		      samples, W, H, vcrt_fbo_status_name(status), (unsigned)eColor, (unsigned)eDepth);
		return false;
	}

	g_msaaReady = true;
	VCLOG(@"[vc-msaa] MSAA %dx render target READY %dx%d (colour+depth multisample renderbuffers, resolve via glBlitFramebuffer; colourStorageErr=0x%x depthStorageErr=0x%x)",
	      samples, W, H, (unsigned)eColor, (unsigned)eDepth);
	return true;
}

// Resolve the pending eye's multisample content into its single-sample slice FBO.
// Called between eyes (from vc_stereo_eye_fbo) and for the last eye (from
// vc_stereo_restore_main). Leaves GL_FRAMEBUFFER bound to g_msaaFbo so librw's
// currentFramebuffer cache stays coherent (the next eye binds g_msaaFbo; a cache
// early-out is then safe because the real binding already matches).
extern "C" void
vc_stereo_msaa_resolve_pending(void)
{
	if (!g_msaaReady || g_msaaPending < 0) return;
	int eye = g_msaaPending;
	g_msaaPending = -1;

	GLuintVC dstFbo = g_stereoBuf[g_currentBack].fbo[eye];
	const int W = g_rtWidth, H = g_rtHeight;

	p_glBindFramebuffer(0x8CA8 /* GL_READ_FRAMEBUFFER */, g_msaaFbo);
	p_glBindFramebuffer(0x8CA9 /* GL_DRAW_FRAMEBUFFER */, dstFbo);
	p_msBlitFramebuffer(0, 0, W, H, 0, 0, W, H, VC_GL_COLOR_BUFFER_BIT, VC_GL_NEAREST);
	GLenumVC err = p_glGetError();
	p_glBindFramebuffer(VC_GL_FRAMEBUFFER, g_msaaFbo);   // leave both READ+DRAW = MSAA, cache-coherent

	static bool loggedOnce = false;
	if (!loggedOnce) {
		loggedOnce = true;
		VCLOG(@"[vc-msaa] first resolve: blit %dx%d MSAA->slice eye=%d dstFbo=%u glErr=0x%x %s",
		      W, H, eye, dstFbo, (unsigned)err, err == 0 ? "OK" : "*** BLIT FAILED ***");
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
	if (vcMsaaSamples() > 0 && vcrt_msaa_ensure()) {
		vc_stereo_msaa_resolve_pending();   // resolve the previous eye before reusing the shared FBO
		g_msaaPending = eye;
		return g_msaaFbo;
	}
	return g_stereoBuf[g_currentBack].fbo[eye];
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
