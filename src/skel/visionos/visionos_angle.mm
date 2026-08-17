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

// EGL state owned here (the context is created here and handed to librw).
static void      *g_libEGL    = NULL;
static void      *g_libGLESv2 = NULL;
static EGLDisplay g_display   = NULL;
static EGLContext g_context   = NULL;

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

	// 5) Make current, surfaceless (EGL_NO_SURFACE for draw and read).
	if (!eglMakeCurrent(g_display, (EGLSurface)0, (EGLSurface)0, g_context)) {
		VCLOG(@"FAIL eglMakeCurrent surfaceless (egl error 0x%x)", eglGetError());
		return false;
	}
	VCLOG(@"eglMakeCurrent surfaceless OK");

	// 6) Diagnostic: log ANGLE's backing MTLDevice name (non-fatal).
	PFN_eglQueryDisplayAttribEXT eglQueryDisplayAttribEXT =
		(PFN_eglQueryDisplayAttribEXT)eglGetProcAddress("eglQueryDisplayAttribEXT");
	PFN_eglQueryDeviceAttribEXT eglQueryDeviceAttribEXT =
		(PFN_eglQueryDeviceAttribEXT)eglGetProcAddress("eglQueryDeviceAttribEXT");
	if (eglQueryDisplayAttribEXT && eglQueryDeviceAttribEXT) {
		EGLAttrib eglDevice = 0, mtl = 0;
		if (eglQueryDisplayAttribEXT(g_display, VC_EGL_DEVICE_EXT, &eglDevice) &&
		    eglQueryDeviceAttribEXT((EGLDeviceEXT)eglDevice, VC_EGL_METAL_DEVICE_ANGLE, &mtl) && mtl) {
		#if __has_feature(objc_arc)
			id<MTLDevice> device = (__bridge id<MTLDevice>)(void *)mtl;
		#else
			id<MTLDevice> device = (id<MTLDevice>)(void *)mtl;
		#endif
			VCLOG(@"ANGLE MTLDevice = '%@'", device.name);
		} else {
			VCLOG(@"MTLDevice query returned nothing (non-fatal)");
		}
	} else {
		VCLOG(@"eglQuery*AttribEXT unavailable (non-fatal)");
	}

	if (outGetProcAddress) *outGetProcAddress = (void *)eglGetProcAddress;
	VCLOG(@"ANGLE ready; handing eglGetProcAddress to librw");
	return true;
}

#endif // LIBRW_VISIONOS
