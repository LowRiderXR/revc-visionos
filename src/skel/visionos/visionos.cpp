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

#include "common.h"
#include "rwcore.h"
#include "skeleton.h"
#include "platform.h"
#include "crossplatform.h"

// Hardcoded drawable resolution of the Vision Pro (per-eye). One video mode.
#define VISIONOS_SCREEN_WIDTH  2048
#define VISIONOS_SCREEN_HEIGHT 1984

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

	// TODO(visionos): echtes Startup fehlt noch (kommt mit FS-/Kontext-Schritt):
	//   CFileMgr::Initialise(); InitialiseLanguage();
	//   C_PcSave::SetSaveDirectory(_psGetUserFilesFolder());
	//   FrontEndMenuManager.LoadSettings(); TheText.Unload();
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
	// TODO(visionos): echtes RwCameraBeginUpdate, sobald der ANGLE-Kontext steht.
	return TRUE;
}

void
psCameraShowRaster(RwCamera *camera)
{
	// TODO(visionos): kein SwapBuffers; das Ergebnis wird ausserhalb von reVC
	// in eine Metal-Textur geblittet.
	return;
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

void
CapturePad(RwInt32 padID)
{
	// TODO(visionos): Pad-Status ueber GCController einlesen; vorerst nichts.
	return;
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

static RwChar  _VMString[] = "2048 X 1984 X 32";
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
	// TODO(visionos): Sprach-/Locale-Erkennung (setlocale) uebernehmen; vorerst
	// bleibt die Default-Sprache (American) aktiv.
	return;
}

// ===========================================================================
// File-system helpers (nicht in platform.h, aber vom Kern forward-declared:
// core/FileMgr.cpp, save/PCSave.cpp, save/MemoryCard.cpp).
// ===========================================================================
const char *
_psGetUserFilesFolder()
{
	// TODO(visionos): echten, beschreibbaren Pfad liefern (App-Sandbox Documents).
	// Leerer, gueltiger C-String, damit strcpy/strcat nicht auf nil laufen.
	static char userFiles[] = "";
	return userFiles;
}

void
_psCreateFolder(const char *path)
{
	// TODO(visionos): Verzeichnis anlegen (mkdir im Sandbox-Pfad).
	return;
}

#endif // LIBRW_VISIONOS
