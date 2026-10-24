#ifndef BC_OVERLAY_H
#define BC_OVERLAY_H

#include "detect.h"   /* BC_MAX_DET */

/*
 * The most boxes one present may carry.
 *
 * Not the track count. A tracked subject contributes its own cover and, when
 * eye bars are on, a second box drawn on top of it -- so the ceiling is twice
 * the tracker's capacity, plus one for the pinned debug box. Sized to the
 * track count alone, the caller's array overflowed as soon as half the tracks
 * grew a bar: the bar push was bounds-checked and the cover push was not, so
 * the write that went past the end was the one that always happens.
 */
#define BC_MAX_BOXES (BC_MAX_DET * 2 + 1)

/*
 * Click-through always-on-top overlay bound to a single output (DESIGN.md §3[7]).
 *
 * Uses wlr-layer-shell on the OVERLAY layer with an empty input region, so the
 * surface floats above everything (including fullscreen windows) yet never
 * receives a click -- pointer events fall through to whatever is underneath.
 *
 * Boxes are in output *pixel* space, i.e. the same coordinate system as the
 * frame bc_capture hands us. That is deliberate: in Phase 3 detector output
 * feeds straight in with no rescaling step to get wrong.
 */

enum bc_effect {
	BC_EFFECT_FILL = 0,   /* solid colour (r,g,b,a); the default */
	BC_EFFECT_PIXELATE,   /* block-average of the captured screen */
	BC_EFFECT_BLUR,       /* wide box blur of the captured screen */
	BC_EFFECT_NOISE,      /* animated static; samples nothing, so always covers */
	BC_EFFECT_GLITCH,     /* torn strips of the pixelated screen */
	BC_EFFECT_RGBSHIFT,   /* channel-split over a pixelated base */
	BC_EFFECT_TAPE,       /* hazard-striped bars spanning the screen */
	BC_EFFECT_HALFTONE,   /* print-style dots over a pixelated base */
	BC_EFFECT_SOBEL,      /* edge outlines of a pixelated base, on black */
	BC_EFFECT_TRIANGLE,   /* triangular mosaic */
	BC_EFFECT_CLUSTER,    /* mass of discs; samples nothing */
	BC_EFFECT_SPLATTER,   /* fused ink blobs; samples nothing */
	BC_EFFECT_SHAPE,      /* hexagonal tiling of the screen's own colours */
	BC_EFFECT_CAPTION,    /* opaque bar with a word on it; samples nothing */
	BC_EFFECT_WALL,       /* box filled with characters; samples nothing */
};

struct bc_box {
	int x, y, w, h;      /* top-left origin, pixels */
	float r, g, b, a;    /* premultiplied alpha -- Wayland requires it */
	int   effect;        /* enum bc_effect */
	float strength;      /* pixelate: block size as % of min(w,h), per §5 */
	float angle;         /* radians, CCW about the box centre; 0 = upright */

	/*
	 * Caption text, for BC_EFFECT_CAPTION. Empty is legal and draws a
	 * plain bar -- the bar is what covers, so there is no case where
	 * missing text means missing coverage.
	 */
	char  text[24];
	/*
	 * Glyph colour for those two effects.
	 *   0 -- the effect's own colour; text_rgb is not read
	 *   1 -- solid text_rgb
	 *   2 -- a hue that moves across the box and over time
	 *
	 * Zero is deliberately "leave it as it was", so a box built without a
	 * thought for text keeps the look it has always had rather than
	 * defaulting to black ink on a black bar.
	 */
	int   text_mode;
	float text_rgb[3];

	/*
	 * The censor *shape*, as opposed to what fills it.
	 *
	 * Both are drawn outward from the box, never into it. The outline adds
	 * opaque covered area, so it can only ever help; the shadow lies beyond
	 * the outline, over screen that was never censored, so it cannot uncover
	 * anything either. Feathering or insetting them would put a partly
	 * transparent ring over content the detector did flag, which is the same
	 * exposure the absence of fade-in exists to prevent.
	 */
	float outline;       /* px outward; 0 = none */
	float outline_rgb[3];
	float shadow;        /* px of falloff beyond the outline; 0 = none */
	float shadow_a;      /* peak shadow alpha */
	int   auto_color;    /* sample the fill colour from around the box */
	/*
	 * Draw this box on top of everything, and never punch it out.
	 *
	 * In reverse mode every box is a hole in the mask -- the detections are
	 * what gets revealed. An eye bar is the exception: it has to stay an
	 * opaque cover in both modes, or "reveal the face but bar the eyes"
	 * would reveal the eyes too. Set on the bar, so it is excluded from the
	 * cutouts and drawn after them.
	 */
	int   over;
	/*
	 * Clip this box to the subject silhouette from the segmentation model,
	 * so the cover follows the subject's shape instead of being the
	 * rectangle that makes these tools recognisable on sight.
	 *
	 * Opt-in per box, and deliberately so: it is the only thing here that
	 * *removes* covered area, and everything else in this file is built the
	 * other way round. A box that sets it and finds no mask bound is covered
	 * whole, which is the direction §6 requires.
	 */
	int   use_mask;
};


struct bc_overlay;

/*
 * How the overlay surface is mapped.
 *
 * LAYER   -- wlr-layer-shell on the OVERLAY layer. Always above everything,
 *            including fullscreen windows, and invisible to the task manager.
 *            It is therefore *not* excludable from screen capture, so our own
 *            censor bars come back round through the capture and blind the
 *            detector (see DESIGN.md §3[7]).
 *
 * TOPLEVEL-- an ordinary fullscreen xdg-toplevel. It shows up in the task
 *            manager, which is the point: KWin's per-window "Hide from
 *            Screencast" is only offered for managed windows. With that set,
 *            the detector sees a pristine screen and every workaround for the
 *            feedback loop becomes unnecessary. Costs a task-manager entry and
 *            needs a keep-above window rule to stack reliably.
 */
enum bc_overlay_mode {
	BC_OVERLAY_LAYER = 0,
	BC_OVERLAY_TOPLEVEL,
};

/*
 * Connect to the compositor and map a surface on the output whose connector
 * name matches `output_name` (e.g. "DP-3"). Pass NULL to take whichever output
 * the compositor considers first.
 *
 * Returns NULL on failure. Does not return until the surface is configured.
 */
struct bc_overlay *bc_overlay_create_ex(const char *output_name,
                                        enum bc_overlay_mode mode);

/* Convenience: layer-shell mode. */
struct bc_overlay *bc_overlay_create(const char *output_name);
void bc_overlay_destroy(struct bc_overlay *o);

/* Buffer dimensions in pixels (logical size * scale). */
int bc_overlay_width(const struct bc_overlay *o);
int bc_overlay_height(const struct bc_overlay *o);

/*
 * Present one frame with nothing drawn on it, so the capture can see the
 * screen as it really is.
 *
 * The only way to learn whether content under a cover is still there, on a
 * compositor that cannot hide this surface from the capture: while the cover
 * is up the detector sees our bar, and its silence is not evidence.
 *
 * Rate-limited *here*, at the point of drawing, rather than by the caller.
 * Repeatedly uncovering the screen is a photosensitive-seizure risk -- the
 * hazardous band is roughly 3-60 Hz, worst around 15-25 Hz, for high-contrast
 * changes over a large area, which is exactly what a full-screen censor bar
 * blinking would be. A limit the caller is trusted to respect is one bug away
 * from a hazard, so the refusal lives at the bottom where nothing can route
 * around it.
 *
 * Returns 1 if the empty frame was presented, 0 if it was refused as too soon.
 * A refusal is not an error: it means the screen stays covered, which is the
 * safe answer to every question this asks.
 */
int bc_overlay_peek(struct bc_overlay *o, double now_ms);
unsigned long bc_overlay_peek_count(const struct bc_overlay *o);

/* Smallest gap between peeks, in ms. 500 = 2 Hz, below the 3 Hz floor of the
 * flash guidance. Deliberately a constant and not a setting. */
#define BC_PEEK_MIN_MS 500.0

/* The output's own mode and name, for checking the capture matches the screen
 * being drawn on. See bc_overlay_output_mode() in overlay.c. */
void        bc_overlay_output_mode(const struct bc_overlay *o, int *w, int *h);
const char *bc_overlay_output_name(const struct bc_overlay *o);

/*
 * The EGLDisplay this overlay owns, with its context already current.
 * The joined pipeline hands this to bc_gpu_create_shared() so capture imports
 * onto the same display -- EGLImages cannot cross displays.
 */
void *bc_overlay_egl_display(const struct bc_overlay *o);

/*
 * Give the overlay the captured screen texture to pixelate from. Must be a
 * texture on this overlay's EGLDisplay. Pass tex=0 to disable, in which case
 * BC_EFFECT_PIXELATE degrades to a solid fill -- fail-closed, never to
 * "draw nothing".
 */
void bc_overlay_set_source(struct bc_overlay *o, unsigned tex, int w, int h);

/*
 * Present the source frame (see bc_overlay_set_source) as an opaque full-screen
 * background beneath the boxes. This is the delayed screen: the surface stops
 * being a transparent layer over the live desktop and becomes the picture
 * itself, so it must only be on while a delayed frame is actually being shown.
 */
void bc_overlay_set_background(struct bc_overlay *o, int on);

/*
 * The subject silhouette that `use_mask` boxes are clipped to.
 *
 * `px` is one byte per pixel, w*h, in the segmentation model's letterboxed
 * input space; NULL clears it, after which every box is covered whole again.
 * The texture is owned here so the caller needs no GL of its own.
 *
 * The uv pair maps the screen onto it: `u0,v0` is where the bottom-left of the
 * output lands and `u1,v1` the top-right. The letterbox padding is carried in
 * those numbers, so nothing here has to know the model's geometry.
 */
void bc_overlay_set_silhouette(struct bc_overlay *o, const unsigned char *px,
                               int w, int h,
                               float u0, float v0, float u1, float v1);

/*
 * Draw `n` boxes and present. Returns 0 on success, -1 on a fatal protocol or
 * EGL error, and 1 if the compositor asked the surface to close.
 */
/*
 * Reverse mode: mask the whole output with the current effect and cut the
 * detected boxes *out* of it, showing them clear. This inverts what the
 * program is for, so the caller must turn it off for any fail-closed path --
 * see the note in bc_overlay_present().
 */
void bc_overlay_set_reverse(struct bc_overlay *o, int on);

/*
 * Style for the reverse mask: the effect, strength and text it is drawn with.
 * Geometry is ignored -- the mask is always the whole output.
 *
 * This is set separately from the boxes on purpose. The mask used to copy its
 * style from boxes[0], which meant its appearance depended on a detection
 * existing: the moment the count hit zero, even for one frame, there was no
 * box to copy and the mask fell back to a zeroed one -- solid black, whole
 * screen. At display rate that is a strobe, and it went uncounted because it
 * is not the fail-closed path. Coverage was never lost, but a full-screen
 * black flash is its own hazard.
 *
 * Until this is called the mask is opaque black, so the failure direction of
 * forgetting it is "covered", not "revealed".
 */
void bc_overlay_set_mask(struct bc_overlay *o, const struct bc_box *style);

int bc_overlay_present(struct bc_overlay *o, const struct bc_box *boxes, int n);

/* --- drawing mode input ---------------------------------------------------
 *
 * The overlay is click-through because its input region is empty, and that is
 * load-bearing: on the OVERLAY layer a surface that accepts input swallows
 * every click on the monitor, which locks the user out of their desktop. So
 * input is off unless something asks for it, and asking is expected to be
 * brief -- the one caller is the region-drawing mode, which gives it straight
 * back.
 *
 * Drawing is done *here*, in the same surface that draws the censors, rather
 * than in a window of the GUI's. A separate window would stack somewhere
 * relative to the overlay, and if it landed above it, the screen it is
 * covering would be uncensored in the capture for as long as the user spent
 * aiming. There is no stacking question if there is only one surface.
 */

enum bc_key {
	BC_KEY_NONE = 0,
	BC_KEY_ENTER,
	BC_KEY_ESC,
};

/*
 * Take (on) or give back (off) pointer and keyboard input.
 *
 * Off is the resting state and the one every failure path must reach. Nothing
 * here reaches it on its own -- the caller decides -- so the caller owes the
 * user an exit that does not depend on this surface having focus. bc_main.c's
 * is the config flag, which the GUI can clear from its own window.
 */
void bc_overlay_set_input(struct bc_overlay *o, int on);

/*
 * Where the pointer is, in the same pixel space as bc_box. Returns 0 if it is
 * not over us -- during which the caller has no cursor to draw and must not
 * place anything, since the last known position is somewhere it no longer is.
 */
int bc_overlay_pointer(const struct bc_overlay *o, int *x, int *y);

/*
 * Pop one key press, or BC_KEY_NONE. Only Enter and Escape are reported: they
 * are read as raw evdev codes, which is exact for those two and saves a
 * dependency on xkbcommon for a mode that has no text entry in it.
 */
int bc_overlay_next_key(struct bc_overlay *o);

/*
 * The left mouse button: returns 1 while it is held, and reports how many
 * press and release edges have happened since the last call.
 *
 * Both, because the caller needs both. A freehand stroke follows the held
 * state, but a *click* is press and release inside a frame or two, and a
 * caller sampling only the level would miss it entirely. The counts are
 * cleared by reading them.
 */
int bc_overlay_button(struct bc_overlay *o, int *pressed, int *released);

#endif /* BC_OVERLAY_H */
