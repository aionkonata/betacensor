#ifndef BC_OFFLINE_H
#define BC_OFFLINE_H

/*
 * The headless half shared by the two offline passes, bc-video and bc-image.
 *
 * Both take pixels from a decoder pipe instead of a PipeWire capture, run the
 * same detector over them, and draw the same effects into an FBO instead of a
 * Wayland surface. Everything that is the same between the two lives here, for
 * the reason struct bc_box is shared with the live overlay: the copy nobody
 * runs is the one that drifts, and an effect that renders differently in a
 * photo than in a video of the same config is a bug nobody would think to
 * look for.
 */

#include "config.h"
#include "detect.h"
#include "overlay.h"
#include "preprocess.h"

#include <EGL/egl.h>
#include <stddef.h>

#include <stdint.h>

struct bc_off_gl {
	EGLDisplay dpy;
	EGLContext ctx;
	unsigned   prog, blit, vao, harden;
	unsigned   fbo, fbo_tex, src_tex, font_tex, work_tex;
	int        h_block, h_seed, h_amp, h_glyph, h_mark;
	int        h_box, h_angle, h_turned;
	int        u_size, u_block, u_alpha, u_effect, u_time, u_box;
	int        u_feather, u_angle, u_color, u_text, u_text_len;
	int        u_text_mode, u_text_col;
	int        b_size;
	int        w, h;
	int        max_tex;   /* GL_MAX_TEXTURE_SIZE: the largest side we can hold */
};

/* A surfaceless context and a w x h render target. 0 on success. */
int  bc_off_gl_init(struct bc_off_gl *g, int w, int h);

/*
 * Re-size the source texture and render target, for a pass whose frames do
 * not all share one size -- every image in a folder is its own size. A no-op
 * when the size is unchanged. Fails, rather than clamping, on a side larger
 * than the GPU can hold: a clamped texture would be a different picture.
 */
int  bc_off_gl_resize(struct bc_off_gl *g, int w, int h);

/* Upload a top-down, tightly packed RGB24 frame of the current size. */
void bc_off_upload(struct bc_off_gl *g, const unsigned char *rgb);

/*
 * The uploaded frame with `n` boxes drawn over it, into the render target.
 * `t` is effect time in seconds (the animated effects use it).
 */
void bc_off_draw(struct bc_off_gl *g, const struct bc_box *b, int n, double t);

/*
 * Make the drawn covers harder to undo. Run after bc_off_draw(), over the
 * same boxes; only effects that sample the picture beneath are touched --
 * fill, static, tape, captions and the rest sample nothing, so there is
 * nothing under them to recover.
 *
 * A mosaic or a blur is a low-resolution copy of what it covers, and the
 * tools that reverse them (block matching in the Depix style, restoration
 * models) lean on its values being exact. This shifts every block by its
 * own random amount -- on the pixelate effect's own grid, so each block's
 * mean is moved, not merely speckled -- adds fine grain, and tiles a faint
 * diagonal BETACENSOR mark across the cover.
 *
 * `seed` decides the pattern. It must not be guessable: a known seed is a
 * known pattern, and a known pattern can be subtracted. The pattern is fixed
 * to the frame, not to time, so averaging the frames of a still scene does
 * not cancel it.
 */
void bc_off_harden(struct bc_off_gl *g, const struct bc_box *b, int n,
                   uint32_t seed);

/* Is pixel (x, y), top-down, inside box `b` -- turned by its angle if it has
 * one? The same test the shader makes, for code that has to agree with it. */
int  bc_off_box_has(const struct bc_box *b, int x, int y);

/* Does this overlay effect sample the content under it? */
int  bc_off_effect_samples(int effect);

/* Read the render target back as top-down RGB24, ready for an encoder. */
void bc_off_read(struct bc_off_gl *g, unsigned char *rgb);

/*
 * The "censored with BetaCensor" badge: white text on a black rectangle in
 * the bottom-right corner, sized to the frame. Rendered once per frame size
 * (by ffmpeg's drawtext, for a real font) and then copied into each frame.
 * Falls back to the built-in bitmap font when drawtext or a font file is not
 * available, so a missing package costs looks, never the badge.
 */
struct bc_off_badge {
	int            fw, fh;        /* frame size it was made for */
	int            x, y, w, h;    /* where it goes */
	unsigned char *rgb;           /* w*h*3, top-down */
	int            bitmap;        /* drawn with the fallback font */
};

/* Make (or keep) the badge for a fw x fh frame. -1 if it cannot fit. */
int  bc_off_badge_make(struct bc_off_badge *b, int fw, int fh);
/* Copy it into a top-down RGB24 frame of the size it was made for. */
void bc_off_badge_apply(const struct bc_off_badge *b, unsigned char *rgb);
void bc_off_badge_free(struct bc_off_badge *b);

/*
 * A run's hardening seed: BC_HARDEN_SEED (hex) when set -- for reproducible
 * runs and for the GUI, whose preview has to match its batch -- otherwise
 * fresh from the kernel's random source.
 */
uint64_t bc_off_run_seed(void);
/* Fold a per-image value into the run seed. */
uint32_t bc_off_mix_seed(uint64_t run, uint64_t v);

/*
 * Run the detector over the uploaded frame and wait for its outcome.
 *
 * The detector is asynchronous because the live path must not block on it.
 * Offline there is nothing else to do with the time, and skipping a frame
 * would be a hole in the coverage, so this waits. *hung is set when no outcome
 * arrived at all within ten seconds -- the caller must then stop rather than
 * submit again, or the late result is handed to the wrong frame.
 *
 * Anything but BC_DET_OK means nobody looked at this frame, and §6 says the
 * caller must treat it as unexamined, never as clean.
 */
struct bc_det_result bc_off_detect(struct bc_off_gl *g, struct bc_preproc *pp,
                                   struct bc_detector *det, float *tensor,
                                   struct bc_det *dets, int max, int *hung);

/* Config effect index -> overlay effect. */
int  bc_off_effect(int cfg_effect);

/*
 * A box styled for class `cls` the way the live path styles it: the class's
 * own effect, that effect's strength, the caption. Geometry is the caller's.
 */
struct bc_box bc_off_box(const struct bc_config *cfg, int cls,
                         int x, int y, int w, int h);

/* The same with the effect named, for a cover whose effect is not its class's
 * -- the eye bar's own. `ce` is a config effect index (BC_CFG_*). */
struct bc_box bc_off_box_fx(const struct bc_config *cfg, int ce, int cls,
                            int x, int y, int w, int h);

/*
 * Transparency survives where the format can hold it. `render` is the censored
 * frame, `flat` the source composited over white (what was detected), `rgba`
 * the source as decoded. Every pixel an effect touched -- inside a box, or
 * changed at all, which also catches tape's full-width bars and static's
 * feathered edge -- is taken from the render and made opaque, so no original
 * colour hides under a transparent cover. Everything else is the source pixel
 * exactly. Writes w*h*4 bytes to `out`.
 */
void bc_off_compose_alpha(const unsigned char *render, const unsigned char *flat,
                          const unsigned char *rgba, int w, int h,
                          const struct bc_box *boxes, int nb,
                          unsigned char *out);

/* ---- the eye bar ---------------------------------------------------------
 *
 * Shared by the image and video passes, on the geometry in eyebar.h that the
 * live censor uses too. A face gets a bar from the landmark model where it
 * gave a believable answer and from the band otherwise -- the band covers more
 * than the eyes, which is the fail-closed direction, so a model that is
 * missing or wrong costs tightness and never coverage.
 */
#include "eyebar.h"

struct bc_landmark;

struct bc_off_face { float x, y, w, h; };

enum { BC_OFF_EB_EYES = 1, BC_OFF_EB_BAND = 2 };

struct bc_off_eb {
	struct bc_landmark *lm;
	struct bc_preproc  *lpp;
	float              *lt;
	char                lm_path[512];   /* last tried, loaded or not */
	int                 lm_warned;
	/* What it did, for the run summary. */
	unsigned long       bars, bands, weak, rejected, eyes_dropped;
};

/*
 * A bar for each of `n` faces (image pixels), computed from the frame
 * currently uploaded to `g` -- the crops are cut from that texture. kind[i]
 * is BC_OFF_EB_EYES or BC_OFF_EB_BAND. The landmark model comes from the
 * config's landmark_model, or BC_LANDMARK_MODEL, and is loaded on first use
 * and reloaded if the name changes.
 */
void bc_off_eb_bars(struct bc_off_eb *eb, const struct bc_config *cfg,
                    struct bc_off_gl *g, const struct bc_off_face *f, int n,
                    struct bc_eyebar *bars, int *kind);
void bc_off_eb_free(struct bc_off_eb *eb);

/*
 * The cover for one bar, or 0 if it lies wholly off the image. `face` is the
 * face it belongs to; `grow` is the dilation factor still to apply. With the
 * face covered too (cfg->eye_bar_face) the bar scales about the face's centre,
 * as the live censor's fractions-of-the-face layout makes it; alone, about its
 * own. The bar's effect is eye_bar_effect when set, else its class's.
 */
int  bc_off_eb_box(const struct bc_config *cfg, int cls,
                   const struct bc_off_face *face, const struct bc_eyebar *bar,
                   float grow, int iw, int ih, struct bc_box *out);

/*
 * With the eye bar on, an EYE detection inside a head is the bar's business,
 * not a cover of its own: drawn as one it puts a window with the class's
 * effect over the eye. Compacts them out of d[] and returns the new count.
 */
int  bc_off_drop_eyes(const struct bc_config *cfg, struct bc_det *d, int n,
                      unsigned long *dropped);

/* The live path's class pairing: the BREAST variants are one physical thing. */
int  bc_off_classes_similar(int a, int b);

/* Single-quote `s` for a shell command line. -1 if it does not fit. */
int  bc_off_shq(char *dst, size_t cap, const char *s);

double bc_off_now_ms(void);

#endif /* BC_OFFLINE_H */
