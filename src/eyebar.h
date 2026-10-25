#ifndef BC_EYEBAR_H
#define BC_EYEBAR_H

/*
 * The eye bar's geometry: where a bar goes, how big, and at what angle, given
 * what the landmark model says about a face. Pure arithmetic -- no GL, no
 * models -- so the live censor and the offline passes decide it identically,
 * and so it can be tested without either.
 *
 * It lives here, and not inside bc_main.c, because the offline passes need the
 * same rules. A second copy of "which landmark answers are believable" is how
 * an image written by bc-image ends up barred differently from the screen it
 * was meant to match.
 */

/* Tunables. The values are the live censor's, measured there. */
#define BC_EB_KPT_BAR_WIDE  1.90f  /* bar length, x eye separation: past the outer corners */
#define BC_EB_KPT_BAR_TALL  0.60f  /* bar height, x eye separation */
#define BC_EB_KPT_SEP_MIN   0.20f  /* eye separation, as a fraction of the face box */
#define BC_EB_KPT_SEP_MAX   0.90f
#define BC_EB_KPT_IN_FACE   0.15f  /* how far outside its face an eye may still sit */
#define BC_EB_LM_PAD        1.25f  /* face box padded by this before the landmark crop */
#define BC_EB_LM_MIN_SCORE  0.35f  /* mean heatmap peak below which the answer is ignored */
#define BC_EB_LM_BAR_PAD    1.45f  /* measured eye extents grown by this, for lids and lashes */
#define BC_EB_INSET         0.04f  /* the band is trimmed this much each side */
#define BC_EB_BAND_TOP      0.22f  /* the band's top, as a fraction of the face height */
#define BC_EB_BAND_BOT      0.78f

/*
 * Does this class name a head -- one an eye bar can be built on? Matching "FACE"
 * stopped being enough once models arrived that call it HEAD: booru_yolo names
 * human and anthro heads separately, and an anthro head is a face for this
 * purpose. The others are matched exactly rather than by prefix: HIP would
 * otherwise read as a head class.
 */
int bc_is_head_class(const char *name);

/* A bar is symmetric under a half turn, so every angle has a representative
 * in (-pi/2, pi/2]. Keeping them there means easing never takes the long way
 * round a wrap that is not a real rotation. */
float bc_eyebar_wrap(float a);

struct bc_eyebar {
	float cx, cy;      /* centre, image pixels, y down */
	float w, h;        /* size along and across the bar */
	float angle;       /* radians, counter-clockwise as GL sees it */
};

/*
 * A bar from two eye centres (image pixels, y down) and, when the landmark
 * model measured them, the half-extents of the eye points along and across
 * the eye line (0, 0 when it did not).
 *
 * Returns
 *    1  a bar, in *out;
 *   -1  the eyes are not believable for this face -- the caller falls back to
 *       the band, and may count it;
 *    0  the eyes coincide, so there is no direction to give a bar. Same
 *       fallback, but not worth counting as the model being wrong.
 *
 * Confidence is the model's opinion of its own output and cannot catch it
 * being confidently wrong, which is what a face model does on a turned-away
 * head: both "eyes" land on one spot. So the geometry is checked as well: the
 * eyes belong inside their own face box, separated by a sane fraction of it.
 *
 * `pad` grows the measured extents; `nofloor` lets them shrink the bar below
 * the separation-scaled size, which is otherwise a floor -- §2 prefers
 * covering too much, and a model that clustered its points tightly must not
 * shrink the cover below what the old heuristic gave.
 */
int bc_eyebar_from_eyes(float ex0, float ey0, float ex1, float ey1,
                        float ext_along, float ext_across,
                        float pad, int nofloor,
                        float fx, float fy, float fw, float fh,
                        struct bc_eyebar *out);

/* The band: an upright strip across the face where eyes usually are, for when
 * no eyes could be found. Inside the face by construction. */
void bc_eyebar_band(float fx, float fy, float fw, float fh,
                    float top, float bot, struct bc_eyebar *out);

#endif /* BC_EYEBAR_H */
