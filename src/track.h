#ifndef BC_TRACK_H
#define BC_TRACK_H

#include "detect.h"

/*
 * ByteTrack (Zhang et al. 2022) with a Kalman filter, in place of the IoU
 * matcher in bc_main.c and the cluster of patches around it.
 *
 * What is being replaced, and why it needed replacing:
 *
 *   sticky_ms      a box outlives its last detection by a fixed time
 *   hold           a box survives while the pixels under it do not change
 *   hysteresis     the detector runs below threshold so weak detections can
 *                  sustain (but never create) a track
 *   prediction     a linear extrapolation that only ever grows a box
 *
 * Each of those exists because the detector's score for content it is already
 * looking at oscillates, so a single threshold deleted the box on every dip.
 * They work, but they are four independent knobs approximating one thing:
 * "how long should I believe in something I can no longer clearly see?"
 *
 * ByteTrack answers that directly. A Kalman filter carries each track's motion,
 * so a gap is *predicted* rather than frozen or linearly extrapolated, and the
 * association runs in two passes -- confident detections first, then the weak
 * ones against whatever tracks are still unmatched. That second pass is what
 * our score hysteresis was crudely imitating: a weak detection may sustain a
 * track but never open one.
 *
 * It also fixes a defect the old matcher had outright. That one took the *first*
 * track whose IoU cleared the threshold, in array order, and nothing stopped two
 * detections in one frame from claiming the same track -- the last one processed
 * won, and "last" depends on score order, which flips between inferences. The
 * association here is one-to-one by construction.
 *
 * Parameter naming follows Pure Vision's, which the teardown recovered
 * (PUREVISION.md): two score thresholds, one IoU threshold, and a memory
 * expressed as a duration rather than a frame count -- correct here, where the
 * frame rate varies and a frame count would mean different things at 60 and 180
 * Hz.
 */

#define BC_TRK_MAX 64

enum bc_trk_state {
	BC_TRK_FREE = 0,
	BC_TRK_TENTATIVE,   /* seen once; not yet drawn */
	BC_TRK_CONFIRMED,   /* drawn */
	BC_TRK_LOST,        /* predicted through a gap, still drawn */
};

struct bc_trk {
	/*
	 * Kalman state: centre, aspect ratio, height, and their velocities.
	 * Tracking aspect and height rather than width and height is the SORT
	 * formulation ByteTrack inherits -- an object's aspect is far steadier
	 * than its width, so noise in the box lands where it does least harm.
	 */
	float mean[8];      /* cx, cy, a, h, vcx, vcy, va, vh */
	float cov[8][8];

	int    cls;
	float  score;
	int    state;
	double last_update;   /* ms; when a detection last matched */

	/*
	 * When a detection last actually matched -- as opposed to last_update,
	 * which the change gate's hold also moves forward. Keeping them apart
	 * is what stops a still screen carrying a track forever: the hold may
	 * postpone expiry, but the ceiling that bounds the hold has to read a
	 * clock the hold does not touch, or it can never fire.
	 */
	double last_measured;
	/*
	 * When the filter was last advanced. Distinct from last_update, and it
	 * has to be: prediction steps forward from where the state already is,
	 * so timing it from the last *match* re-applies the whole gap on every
	 * frame it stays unmatched. A track then accelerated away from its
	 * content, lost the overlap, and the next detection opened a fresh track
	 * instead of matching -- one new track per frame.
	 */
	double last_predict;
	/*
	 * The last box actually measured, and its centre. Prediction is clamped
	 * against this: past a fraction of the box's own size the prediction is
	 * a bigger claim than the measurement it came from, and one bad velocity
	 * sample sends the box across the screen.
	 */
	float  anchor[4];
	/*
	 * Frozen: held with no evidence, so not predicted either.
	 *
	 * A hold advances the clocks; prediction then keeps integrating the
	 * filter's velocity -- including vh, the growth rate of the box -- and
	 * with nothing ever re-measuring it, that integral has no bound. Covers
	 * grew in every direction until they swallowed the screen. A track we
	 * have admitted we cannot see must not also be guessed about: freezing
	 * keeps the box exactly where the last real measurement put it.
	 */
	int    frozen;
	unsigned long hits;
	/*
	 * When this track opened. Distinct from every other clock here, all of
	 * which move forward: growth is a function of how long a thing has been
	 * on screen, so it needs the one that does not.
	 */
	double born;
};

struct bc_tracker {
	struct bc_trk t[BC_TRK_MAX];
	float  high_score;      /* open a track at or above this */
	float  low_score;       /* may only sustain, never open */
	/*
	 * Per-class override for high_score, or -1 to follow it.
	 *
	 * The weak band moves with it: what the two thresholds encode is one
	 * hysteresis ratio, not two independent numbers, so a class given its
	 * own opening score keeps the same proportional band beneath it rather
	 * than inheriting the global one -- which, under a lowered floor, would
	 * sit above the very detections the override exists to admit.
	 */
	float  class_score[BC_MAX_CLASSES];
	float  iou_thresh;
	double memory_ms;       /* how long a lost track is predicted for */
	int    confirm_hits;    /* hits before a track is drawn */
	/*
	 * May a detection of class `b` match a track of class `a`? NULL means
	 * exact equality. bc_main passes its similar-class pairing through here:
	 * FEMALE_BREAST_EXPOSED and MALE_BREAST_EXPOSED are one physical thing
	 * the model flips between, and without the pairing a flipped frame
	 * abandons the track and opens another.
	 */
	int  (*same_class)(int a, int b);

	/* Counters, reported at exit like every other metric here. */
	unsigned long created, confirmed, lost, removed;
	unsigned long matched_high, matched_low, double_claim_avoided;
	unsigned long revived;   /* detections that took over a nearby lost track */
	int    revive;           /* the revival pass is on; BC_NO_REVIVE turns it off */
};

/*
 * How far from a lost track (in its own size) and how different in area a
 * detection may be and still take it over. See the revival pass in
 * bc_tracker_update().
 */
#define BC_REVIVE_DIST 1.0f
#define BC_REVIVE_AREA 2.5f

void bc_tracker_init(struct bc_tracker *tr, float conf, float iou,
                     double memory_ms);

/*
 * Advance every track to `now` and associate `dets` against them.
 * Detections are in screen pixels. Returns the number of live tracks.
 *
 * `det_track`, if given, receives the track index each detection landed on, or
 * -1 for one that opened nothing. The caller needs it because a track carries
 * state this module knows nothing about -- an eye bar's angle and the face it
 * is clamped inside -- which has to follow the detection to its track.
 */
int  bc_tracker_update(struct bc_tracker *tr, const struct bc_det *dets,
                       int n, double now, const int *censor, int *det_track);

/*
 * Hold every live track at `now` without associating anything.
 *
 * For frames the change gate skipped. Not the same as an update with no
 * detections: "nothing was inferred" must not read as "nothing is there", or a
 * still screen would expire every track and uncover itself. This only moves the
 * clocks forward, so the gap is never charged against a track's memory and the
 * next real update predicts from here rather than from before the skip.
 */
void bc_tracker_hold(struct bc_tracker *tr, double now);

/*
 * The same, but a track that has gone `ceiling_ms` without a real measurement
 * is left to age instead of being held. Returns how many were left. Zero
 * means hold everything, which is what bc_tracker_hold passes.
 */
int bc_tracker_hold_bounded(struct bc_tracker *tr, double now,
                            double ceiling_ms);

/*
 * Hold one track, on evidence that applies to it alone.
 *
 * The occlusion hold (src/hold.h) answers per box, not per frame: the picture
 * under *this* cover did not change, so nothing was learned about *this*
 * track, whatever the rest of the screen did. Same ceiling semantics as
 * bc_tracker_hold_bounded -- returns 1 when the track is past the ceiling and
 * was left to age instead of held.
 */
int bc_trk_hold(struct bc_trk *t, double now, double ceiling_ms);

/*
 * Hold a track *and* stop predicting it. Cleared automatically the moment a
 * detection matches it again. See the `frozen` field.
 */
void bc_trk_freeze(struct bc_trk *t, double now);

/*
 * The picture changed wholesale. Drops tracks that have not been measured
 * since (their position describes content that has scrolled away) and zeroes
 * the velocity of the ones that have (it was measured on the old picture).
 */
void bc_tracker_scene_cut(struct bc_tracker *tr);

/* Read a track's current box, in screen pixels. Returns 0 if not drawable. */
int  bc_trk_box(const struct bc_trk *t, float *x, float *y, float *w, float *h);

#endif /* BC_TRACK_H */
