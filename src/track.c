#include "track.h"

#include <math.h>
#include <string.h>

/*
 * Noise weights from the SORT/ByteTrack formulation. All scale with the box
 * height, because a 20 px error on a face filling the screen and the same error
 * on a distant one are not the same error. Aspect ratio gets a small fixed noise
 * instead -- it does not scale with size.
 *
 * The published constants (1/20 position, 1/160 velocity) are *per frame*, and
 * they cannot be used as-is here. This filter integrates in seconds, because our
 * frame interval varies -- inference runs between 68 and 138 Hz and the change
 * gate skips frames entirely -- so a per-frame step would mean different amounts
 * of time on different frames. Velocity in px/s is ~60x the same motion in
 * px/frame, so its noise must be ~60x too. Copying the constants across without
 * that conversion is a real and quiet failure: measured, the filter learned
 * 36 px/s of a true 600 px/s, lagged its content, dropped below the IoU
 * threshold and opened a fresh track every frame -- three tracks for one object.
 *
 * Converting: a per-step variance v becomes a rate v/dt_nominal, so a per-step
 * standard deviation s becomes s/sqrt(dt_nominal), with dt_nominal = 1/60 s
 * (sqrt(60) = 7.75). Position and measurement noise are in pixels and need no
 * conversion of units, only of rate; velocity needs both.
 */
#define STD_POS      (1.0f / 20.0f)   /* px, per px of height: measurement */
#define STD_POS_Q    0.39f            /* px/sqrt(s):   position process noise */
#define STD_VEL_Q    2.90f            /* px/s/sqrt(s): velocity process noise */
#define STD_VEL_0    3.75f            /* px/s:         initial velocity spread */

/* --- small dense linear algebra ------------------------------------------- */

/* Invert a 4x4 in place by Gauss-Jordan with partial pivoting. 0 on success. */
static int inv4(float m[4][4])
{
	float a[4][8];
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			a[i][j] = m[i][j];
			a[i][j + 4] = (i == j) ? 1.0f : 0.0f;
		}
	}

	for (int c = 0; c < 4; c++) {
		int piv = c;
		for (int r = c + 1; r < 4; r++)
			if (fabsf(a[r][c]) > fabsf(a[piv][c]))
				piv = r;
		if (fabsf(a[piv][c]) < 1e-12f)
			return -1;
		if (piv != c)
			for (int j = 0; j < 8; j++) {
				float t = a[c][j];
				a[c][j] = a[piv][j];
				a[piv][j] = t;
			}

		const float inv = 1.0f / a[c][c];
		for (int j = 0; j < 8; j++)
			a[c][j] *= inv;

		for (int r = 0; r < 4; r++) {
			if (r == c)
				continue;
			const float f = a[r][c];
			if (f == 0.0f)
				continue;
			for (int j = 0; j < 8; j++)
				a[r][j] -= f * a[c][j];
		}
	}

	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			m[i][j] = a[i][j + 4];
	return 0;
}

/* --- Kalman filter --------------------------------------------------------- */

/*
 * The measurement is the box itself, so H selects the first four state
 * components and is never formed: H*cov*H' is the top-left 4x4 block of cov,
 * and cov*H' is its first four columns. Writing those out directly is both
 * shorter and faster than a general matrix multiply.
 */
static void kf_init(struct bc_trk *t, const float z[4])
{
	memset(t->mean, 0, sizeof t->mean);
	memcpy(t->mean, z, 4 * sizeof *z);
	memset(t->cov, 0, sizeof t->cov);

	const float h = z[3];
	const float sd[8] = {
		2.0f * STD_POS * h, 2.0f * STD_POS * h, 1e-2f, 2.0f * STD_POS * h,
		STD_VEL_0 * h,      STD_VEL_0 * h,      1e-5f, STD_VEL_0 * h,
	};
	for (int i = 0; i < 8; i++)
		t->cov[i][i] = sd[i] * sd[i];
	memcpy(t->anchor, z, 4 * sizeof *z);
}

static void kf_predict(struct bc_trk *t, float dt)
{
	/* Constant velocity: position += velocity * dt. */
	for (int i = 0; i < 4; i++)
		t->mean[i] += t->mean[i + 4] * dt;

	/*
	 * cov = F cov F' + Q, with F = I except F[i][i+4] = dt. Expanding that
	 * is four rank-one updates rather than two 8x8 multiplies.
	 */
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 8; j++)
			t->cov[i][j] += dt * t->cov[i + 4][j];
	for (int j = 0; j < 4; j++)
		for (int i = 0; i < 8; i++)
			t->cov[i][j] += dt * t->cov[i][j + 4];

	const float h = fabsf(t->mean[3]);
	const float sp = STD_POS_Q * h, sv = STD_VEL_Q * h;
	/* Process noise grows with the gap, so a track predicted through a long
	 * absence becomes correspondingly less certain -- which is what lets a
	 * returning detection pull it back sharply instead of being resisted. */
	const float q[8] = { sp, sp, 1e-2f, sp, sv, sv, 1e-5f, sv };
	for (int i = 0; i < 8; i++)
		t->cov[i][i] += q[i] * q[i] * dt;
}

static void kf_update(struct bc_trk *t, const float z[4])
{
	const float h = fabsf(t->mean[3]);
	const float sp = STD_POS * h;
	const float r[4] = { sp, sp, 1e-1f, sp };

	float s[4][4];
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			s[i][j] = t->cov[i][j] + ((i == j) ? r[i] * r[i] : 0.0f);
	if (inv4(s) != 0)
		return;      /* singular: keep the prediction rather than diverge */

	/* K = cov[:, :4] * S^-1, an 8x4. */
	float k[8][4];
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 4; j++) {
			float acc = 0.0f;
			for (int m = 0; m < 4; m++)
				acc += t->cov[i][m] * s[m][j];
			k[i][j] = acc;
		}

	float y[4];
	for (int i = 0; i < 4; i++)
		y[i] = z[i] - t->mean[i];
	for (int i = 0; i < 8; i++) {
		float acc = 0.0f;
		for (int j = 0; j < 4; j++)
			acc += k[i][j] * y[j];
		t->mean[i] += acc;
	}

	/* cov -= K * cov[:4, :] */
	float upd[8][8];
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++) {
			float acc = 0.0f;
			for (int m = 0; m < 4; m++)
				acc += k[i][m] * t->cov[m][j];
			upd[i][j] = acc;
		}
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			t->cov[i][j] -= upd[i][j];
}

/* --- boxes ----------------------------------------------------------------- */

static void box_to_z(const struct bc_det *d, float z[4])
{
	z[0] = d->x + d->w * 0.5f;
	z[1] = d->y + d->h * 0.5f;
	z[2] = d->h > 0.0f ? d->w / d->h : 1.0f;
	z[3] = d->h;
}

#define PRED_MAX_FRAC 0.5f   /* how far a prediction may leave its measurement */

int bc_trk_box(const struct bc_trk *t, float *x, float *y, float *w, float *h)
{
	if (t->state != BC_TRK_CONFIRMED && t->state != BC_TRK_LOST)
		return 0;
	float th = t->mean[3];
	float tw = t->mean[2] * th;

	/*
	 * Clamp the prediction to half the box's own size away from the last
	 * real measurement. Unclamped, a track that goes lost while moving is
	 * carried at its last velocity for the whole memory window and can end
	 * up anywhere -- measured at 1451 px of box movement in one frame, which
	 * is a bar sliding across the screen over content it was never on. The
	 * cap is proportional because the growth is: the same reasoning capped
	 * the linear extrapolation this replaces, after 150 px eye boxes doubled
	 * and the two bars met in the middle of the face.
	 */
	float cx = t->mean[0], cy = t->mean[1];
	const float ah = t->anchor[3] > 0.0f ? t->anchor[3] : th;
	const float aw = t->anchor[2] * ah;
	const float mx = aw * PRED_MAX_FRAC, my = ah * PRED_MAX_FRAC;
	if (cx > t->anchor[0] + mx) cx = t->anchor[0] + mx;
	if (cx < t->anchor[0] - mx) cx = t->anchor[0] - mx;
	if (cy > t->anchor[1] + my) cy = t->anchor[1] + my;
	if (cy < t->anchor[1] - my) cy = t->anchor[1] - my;

	/*
	 * The same cap on the size. The centre was clamped above because an
	 * unclamped prediction slides the box off its content; the size needs
	 * it for the same reason and one worse. Height and aspect are filter
	 * states with velocities of their own, so a track that goes lost while
	 * its box was growing keeps growing -- observed as a bar swelling to
	 * fill the monitor and snapping back when the track died -- and a
	 * shrinking one runs through zero and comes out negative, which is not
	 * a small box but an inside-out one.
	 *
	 * Growth is allowed a little room because growing covers more; shrink
	 * is floored well above zero because a box that has lost its size has
	 * not lost its subject.
	 */
	if (t->anchor[3] > 0.0f) {
		const float lo = ah * 0.5f, hi = ah * 2.0f;
		if (th < lo) th = lo;
		if (th > hi) th = hi;
		const float alo = aw * 0.5f, ahi = aw * 2.0f;
		if (tw < alo) tw = alo;
		if (tw > ahi) tw = ahi;
	}
	if (!(tw > 1.0f)) tw = 1.0f;   /* also catches NaN */
	if (!(th > 1.0f)) th = 1.0f;

	*w = tw;
	*h = th;
	*x = cx - tw * 0.5f;
	*y = cy - th * 0.5f;
	return 1;
}


static float trk_iou(const struct bc_trk *t, const struct bc_det *d)
{
	float ax, ay, aw, ah;
	const float th = t->mean[3], tw = t->mean[2] * th;
	ax = t->mean[0] - tw * 0.5f;
	ay = t->mean[1] - th * 0.5f;
	aw = tw;
	ah = th;

	const float x1 = ax > d->x ? ax : d->x;
	const float y1 = ay > d->y ? ay : d->y;
	const float x2 = (ax + aw) < (d->x + d->w) ? (ax + aw) : (d->x + d->w);
	const float y2 = (ay + ah) < (d->y + d->h) ? (ay + ah) : (d->y + d->h);
	const float iw = x2 - x1, ih = y2 - y1;
	if (iw <= 0.0f || ih <= 0.0f)
		return 0.0f;
	const float inter = iw * ih;
	const float uni = aw * ah + d->w * d->h - inter;
	return uni > 0.0f ? inter / uni : 0.0f;
}

/* --- association ----------------------------------------------------------- */

void bc_tracker_init(struct bc_tracker *tr, float conf, float iou,
                     double memory_ms)
{
	memset(tr, 0, sizeof *tr);
	tr->high_score   = conf;
	/*
	 * The low band is what score hysteresis used to be: detections below the
	 * censoring threshold are still evidence about a track that already
	 * exists, just not enough to start one.
	 */
	/* Default only; bc_main sets this from its own HYST_RATIO, which is the
	 * same threshold the detector is actually run at. */
	tr->low_score    = conf * 0.5f;
	tr->iou_thresh   = iou;
	tr->memory_ms    = memory_ms;
	tr->confirm_hits = 1;   /* §2: cover on first sight, do not wait */
	/* -1 = follow high_score. memset above already zeroed it, and zero is a
	 * real threshold meaning "admit everything", so say it explicitly. */
	for (int i = 0; i < BC_MAX_CLASSES; i++)
		tr->class_score[i] = -1.0f;
}

/*
 * The score a detection of `cls` must reach to open a track, and the weak band
 * beneath it, which keeps the global hysteresis ratio.
 *
 * Both come out of one place because they are one decision. Reading the
 * override into `hi` while leaving `lo` on the global low_score is the bug this
 * is shaped to prevent: with an override below the global floor, the weak band
 * would sit *above* the opening score, and every detection the override was
 * added to admit would be classed as too weak to open anything.
 */
static void class_band(const struct bc_tracker *tr, int cls,
                       float *hi, float *lo)
{
	const float over = (cls >= 0 && cls < BC_MAX_CLASSES)
	                 ? tr->class_score[cls] : -1.0f;
	if (!(over > 0.0f)) {
		*hi = tr->high_score;
		*lo = tr->low_score;
		return;
	}
	const float ratio = tr->high_score > 0.0f
	                  ? tr->low_score / tr->high_score : 0.5f;
	*hi = over;
	*lo = over * ratio;
}

/*
 * Greedy one-to-one association by descending IoU.
 *
 * ByteTrack uses the Hungarian algorithm for a globally optimal assignment.
 * Greedy differs from it only when one detection is a better match for a track
 * that some other detection also wants, which needs three boxes mutually
 * overlapping above threshold; with at most a few dozen boxes on a screen that
 * is rare, and the cost of getting it wrong is one frame of a slightly
 * mismatched box rather than a missed cover. What matters here is that it is
 * one-to-one at all: the matcher this replaces let two detections claim the
 * same track in one frame, and the later one silently overwrote the earlier.
 */
static void associate(struct bc_tracker *tr, const struct bc_det *dets,
                      const int *idx, int n, int *trk_used, int *det_used,
                      double now, unsigned long *counter, int *det_track)
{
	for (;;) {
		float best = tr->iou_thresh;
		int bt = -1, bd = -1;

		for (int a = 0; a < BC_TRK_MAX; a++) {
			if (tr->t[a].state == BC_TRK_FREE || trk_used[a])
				continue;
			for (int b = 0; b < n; b++) {
				if (det_used[b])
					continue;
				const struct bc_det *d = &dets[idx[b]];
				if (tr->same_class ? !tr->same_class(tr->t[a].cls, d->cls)
				                   : d->cls != tr->t[a].cls)
					continue;
				const float v = trk_iou(&tr->t[a], d);
				if (v > best) {
					best = v;
					bt = a;
					bd = b;
				}
			}
		}
		if (bt < 0)
			break;

		const struct bc_det *d = &dets[idx[bd]];
		float z[4];
		box_to_z(d, z);
		kf_update(&tr->t[bt], z);
		memcpy(tr->t[bt].anchor, z, sizeof tr->t[bt].anchor);
		tr->t[bt].last_update   = now;
		tr->t[bt].last_measured = now;
		tr->t[bt].hits++;
		if (d->score > tr->t[bt].score)
			tr->t[bt].score = d->score;
		if (tr->t[bt].state != BC_TRK_CONFIRMED &&
		    tr->t[bt].hits >= (unsigned long)tr->confirm_hits) {
			tr->t[bt].state = BC_TRK_CONFIRMED;
			tr->confirmed++;
		}
		/* Measured again: it is no longer a track we cannot see. */
		tr->t[bt].frozen = 0;
		trk_used[bt] = 1;
		det_used[bd] = 1;
		if (det_track)
			det_track[idx[bd]] = bt;
		(*counter)++;
	}
}

void bc_tracker_hold(struct bc_tracker *tr, double now)
{
	bc_tracker_hold_bounded(tr, now, 0.0);
}

int bc_trk_hold(struct bc_trk *t, double now, double ceiling_ms)
{
	if (!t || t->state == BC_TRK_FREE)
		return 0;
	/*
	 * The ceiling reads last_measured, never last_update: a hold moves
	 * last_update forward, so a ceiling reading that would ask "was this
	 * held recently", to which the answer right after a hold is always yes,
	 * and it could never fire. That distinction is the whole mechanism --
	 * see the note above hold_or_cap() in bc_main.c, where getting it wrong
	 * held every track forever on a still screen.
	 */
	if (ceiling_ms > 0.0 && now - t->last_measured > ceiling_ms)
		return 1;
	t->last_update  = now;
	t->last_predict = now;
	return 0;
}

int bc_tracker_hold_bounded(struct bc_tracker *tr, double now, double ceiling_ms)
{
	int capped = 0;
	for (int i = 0; i < BC_TRK_MAX; i++) {
		if (tr->t[i].state == BC_TRK_FREE)
			continue;
		/*
		 * Per track, not all or nothing. One track sitting past its
		 * ceiling says nothing about the others -- they may have been
		 * measured on this very frame -- so letting it cancel their
		 * hold ages every box on screen for one box's staleness, which
		 * is turnover the picture did not ask for.
		 */
		capped += bc_trk_hold(&tr->t[i], now, ceiling_ms);
	}
	return capped;
}

/*
 * The picture was replaced wholesale -- a scroll, a page turn, a cut.
 *
 * Both halves of the filter's model just became false, and they fail in
 * different ways. The velocity is the loud one: it was measured on content
 * that is no longer there, so an unmeasured track keeps travelling on it and
 * the box flies off across the screen. The position is the quiet one: a track
 * that has not been measured since the cut describes where something *was*,
 * and after a scroll that place holds different content.
 *
 * Left alone, both compound. Scrolling a gallery opened 186 tracks in 30 s and
 * expired 122, and 15 boxes per presented frame were being drawn from
 * prediction rather than measurement -- stale boxes stacking up until the
 * whole screen was covered, then sliding away as their velocities carried
 * them off. That is not caution, it is noise: the detector re-runs on the new
 * picture and the content that is actually there gets its own fresh box, so
 * the ghosts add no coverage of anything real. Dropping them costs nothing
 * that was protecting anything.
 *
 * Confirmed tracks are kept. They were measured on the picture as it is now,
 * which is the whole distinction being drawn here.
 */
void bc_tracker_scene_cut(struct bc_tracker *tr)
{
	for (int i = 0; i < BC_TRK_MAX; i++) {
		if (tr->t[i].state == BC_TRK_FREE)
			continue;
		if (tr->t[i].state == BC_TRK_LOST) {
			tr->t[i].state = BC_TRK_FREE;
			continue;
		}
		for (int v = 4; v < 8; v++)
			tr->t[i].mean[v] = 0.0f;
		tr->t[i].anchor[0] = tr->t[i].mean[0];
		tr->t[i].anchor[1] = tr->t[i].mean[1];
		tr->t[i].anchor[2] = tr->t[i].mean[2];
		tr->t[i].anchor[3] = tr->t[i].mean[3];
	}
}

int bc_tracker_update(struct bc_tracker *tr, const struct bc_det *dets, int n,
                      double now, const int *censor, int *det_track)
{
	int trk_used[BC_TRK_MAX] = { 0 };

	for (int i = 0; i < n; i++)
		if (det_track)
			det_track[i] = -1;

	/* 1. Predict every live track forward to now. */
	for (int i = 0; i < BC_TRK_MAX; i++) {
		if (tr->t[i].state == BC_TRK_FREE)
			continue;
		/*
		 * A frozen track is one nothing can currently see. Advancing
		 * its clock without integrating its velocity is the whole
		 * point: prediction here would grow the box every frame with
		 * nothing to correct it.
		 */
		if (tr->t[i].frozen) {
			tr->t[i].last_predict = now;
			continue;
		}
		double dt = now - tr->t[i].last_predict;
		if (dt < 0.0)
			dt = 0.0;
		kf_predict(&tr->t[i], (float)(dt / 1000.0));
		tr->t[i].last_predict = now;
	}

	/* 2. Split the detections by score. */
	int hi[BC_MAX_DET], lo[BC_MAX_DET], nhi = 0, nlo = 0;
	for (int i = 0; i < n; i++) {
		if (dets[i].cls < 0 || dets[i].cls >= BC_MAX_CLASSES)
			continue;
		float hi_s, lo_s;
		class_band(tr, dets[i].cls, &hi_s, &lo_s);
		if (dets[i].score >= hi_s)
			hi[nhi++] = i;
		else if (dets[i].score >= lo_s)
			lo[nlo++] = i;
	}

	/*
	 * 3. Confident detections first, then the weak ones against whatever is
	 * still unmatched. Doing it in one pass would let a weak detection win a
	 * track from the confident detection that belongs to it.
	 */
	int hi_used[BC_MAX_DET] = { 0 }, lo_used[BC_MAX_DET] = { 0 };
	associate(tr, dets, hi, nhi, trk_used, hi_used, now, &tr->matched_high,
	          det_track);
	associate(tr, dets, lo, nlo, trk_used, lo_used, now, &tr->matched_low,
	          det_track);

	/* 4. Confident detections that matched nothing open a track -- but only
	 *    for a class we are actually censoring. */
	for (int b = 0; b < nhi; b++) {
		if (hi_used[b])
			continue;
		const struct bc_det *d = &dets[hi[b]];
		if (censor && !censor[d->cls])
			continue;
		for (int i = 0; i < BC_TRK_MAX; i++) {
			if (tr->t[i].state != BC_TRK_FREE)
				continue;
			float z[4];
			box_to_z(d, z);
			kf_init(&tr->t[i], z);
			tr->t[i].cls = d->cls;
			tr->t[i].score = d->score;
			tr->t[i].hits = 1;
			tr->t[i].last_update   = now;
			tr->t[i].last_measured = now;
			tr->t[i].last_predict  = now;
			tr->t[i].born          = now;
			tr->t[i].state = (tr->confirm_hits <= 1)
			                 ? BC_TRK_CONFIRMED : BC_TRK_TENTATIVE;
			tr->created++;
			if (det_track)
				det_track[hi[b]] = i;
			break;
		}
	}

	/*
	 * 5. Tracks nothing matched are not deleted -- they are predicted, which
	 * is the whole point. A box that the detector blinked on keeps moving
	 * with its content instead of freezing where it was or vanishing.
	 * memory_ms bounds that: past it, we are guessing about content we have
	 * no evidence for, and §2's bias does not extend to inventing coverage.
	 */
	int live = 0;
	for (int i = 0; i < BC_TRK_MAX; i++) {
		if (tr->t[i].state == BC_TRK_FREE)
			continue;
		if (!trk_used[i]) {
			if (tr->t[i].state == BC_TRK_CONFIRMED) {
				tr->t[i].state = BC_TRK_LOST;
				tr->lost++;
			} else if (tr->t[i].state == BC_TRK_TENTATIVE) {
				/* Never confirmed and already gone: it was
				 * probably never there. */
				tr->t[i].state = BC_TRK_FREE;
				tr->removed++;
				continue;
			}
			if (now - tr->t[i].last_update > tr->memory_ms) {
				tr->t[i].state = BC_TRK_FREE;
				tr->removed++;
				continue;
			}
		}
		live++;
	}
	return live;
}

void bc_trk_freeze(struct bc_trk *t, double now)
{
	if (!t || t->state == BC_TRK_FREE)
		return;
	t->frozen       = 1;
	t->last_update  = now;
	t->last_predict = now;
}
