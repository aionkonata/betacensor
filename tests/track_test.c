/*
 * ByteTrack behaviours that the tracker it replaces got wrong.
 *
 * Three claims, each pinned to the specific defect that motivated it. None of
 * these need the screen: they are properties of the association, so driving it
 * directly is both decisive and repeatable, which measuring through content is
 * not.
 */
#include "../src/track.c"

#include <stdio.h>

static int failures;

static void check(int ok, const char *what)
{
	printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
	if (!ok)
		failures++;
}

static struct bc_det det(float x, float y, float w, float h, float s)
{
	struct bc_det d = { 0 };
	d.x = x; d.y = y; d.w = w; d.h = h;
	d.score = s; d.cls = 0;
	return d;
}

static int live_tracks(const struct bc_tracker *tr)
{
	int n = 0;
	for (int i = 0; i < BC_TRK_MAX; i++)
		if (tr->t[i].state == BC_TRK_CONFIRMED ||
		    tr->t[i].state == BC_TRK_LOST)
			n++;
	return n;
}

/*
 * The old matcher took the first track whose IoU cleared the threshold and had
 * no notion of a track already being claimed this frame, so two detections
 * could both land on one track and the later silently overwrote the earlier.
 * One object's box then jumped to the other's geometry, and which one won
 * depended on score order, which flips between inferences.
 */
static void test_one_to_one(const int *censor)
{
	struct bc_tracker tr;
	bc_tracker_init(&tr, 0.35f, 0.30f, 500.0);

	struct bc_det a = det(100, 100, 100, 200, 0.80f);
	bc_tracker_update(&tr, &a, 1, 0.0, censor, NULL);
	check(live_tracks(&tr) == 1, "one detection opens one track");

	/* Both of these overlap the track above threshold (IoU 1.00 and 0.43). */
	struct bc_det two[2] = { det(100, 100, 100, 200, 0.80f),
	                         det(140, 100, 100, 200, 0.75f) };
	bc_tracker_update(&tr, two, 2, 16.0, censor, NULL);
	check(live_tracks(&tr) == 2,
	      "two overlapping detections cannot share one track");
	check(tr.matched_high == 1 && tr.created == 2,
	      "  one matched, the other opened its own track");
}

/*
 * A track the detector blinked on used to freeze where it was, or be grown by a
 * linear extrapolation that only ever expanded it. A Kalman filter carries the
 * motion instead, so the box keeps moving with its content through the gap.
 */
static void test_predicts_through_a_gap(const int *censor)
{
	struct bc_tracker tr;
	bc_tracker_init(&tr, 0.35f, 0.30f, 500.0);

	const float vx = 600.0f;          /* px/s, i.e. 10 px per 60 Hz frame */
	const double step = 1000.0 / 60.0;
	double t = 0.0;
	float x = 100.0f;

	for (int f = 0; f < 30; f++, t += step, x += vx * (float)(step / 1000.0)) {
		struct bc_det d = det(x, 100, 100, 200, 0.80f);
		bc_tracker_update(&tr, &d, 1, t, censor, NULL);
	}

	/* Five frames with no detection at all. */
	float bx = 0, by = 0, bw = 0, bh = 0;
	for (int f = 0; f < 5; f++, t += step, x += vx * (float)(step / 1000.0))
		bc_tracker_update(&tr, NULL, 0, t, censor, NULL);

	int drawn = 0;
	for (int i = 0; i < BC_TRK_MAX; i++)
		if (bc_trk_box(&tr.t[i], &bx, &by, &bw, &bh))
			drawn++;
	check(drawn == 1, "a track with no detection is still drawn");

	/*
	 * Where the object actually is by now. Allow a quarter of a box width:
	 * the filter is meant to follow the content, not to be exact.
	 */
	const float err = fabsf(bx - x);
	printf("      predicted x=%.1f, true x=%.1f, error %.1f px\n", bx, x, err);
	check(err < 25.0f, "  and predicted forward, not frozen in place");

	/* Frozen would mean the error equals the whole distance travelled. */
	check(err < vx * (float)(5 * step / 1000.0) * 0.5f,
	      "  error is well under the distance it would have lagged");
}

/*
 * Score hysteresis by another name: a weak detection is evidence about
 * something already being tracked, but not enough to start covering a new part
 * of the screen.
 */
static void test_low_scores_sustain_but_never_create(const int *censor)
{
	struct bc_tracker tr;
	bc_tracker_init(&tr, 0.35f, 0.30f, 500.0);

	struct bc_det weak = det(100, 100, 100, 200, 0.20f);
	bc_tracker_update(&tr, &weak, 1, 0.0, censor, NULL);
	check(live_tracks(&tr) == 0, "a weak detection alone opens no track");

	struct bc_det strong = det(100, 100, 100, 200, 0.80f);
	bc_tracker_update(&tr, &strong, 1, 16.0, censor, NULL);
	check(live_tracks(&tr) == 1, "a confident one does");

	bc_tracker_update(&tr, &weak, 1, 32.0, censor, NULL);
	check(tr.matched_low == 1, "and a weak one then sustains it");
	check(live_tracks(&tr) == 1, "  keeping it covered through the dip");
}

/*
 * Per-class opening scores.
 *
 * One global floor is a single compromise across classes whose cost of being
 * missed differs wildly. The property under test is that an override moves
 * *only* its own class, and that the weak band moves with it -- the bug this
 * is shaped around is an override read into the opening score while the low
 * band stays global, which for an override below the global floor puts the
 * weak band above the opening score and rejects the very detections the
 * override was added to admit.
 */
static void test_per_class_scores(const int *censor)
{
	struct bc_tracker tr;
	bc_tracker_init(&tr, 0.50f, 0.30f, 500.0);
	tr.low_score = 0.25f;             /* the global 0.5 hysteresis ratio */

	/* Class 0 is pinned well below the global floor; class 1 is not. */
	tr.class_score[0] = 0.20f;

	struct bc_det d0 = det(100, 100, 100, 200, 0.30f);
	bc_tracker_update(&tr, &d0, 1, 0.0, censor, NULL);
	check(live_tracks(&tr) == 1,
	      "a detection under the global floor opens a track on a pinned class");

	struct bc_tracker tr2;
	bc_tracker_init(&tr2, 0.50f, 0.30f, 500.0);
	tr2.low_score = 0.25f;
	tr2.class_score[0] = 0.20f;

	struct bc_det d1 = det(100, 100, 100, 200, 0.30f);
	d1.cls = 1;
	bc_tracker_update(&tr2, &d1, 1, 0.0, censor, NULL);
	check(live_tracks(&tr2) == 0,
	      "  while the same score on an unpinned class still opens none");

	/*
	 * The band beneath the override. At 0.20 opening and the global 0.5
	 * ratio the weak band starts at 0.10, so 0.12 must sustain rather than
	 * be discarded -- under a global low_score of 0.25 it would be neither
	 * high enough to open nor low enough to count as weak, and the track
	 * would go unsustained through the dip.
	 */
	struct bc_det weak = det(100, 100, 100, 200, 0.12f);
	bc_tracker_update(&tr, &weak, 1, 16.0, censor, NULL);
	check(tr.matched_low == 1,
	      "the weak band follows the override down");
	check(live_tracks(&tr) == 1,
	      "  keeping a pinned class covered through the dip");

	/* An override at or below zero means "no override". */
	struct bc_tracker tr3;
	bc_tracker_init(&tr3, 0.50f, 0.30f, 500.0);
	tr3.low_score = 0.25f;
	struct bc_det mid = det(100, 100, 100, 200, 0.30f);
	bc_tracker_update(&tr3, &mid, 1, 0.0, censor, NULL);
	check(live_tracks(&tr3) == 0,
	      "an unset override leaves the class on the global floor");
}

/* A class that is switched off must never open a track. */
static void test_respects_the_class_policy(void)
{
	int off[BC_MAX_CLASSES] = { 0 };
	struct bc_tracker tr;
	bc_tracker_init(&tr, 0.35f, 0.30f, 500.0);

	struct bc_det d = det(100, 100, 100, 200, 0.90f);
	bc_tracker_update(&tr, &d, 1, 0.0, off, NULL);
	check(live_tracks(&tr) == 0, "an uncensored class opens no track");
}

/*
 * A track whose box was growing when it lost its detection keeps growing: the
 * filter carries a velocity on the height and on the aspect ratio, and nothing
 * downstream of it disagreed. Fed a shrinking one instead, the same velocity
 * takes the height through zero and out the other side, and a negative width
 * or height is not a small rectangle -- it is an inverted one, which is what
 * glScissor answers with GL_INVALID_VALUE.
 *
 * Both directions are driven here rather than one, because the reported
 * symptoms were both: a bar swelling to fill the monitor, and a GL error that
 * killed the capture.
 */
static void test_coasting_size_stays_sane(const int *censor)
{
	for (int shrinking = 0; shrinking < 2; shrinking++) {
		struct bc_tracker tr;
		bc_tracker_init(&tr, 0.35f, 0.30f, 100000.0);

		/* Establish a track that is changing size fast. */
		double t = 0.0;
		float  h = 200.0f;
		for (int i = 0; i < 12; i++, t += 16.0) {
			struct bc_det d = det(600.0f, 400.0f, h * 0.5f, h, 0.9f);
			bc_tracker_update(&tr, &d, 1, t, censor, NULL);
			h += shrinking ? -14.0f : 14.0f;
		}

		/* Then take the detections away and let it coast a long way. */
		float wmin = 1e9f, hmin = 1e9f, wmax = 0.0f, hmax = 0.0f;
		for (int i = 0; i < 300; i++, t += 16.0) {
			bc_tracker_update(&tr, NULL, 0, t, censor, NULL);
			float x, y, bw, bh;
			for (int k = 0; k < BC_TRK_MAX; k++) {
				if (!bc_trk_box(&tr.t[k], &x, &y, &bw, &bh))
					continue;
				if (bw < wmin) wmin = bw;
				if (bh < hmin) hmin = bh;
				if (bw > wmax) wmax = bw;
				if (bh > hmax) hmax = bh;
			}
		}

		printf("    %s: w %.1f..%.1f  h %.1f..%.1f\n",
		       shrinking ? "shrinking" : "growing",
		       wmin, wmax, hmin, hmax);
		check(wmin > 0.0f && hmin > 0.0f,
		      shrinking ? "a shrinking lost track never inverts"
		                : "a growing lost track never inverts");
		/* The anchor is at most 200x400 by construction; twice that is
		 * the cap, and a 2560x1440 monitor is nowhere near it. */
		check(wmax <= 800.0f && hmax <= 1600.0f,
		      shrinking ? "  and stays bounded" : "  and cannot swell to the screen");
	}
}

/*
 * A held track must still age.
 *
 * The change gate holds every track on a frame it skipped, so that choosing
 * not to infer does not read as "nothing is there". The ceiling that bounds
 * how long it may do that used to read last_update -- the same clock the hold
 * moves forward -- so it asked "was this track held recently", answered yes,
 * and never fired. On a mostly still screen every track lived forever and the
 * boxes stacked up until the screen was covered.
 *
 * The measurement clock must survive any number of holds.
 */
static void test_holding_does_not_forge_a_measurement(const int *censor)
{
	struct bc_tracker tr;
	bc_tracker_init(&tr, 0.35f, 0.30f, 400.0);

	struct bc_det d = det(600, 400, 100, 200, 0.9f);
	bc_tracker_update(&tr, &d, 1, 0.0, censor, NULL);

	int live = -1;
	for (int i = 0; i < BC_TRK_MAX; i++)
		if (tr.t[i].state != BC_TRK_FREE)
			live = i;
	check(live >= 0, "a detection opens a track to hold");

	/* Five seconds of a still screen: gated frame after gated frame. */
	double t = 0.0;
	for (int i = 0; i < 300; i++) {
		t += 16.0;
		bc_tracker_hold(&tr, t);
	}

	printf("    after %.0f ms of holds: last_update %.0f, last_measured %.0f\n",
	       t, tr.t[live].last_update, tr.t[live].last_measured);
	check(tr.t[live].last_update >= t - 1.0,
	      "the hold keeps the track alive");
	check(tr.t[live].last_measured == 0.0,
	      "  but never moves the measurement clock");
	check(t - tr.t[live].last_measured > 1000.0,
	      "  so a 1000 ms re-verify ceiling can fire");

	/*
	 * And the ceiling is per track. One stale track must not cancel the
	 * hold for a track measured on this very frame, or a single ghost ages
	 * every box on screen.
	 */
	struct bc_tracker tr2;
	bc_tracker_init(&tr2, 0.35f, 0.30f, 400.0);
	struct bc_det old_d = det(100, 100, 100, 200, 0.9f);
	bc_tracker_update(&tr2, &old_d, 1, 0.0, censor, NULL);
	struct bc_det both[2] = { det(100, 100, 100, 200, 0.9f),
	                          det(900, 900, 100, 200, 0.9f) };
	bc_tracker_update(&tr2, both, 2, 1500.0, censor, NULL);
	/* Now age only the first by holding past the ceiling. */
	check(bc_tracker_hold_bounded(&tr2, 2000.0, 1000.0) == 0,
	      "two freshly measured tracks are both held");

	/*
	 * Re-measure only the first, holding throughout so the second stays
	 * alive -- which is the situation being tested: alive on holds alone,
	 * and long past its last real measurement.
	 */
	for (double th = 2100.0; th < 2500.0; th += 100.0)
		bc_tracker_hold_bounded(&tr2, th, 1000.0);
	bc_tracker_update(&tr2, &both[0], 1, 2500.0, censor, NULL);
	for (double th = 2600.0; th < 2900.0; th += 100.0)
		bc_tracker_hold_bounded(&tr2, th, 1000.0);
	int capped = bc_tracker_hold_bounded(&tr2, 2900.0, 1000.0);
	int held = 0;
	for (int i = 0; i < BC_TRK_MAX; i++)
		if (tr2.t[i].state != BC_TRK_FREE &&
		    tr2.t[i].last_update >= 2900.0)
			held++;
	printf("    one fresh, one stale: %d capped, %d held\n", capped, held);
	check(capped == 1, "  the stale one is left to age");
	check(held == 1, "  and does not cancel the fresh one's hold");
}

int main(void)
{
	int censor[BC_MAX_CLASSES];
	for (int i = 0; i < BC_MAX_CLASSES; i++)
		censor[i] = 1;

	printf("ByteTrack association:\n");
	test_one_to_one(censor);
	printf("Kalman prediction:\n");
	test_predicts_through_a_gap(censor);
	printf("Gate hold:\n");
	test_holding_does_not_forge_a_measurement(censor);
	printf("Prediction bounds:\n");
	test_coasting_size_stays_sane(censor);
	printf("Two-pass association:\n");
	test_low_scores_sustain_but_never_create(censor);
	printf("Per-class scores:\n");
	test_per_class_scores(censor);
	printf("Policy:\n");
	test_respects_the_class_policy();

	printf(failures ? "\n%d check(s) FAILED\n" : "\nAll checks passed.\n",
	       failures);
	return failures ? 1 : 0;
}
