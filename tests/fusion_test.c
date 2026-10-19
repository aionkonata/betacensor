/*
 * Does box fusion actually stop the pulsing? (DESIGN.md, "Weighted Box Fusion")
 *
 * Measuring this through the screen does not work: the number you get is
 * dominated by whatever content happens to be playing, and an A/B taken across
 * a content change measures the content, not the change. So drive the fusion
 * directly with a synthetic sequence that isolates the mechanism.
 *
 * The mechanism: one object, two overlapping candidate boxes -- a tight one and
 * a loose one -- whose scores sit close together and swap rank every frame, as a
 * detector's scores genuinely do on content that is not moving. NMS must pick
 * one, so its output teleports between the two geometries on every swap. WBF
 * averages them, so a rank swap moves the result by the weight difference only.
 *
 * Builds the real detect.c, so this tests the shipped code rather than a copy.
 */
#include "../src/detect.c"

#include <stdio.h>

#define FRAMES 40

/* Sum of the four edge movements, the same "box jump" the censor reports. */
static float box_jump(const struct bc_det *a, const struct bc_det *b)
{
	return fabsf(a->x - b->x) + fabsf(a->y - b->y) +
	       fabsf(a->w - b->w) + fabsf(a->h - b->h);
}

static void run(const char *label,
                int (*fuse)(const struct bc_det *, int, struct bc_det *, int),
                float *out_mean, float *out_max)
{
	/* One object, two candidates for it. IoU 0.64, so both survive the
	 * decode and land in the same cluster either way. */
	const struct bc_det tight = { .x = 100, .y = 100, .w = 100, .h = 200,
	                              .cls = 0 };
	const struct bc_det loose = { .x =  90, .y =  85, .w = 130, .h = 240,
	                              .cls = 0 };

	struct bc_det prev = { 0 };
	float sum = 0.0f, peak = 0.0f;
	int   n_meas = 0;

	for (int f = 0; f < FRAMES; f++) {
		/* Scores straddle each other: 0.62/0.58, swapping each frame. */
		struct bc_det cand[2] = { tight, loose };
		cand[0].score = (f & 1) ? 0.58f : 0.62f;
		cand[1].score = (f & 1) ? 0.62f : 0.58f;

		qsort(cand, 2, sizeof *cand, det_cmp);

		struct bc_det out[BC_MAX_DET];
		int n = fuse(cand, 2, out, BC_MAX_DET);
		if (n != 1) {
			printf("  %s: FAIL -- expected 1 fused box, got %d\n",
			       label, n);
			*out_mean = *out_max = -1.0f;
			return;
		}

		if (f > 0) {
			float d = box_jump(&out[0], &prev);
			sum += d;
			if (d > peak)
				peak = d;
			n_meas++;
		}
		prev = out[0];
	}

	*out_mean = n_meas ? sum / (float)n_meas : 0.0f;
	*out_max  = peak;
	printf("  %-18s mean %6.2f px   max %6.2f px  (per frame, 4 edges)\n",
	       label, *out_mean, *out_max);
}

int main(void)
{
	printf("Box jump on a stationary object whose candidate ranking flips\n"
	       "every frame -- lower is steadier:\n");

	float nms_mean, nms_max, wbf_mean, wbf_max;
	run("greedy NMS", nms_greedy, &nms_mean, &nms_max);
	run("weighted fusion", wbf, &wbf_mean, &wbf_max);

	if (nms_mean < 0.0f || wbf_mean < 0.0f)
		return 1;

	/*
	 * The claim being tested is not "WBF is a bit better" but "WBF removes
	 * this failure mode", so require a large margin rather than any
	 * improvement at all. Nothing moved on screen; a steady tracker should
	 * report a steady box.
	 */
	if (!(wbf_mean * 10.0f < nms_mean)) {
		printf("FAIL: fusion did not remove the pulsing "
		       "(%.2f vs %.2f px)\n", wbf_mean, nms_mean);
		return 1;
	}
	if (!(wbf_max < 5.0f)) {
		printf("FAIL: fused box still jumps by %.2f px on a still "
		       "object\n", wbf_max);
		return 1;
	}

	printf("PASS -- fusion cuts the jump %.0fx, to under a pixel per edge.\n",
	       nms_mean / (wbf_mean > 0.0f ? wbf_mean : 1e-6f));
	return 0;
}
