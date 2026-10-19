/*
 * Region stability -- the arithmetic the occlusion hold decides on.
 *
 * The hold is the answer to a loop that only appears on a compositor which
 * cannot hide the overlay from screen capture, so the situation it exists for
 * cannot be reproduced on the machine that wrote it. What *can* be checked
 * here is the decision itself: given two tensors and a box, does it say
 * "unchanged" when and only when it should.
 *
 * The thresholds in DESIGN.md §3[7] are the reference points: identical frames
 * score 0.000, two genuinely different screen captures score 0.036-0.080, and
 * the hold triggers below 0.010.
 */
#include "../src/hold.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W  64
#define H  64
#define CH 3
#define N  ((size_t)W * H * CH)

#define HOLD_MAD 0.010f   /* must match bc_main.c */

static int failures;

static void check(int cond, const char *what)
{
	printf("  %-58s %s\n", what, cond ? "ok" : "FAIL");
	if (!cond)
		failures++;
}

static void fill(float *t, float v)
{
	for (size_t i = 0; i < N; i++)
		t[i] = v;
}

/* Paint a rectangle into one plane, in tensor coordinates. */
static void paint(float *t, int plane, int x0, int y0, int x1, int y1, float v)
{
	for (int y = y0; y < y1; y++)
		for (int x = x0; x < x1; x++)
			t[(size_t)plane * W * H + (size_t)y * W + x] = v;
}

int main(void)
{
	float *a = malloc(N * sizeof *a);
	float *b = malloc(N * sizeof *b);
	if (!a || !b)
		return 1;

	printf("hold: region stability\n\n");

	printf("the basic verdict\n");
	{
		fill(a, 0.5f); fill(b, 0.5f);
		check(bc_hold_region_mad(a, b, W, H, CH, 0, 0, W, H) == 0.0f,
		      "identical tensors score exactly zero");

		/* A bar drawn over content: the region under it is a constant,
		 * so consecutive frames of it are identical. That is the whole
		 * signal the hold runs on. */
		fill(b, 0.5f);
		paint(a, 0, 10, 10, 20, 20, 0.9f);
		paint(b, 0, 10, 10, 20, 20, 0.9f);
		check(bc_hold_region_mad(a, b, W, H, CH, 10, 10, 20, 20) == 0.0f,
		      "a region that did not change scores zero");
		check(bc_hold_region_mad(a, b, W, H, CH, 10, 10, 20, 20) < HOLD_MAD,
		      "...and so is below the hold threshold");
	}

	printf("\nmovement under the cover releases it\n");
	{
		fill(a, 0.5f); fill(b, 0.5f);
		/* Half the region changes by 0.2: MAD over the box is 0.1 in
		 * one plane of three, i.e. ~0.033 -- inside the 0.036-0.080
		 * band DESIGN records for two different captures. */
		paint(a, 0, 10, 10, 20, 15, 0.7f);
		const float mad = bc_hold_region_mad(a, b, W, H, CH, 10, 10, 20, 20);
		check(mad > HOLD_MAD, "content moving under the box scores above it");
		printf("        (measured %.4f against a %.3f threshold)\n",
		       mad, HOLD_MAD);
	}

	printf("\nthe box is what is measured, not the frame\n");
	{
		fill(a, 0.5f); fill(b, 0.5f);
		/* Something moves elsewhere on screen. The whole-frame gate
		 * would call this frame "changed" -- which is exactly why the
		 * hold cannot be a whole-frame test. */
		paint(a, 0, 40, 40, 60, 60, 1.0f);
		check(bc_hold_region_mad(a, b, W, H, CH, 10, 10, 20, 20) == 0.0f,
		      "change outside the box does not release it");
		check(bc_hold_region_mad(a, b, W, H, CH, 40, 40, 60, 60) > HOLD_MAD,
		      "change inside a different box releases that one");
	}

	printf("\nevery plane counts\n");
	{
		fill(a, 0.5f); fill(b, 0.5f);
		/* Colour-only movement: a channel the test forgot to read is a
		 * channel content can move in unnoticed. */
		paint(a, 2, 10, 10, 20, 20, 0.9f);
		check(bc_hold_region_mad(a, b, W, H, CH, 10, 10, 20, 20) > 0.0f,
		      "a change in the third plane is seen");
	}

	printf("\nunanswerable questions never hold a box\n");
	{
		fill(a, 0.5f); fill(b, 0.5f);
		/* -1 rather than 0. Zero means identical, identical holds, and
		 * a box that cannot be measured must never be the one that
		 * becomes permanent. */
		check(bc_hold_region_mad(a, b, W, H, CH, 100, 100, 120, 120) < 0.0f,
		      "a box entirely off the tensor returns < 0");
		check(bc_hold_region_mad(a, b, W, H, CH, 10, 10, 10, 20) < 0.0f,
		      "a zero-width box returns < 0");
		check(bc_hold_region_mad(NULL, b, W, H, CH, 0, 0, W, H) < 0.0f,
		      "no current tensor returns < 0");
		check(bc_hold_region_mad(a, NULL, W, H, CH, 0, 0, W, H) < 0.0f,
		      "no previous tensor returns < 0");
	}

	printf("\nboxes that hang off the edge measure the part inside\n");
	{
		fill(a, 0.5f); fill(b, 0.5f);
		paint(a, 0, 0, 0, 8, 8, 0.9f);
		/* Dilate and growth both push boxes past the edge; clamping
		 * rather than refusing keeps those measurable, and reading past
		 * the buffer instead would be a crash on the safe path. */
		check(bc_hold_region_mad(a, b, W, H, CH, -20, -20, 8, 8) > 0.0f,
		      "a box starting off-tensor still measures its overlap");
		check(bc_hold_region_mad(a, b, W, H, CH, W - 4, H - 4,
		                         W + 40, H + 40) == 0.0f,
		      "a box running off the far edge measures what is inside");
	}

	free(a);
	free(b);
	printf("\n%s\n", failures ? "FAILURES" : "All checks passed.");
	return failures ? 1 : 0;
}
