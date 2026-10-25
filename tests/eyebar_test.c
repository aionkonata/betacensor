/*
 * Which landmark answers become an eye bar, and which fall back to the band.
 *
 * The numbers in the rejection cases are measured, not invented: they are what
 * the landmark model returned on the images it was compared on, where its
 * confidence score could not tell a wrong answer from a right one.
 */
#include "../src/eyebar.c"

#include <stdio.h>

static int failures;

static void check(int ok, const char *what)
{
	printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
	if (!ok)
		failures++;
}

static int near(float a, float b, float tol) { return fabsf(a - b) <= tol; }

int main(void)
{
	struct bc_eyebar e;

	printf("a face the model got right\n");
	/* Mai: face 498x504 at (307,318), eyes (432,504) and (688,507), 0.5 deg. */
	int r = bc_eyebar_from_eyes(432, 504, 688, 507, 0, 0, 1.45f, 0,
	                            307, 318, 498, 504, &e);
	check(r == 1, "accepted");
	check(near(e.cx, 560, 0.5f) && near(e.cy, 505.5f, 0.5f),
	      "centred between the eyes");
	check(near(e.angle * 57.29578f, -0.67f, 0.1f),
	      "angle follows the eye line (y flipped: GL's up)");
	check(near(e.w, 256.0f * BC_EB_KPT_BAR_WIDE, 1.0f),
	      "no measured extents: length is 1.9 x separation");

	printf("measured extents, and the floor under them\n");
	r = bc_eyebar_from_eyes(432, 504, 688, 507, 170, 30, 1.45f, 0,
	                        307, 318, 498, 504, &e);
	check(r == 1 && near(e.w, 2 * 170 * 1.45f, 1.0f) &&
	      near(e.w, 493, 1.0f), "wide extents are used, padded");
	check(near(e.h, 256.0f * BC_EB_KPT_BAR_TALL, 1.0f),
	      "but never thinner than 0.6 x separation (§2)");
	r = bc_eyebar_from_eyes(432, 504, 688, 507, 40, 20, 1.45f, 1,
	                        307, 318, 498, 504, &e);
	check(r == 1 && near(e.w, 2 * 40 * 1.45f, 1.0f),
	      "nofloor lets the extents shrink it");

	printf("a face turned away: both 'eyes' on one spot\n");
	/* Meowscarada: face 177x178 at (271,129); eyes 12 px apart, 0.37 score. */
	r = bc_eyebar_from_eyes(336, 197, 341, 208, 0, 0, 1.45f, 0,
	                        271, 129, 177, 178, &e);
	check(r == -1, "rejected: separation is 7% of the face, not 20-90%");

	printf("eyes the model put somewhere else\n");
	r = bc_eyebar_from_eyes(100, 100, 300, 100, 0, 0, 1.45f, 0,
	                        400, 400, 200, 200, &e);
	check(r == -1, "outside the face box: rejected");
	r = bc_eyebar_from_eyes(400, 400, 600, 400, 0, 0, 1.45f, 0,
	                        400, 400, 200, 200, &e);
	check(r == -1, "separation as wide as the face (100%): rejected");
	r = bc_eyebar_from_eyes(385, 480, 550, 480, 0, 0, 1.45f, 0,
	                        400, 400, 200, 200, &e);
	check(r == 1, "a little outside, within the 15% margin: accepted");

	printf("degenerate\n");
	r = bc_eyebar_from_eyes(500, 500, 500, 500, 0, 0, 1.45f, 0,
	                        400, 400, 800, 800, &e);
	check(r == -1 || r == 0, "coincident eyes never make a bar");

	printf("angles wrap to a half turn\n");
	check(near(bc_eyebar_wrap(3.0f), 3.0f - 3.14159265f, 1e-4f),
	      "3 rad is -0.14 rad (the bar is symmetric)");
	check(near(bc_eyebar_wrap(-1.57079633f), 1.57079633f, 1e-4f),
	      "-pi/2 maps to +pi/2");
	/* A face upside down: the eye line points the other way, same bar. */
	struct bc_eyebar a, b;
	bc_eyebar_from_eyes(432, 504, 688, 507, 0, 0, 1.45f, 0, 307, 318, 498, 504, &a);
	bc_eyebar_from_eyes(688, 507, 432, 504, 0, 0, 1.45f, 0, 307, 318, 498, 504, &b);
	check(near(a.angle, b.angle, 1e-4f), "left/right order does not change the bar");

	printf("the band\n");
	bc_eyebar_band(100, 200, 300, 400, BC_EB_BAND_TOP, BC_EB_BAND_BOT, &e);
	check(near(e.w, 300 * 0.92f, 0.01f), "inset 4% each side");
	check(near(e.h, 400 * 0.56f, 0.01f), "22% to 78% of the height");
	check(near(e.cx, 250, 0.01f) && near(e.cy, 200 + 400 * 0.5f, 0.01f),
	      "centred on the face horizontally, mid-band vertically");
	check(e.angle == 0.0f, "upright");

	if (failures) {
		printf("\n%d FAILED\n", failures);
		return 1;
	}
	printf("\nAll checks passed.\n");
	return 0;
}
