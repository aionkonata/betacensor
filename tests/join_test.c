/*
 * Area joining. The properties that matter are geometric, so they are driven
 * directly rather than measured through content.
 *
 * The one that matters most is not "does it merge" but "does it ever uncover
 * anything" -- a joiner that loses a pixel of coverage is worse than no joiner
 * at all, and the failure would be invisible on screen.
 */
#include "../src/join.c"

#include <stdio.h>
#include <stdlib.h>

static int failures;

static void check(int ok, const char *what)
{
	printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
	if (!ok)
		failures++;
}

static struct bc_box bx(int x, int y, int w, int h)
{
	struct bc_box b = { 0 };
	b.x = x; b.y = y; b.w = w; b.h = h;
	b.a = 1.0f;
	return b;
}

/* Is (px,py) inside any of the n boxes? */
static int covered(const struct bc_box *b, int n, int px, int py)
{
	for (int i = 0; i < n; i++)
		if (px >= b[i].x && px < b[i].x + b[i].w &&
		    py >= b[i].y && py < b[i].y + b[i].h)
			return 1;
	return 0;
}

static void test_merges_the_obvious_case(void)
{
	struct bc_box b[2] = { bx(100, 100, 100, 100), bx(150, 100, 100, 100) };
	const int n = bc_join_boxes(b, 2, 6, 0.35f);
	check(n == 1, "two heavily overlapping boxes become one");
	check(b[0].x == 100 && b[0].w == 150 && b[0].y == 100 && b[0].h == 100,
	      "  and the region is their union");
}

static void test_never_uncovers(void)
{
	/* A cluster of the kind a torso produces: several parts, all touching. */
	struct bc_box b[6] = {
		bx(200, 200, 120, 160), bx(260, 240, 100, 100),
		bx(180, 300,  90,  90), bx(300, 180,  80, 140),
		bx(240, 320, 140,  60), bx(210, 210,  40,  40),
	};
	struct bc_box before[6];
	for (int i = 0; i < 6; i++)
		before[i] = b[i];

	const int n = bc_join_boxes(b, 6, 6, 0.35f);
	printf("    6 boxes -> %d\n", n);
	check(n < 6, "an overlapping cluster is reduced");

	int lost = 0;
	for (int y = 150; y < 420; y++)
		for (int x = 150; x < 420; x++)
			if (covered(before, 6, x, y) && !covered(b, n, x, y))
				lost++;
	printf("    pixels covered before but not after: %d\n", lost);
	check(lost == 0, "  and every covered pixel is still covered");
}

/*
 * The failure mode joining invites. Union is transitive: a chain of boxes that
 * each barely touch the next would join end to end into one rectangle spanning
 * the screen -- technically covering, useless in practice, and precisely the
 * "whole screen is pixelated" complaint arrived at from a different direction.
 */
static void test_refuses_to_swallow_the_screen(void)
{
	struct bc_box b[10];
	for (int i = 0; i < 10; i++)
		b[i] = bx(100 + i * 90, 100 + i * 90, 100, 100);

	const int n = bc_join_boxes(b, 10, 6, 0.35f);

	long area = 0;
	for (int i = 0; i < n; i++)
		area += (long)b[i].w * b[i].h;
	printf("    diagonal chain: %d boxes, %ld px total\n", n, area);

	/* The ten inputs cover at most 100000 px. A runaway union of the whole
	 * chain would be about 1000x1000. */
	check(area < 300000, "a diagonal chain does not merge into the screen");
}

static void test_leaves_eye_bars_alone(void)
{
	struct bc_box b[2] = { bx(100, 100, 100, 100), bx(110, 110, 100, 100) };
	b[1].angle = 0.3f;   /* an eye bar, bounded inside its own face */
	const int n = bc_join_boxes(b, 2, 6, 0.35f);
	check(n == 2, "a rotated box is never joined away");
	check(b[1].angle == 0.3f, "  and keeps its angle");
}

static void test_keeps_the_stronger_claim(void)
{
	struct bc_box b[2] = { bx(100, 100, 100, 100), bx(150, 100, 100, 100) };
	b[0].a = 0.25f; b[1].a = 1.0f;
	b[0].strength = 4.0f; b[1].strength = 9.0f;
	bc_join_boxes(b, 2, 6, 0.35f);
	check(b[0].a == 1.0f, "a join takes the higher alpha");
	check(b[0].strength == 9.0f, "  and the higher strength");
}

static void test_far_apart_boxes_stay_apart(void)
{
	struct bc_box b[2] = { bx(100, 100, 100, 100), bx(900, 900, 100, 100) };
	check(bc_join_boxes(b, 2, 6, 0.35f) == 2,
	      "boxes nowhere near each other are left alone");
}

int main(void)
{
	printf("Area joining:\n");
	test_merges_the_obvious_case();
	test_never_uncovers();
	test_refuses_to_swallow_the_screen();
	test_leaves_eye_bars_alone();
	test_keeps_the_stronger_claim();
	test_far_apart_boxes_stay_apart();

	printf(failures ? "\n%d check(s) FAILED\n" : "\nAll checks passed.\n",
	       failures);
	return failures ? 1 : 0;
}
