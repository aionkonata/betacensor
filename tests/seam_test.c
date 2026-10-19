/*
 * Rejoining detections that a tile seam cut in half.
 *
 * The case that matters is the one plain IoU gets wrong, so the first test
 * measures the IoU of the very pair it has to merge and asserts that it is low
 * -- otherwise the whole mechanism would be an elaborate way of writing an IoU
 * threshold, and a later "simplification" back to one would look safe.
 */
#include "../src/seam.c"

#include <stdio.h>

static int failures;

static void check(int ok, const char *what)
{
	printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
	if (!ok)
		failures++;
}

static struct bc_det dt(int tile, float x, float y, float w, float h,
                        int cls, float score)
{
	struct bc_det d = { 0 };
	d.tile = tile; d.x = x; d.y = y; d.w = w; d.h = h;
	d.cls = cls; d.score = score;
	return d;
}

/* Everything is its own class unless told otherwise. */
static int exact(int a, int b) { return a == b; }
/* Classes 0 and 1 are the same object seen two ways -- a gender flip. */
static int similar_01(int a, int b)
{
	if (a == b)
		return 1;
	return (a == 0 && b == 1) || (a == 1 && b == 0);
}

static float iou(const struct bc_det *p, const struct bc_det *q)
{
	const float in = overlap_area(p, q);
	const float un = p->w * p->h + q->w * q->h - in;
	return un > 0.0f ? in / un : 0.0f;
}

/*
 * A 2-column split with a 100 px strip. An object spanning the seam is seen
 * as its left half by tile 0 and its right half by tile 1, and each half is
 * clipped to the tile's own edge -- so they meet only inside the strip.
 */
static void test_rejoins_a_straddling_object(void)
{
	/* Seam strip runs x = 1200..1300. */
	struct bc_det d[2] = {
		dt(0, 1000, 400, 300, 500, 3, 0.7f),   /* 1000..1300 */
		dt(1, 1200, 400, 400, 500, 3, 0.8f),   /* 1200..1600 */
	};

	printf("    the two halves have IoU %.3f\n", iou(&d[0], &d[1]));
	check(iou(&d[0], &d[1]) < 0.45f,
	      "the halves' IoU is too low for an IoU merge to catch");

	const int n = bc_seam_merge(d, 2, 2, -1, 100.0f, 0.0f, 0.4f, exact);
	check(n == 1, "  but they are rejoined anyway");
	check(d[0].x == 1000.0f && d[0].w == 600.0f,
	      "  into the whole object, not either half");
	check(d[0].score > 0.74f && d[0].score < 0.76f,
	      "  with the two scores averaged");
}

/* Two genuinely different objects, both near the seam but not the same thing. */
static void test_leaves_separate_objects_alone(void)
{
	struct bc_det d[2] = {
		dt(0, 1000, 100, 200, 200, 3, 0.7f),
		dt(1, 1250, 700, 200, 200, 3, 0.8f),
	};
	check(bc_seam_merge(d, 2, 2, -1, 100.0f, 0.0f, 0.4f, exact) == 2,
	      "two objects far apart across the seam stay separate");
}

/* Same tile: the model's own NMS already ran, so anything left is real. */
static void test_never_merges_within_a_tile(void)
{
	struct bc_det d[2] = {
		dt(0, 1000, 400, 300, 500, 3, 0.7f),
		dt(0, 1010, 405, 300, 500, 3, 0.8f),
	};
	check(bc_seam_merge(d, 2, 2, -1, 100.0f, 0.0f, 0.4f, exact) == 2,
	      "two detections from the same tile are never merged");
}

/* A model that flips gender across the seam must still produce one box. */
static void test_merges_across_a_class_flip(void)
{
	struct bc_det d[2] = {
		dt(0, 1000, 400, 300, 500, 0, 0.6f),
		dt(1, 1200, 400, 400, 500, 1, 0.9f),
	};
	check(bc_seam_merge(d, 2, 2, -1, 100.0f, 0.0f, 0.4f, similar_01) == 1,
	      "a gender flip across the seam still merges");
	check(d[0].cls == 1, "  and the more confident label wins");

	struct bc_det e[2] = {
		dt(0, 1000, 400, 300, 500, 0, 0.6f),
		dt(1, 1200, 400, 400, 500, 1, 0.9f),
	};
	check(bc_seam_merge(e, 2, 2, -1, 100.0f, 0.0f, 0.4f, exact) == 2,
	      "  but unrelated classes do not");
}

/*
 * A 2x2 grid. Tiles above and below share a horizontal strip only, so a pair
 * split that way must be judged on the vertical overlap -- using the
 * horizontal strip there would reject a perfectly good pair.
 */
static void test_handles_a_vertical_seam(void)
{
	struct bc_det d[2] = {
		dt(0, 500, 500, 400, 300, 3, 0.7f),   /* top-left  */
		dt(2, 500, 700, 400, 300, 3, 0.7f),   /* below it  */
	};
	check(bc_seam_merge(d, 2, 2, -1, 100.0f, 100.0f, 0.4f, exact) == 1,
	      "a pair split top-to-bottom merges on the vertical strip");
	check(d[0].y == 500.0f && d[0].h == 500.0f, "  into their union");
}

/* Three tiles meeting means a merged pair can then reach a third. */
static void test_reaches_a_fixed_point(void)
{
	struct bc_det d[3] = {
		dt(0, 1000, 400, 300, 500, 3, 0.7f),
		dt(1, 1200, 400, 300, 500, 3, 0.7f),
		dt(2, 1000, 700, 500, 300, 3, 0.7f),
	};
	const int n = bc_seam_merge(d, 3, 2, -1, 150.0f, 150.0f, 0.4f, exact);
	printf("    3 detections -> %d\n", n);
	check(n == 1, "a chain across three tiles collapses to one");
}

/* With no overlap there is no strip, so nothing could have been seen twice. */
static void test_no_overlap_means_no_merge(void)
{
	struct bc_det d[2] = {
		dt(0, 1000, 400, 300, 500, 3, 0.7f),
		dt(1, 1200, 400, 400, 500, 3, 0.8f),
	};
	check(bc_seam_merge(d, 2, 2, -1, 0.0f, 0.0f, 0.4f, exact) == 2,
	      "without a shared strip nothing is merged");
}

/*
 * The full-frame tile.
 *
 * It sits after the grid (2x2 here, so tile 4) and holds the whole screen at
 * the letterbox scale. A pair involving it is not a seam pair: neither box is
 * truncated, so the strip caps do not apply and the question is simply whether
 * one contains the other. Its tile index has no meaningful row or column --
 * 4 % 2 and 4 / 2 read as column 0, row 2 -- which is exactly what the caps
 * would be computed from if it were mistaken for a grid tile.
 */
static void test_merges_a_full_frame_detection_with_a_tile(void)
{
	struct bc_det d[2] = {
		dt(4, 1000, 400, 600, 500, 3, 0.6f),  /* the whole object */
		dt(1, 1200, 400, 400, 500, 3, 0.8f),  /* part of it, closer up */
	};
	check(bc_seam_merge(d, 2, 2, 4, 100.0f, 100.0f, 0.4f, exact) == 1,
	      "a full-frame detection and a tile's merge into one");
	check(d[0].x == 1000.0f && d[0].w == 600.0f,
	      "  covering the whole object");
}

/*
 * The overlap is a property of the grid, and the full-frame tile is not in it.
 * Gating its merges on a shared strip would mean a zero-overlap grid could
 * never reconcile the two scales, and the screen would carry both a big box and
 * a small one inside it for the same object.
 */
static void test_full_frame_merges_without_a_strip(void)
{
	struct bc_det d[2] = {
		dt(4, 1000, 400, 600, 500, 3, 0.6f),
		dt(1, 1200, 400, 400, 500, 3, 0.8f),
	};
	check(bc_seam_merge(d, 2, 2, 4, 0.0f, 0.0f, 0.4f, exact) == 1,
	      "a full-frame pair merges with no shared strip at all");
}

/* Containment is still a test, not a licence to merge anything it saw. */
static void test_full_frame_leaves_distant_objects_alone(void)
{
	struct bc_det d[2] = {
		dt(4, 100, 100, 200, 200, 3, 0.6f),
		dt(1, 1200, 700, 200, 200, 3, 0.8f),
	};
	check(bc_seam_merge(d, 2, 2, 4, 100.0f, 100.0f, 0.4f, exact) == 2,
	      "a full-frame detection elsewhere on screen stays separate");
}

/* Grid pairs must keep behaving as grid pairs when a full tile exists. */
static void test_seam_still_works_alongside_a_full_frame_tile(void)
{
	struct bc_det d[2] = {
		dt(0, 1000, 400, 300, 500, 3, 0.7f),
		dt(1, 1200, 400, 400, 500, 3, 0.8f),
	};
	check(bc_seam_merge(d, 2, 2, 4, 100.0f, 0.0f, 0.4f, exact) == 1,
	      "a straddling pair still rejoins with a full tile present");
}

int main(void)
{
	printf("Seam merge:\n");
	test_rejoins_a_straddling_object();
	test_leaves_separate_objects_alone();
	test_never_merges_within_a_tile();
	test_merges_across_a_class_flip();
	test_handles_a_vertical_seam();
	test_reaches_a_fixed_point();
	test_no_overlap_means_no_merge();
	test_merges_a_full_frame_detection_with_a_tile();
	test_full_frame_merges_without_a_strip();
	test_full_frame_leaves_distant_objects_alone();
	test_seam_still_works_alongside_a_full_frame_tile();

	printf(failures ? "\n%d check(s) FAILED\n" : "\nAll checks passed.\n",
	       failures);
	return failures ? 1 : 0;
}
