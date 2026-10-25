/*
 * The delayed screen's box history: certification, pairing and interpolation,
 * and that nothing is ever dropped on the way.
 */
#include "../src/delay.c"

#include <stdio.h>

static int failures;
static void check(int ok, const char *what)
{
	printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
	if (!ok)
		failures++;
}

static struct bc_box box(int x, int y, int w, int h)
{
	struct bc_box b;
	memset(&b, 0, sizeof b);
	b.x = x; b.y = y; b.w = w; b.h = h;
	return b;
}

int main(void)
{
	struct bc_box out[BC_MAX_BOXES];
	double gap;
	struct bc_dh *h = bc_dh_create(64);

	printf("certification\n");
	check(bc_dh_boxes(h, 0.0, out, BC_MAX_BOXES, &gap) == -1, "nothing recorded: nothing certified");
	struct bc_box a = box(100, 100, 200, 100);
	bc_dh_push(h, 1000.0, &a, 1);
	check(bc_dh_boxes(h, 1000.0, out, BC_MAX_BOXES, &gap) == 1, "a frame exactly on a record is certified");
	check(bc_dh_boxes(h, 1010.0, out, BC_MAX_BOXES, &gap) == -1, "a frame after the newest record is not");
	check(bc_dh_boxes(h, 900.0, out, BC_MAX_BOXES, &gap) == 1 && out[0].x == 100,
	      "a frame before all records takes the first one");

	printf("interpolation\n");
	struct bc_box b = box(300, 140, 200, 100);
	bc_dh_push(h, 1100.0, &b, 1);
	int n = bc_dh_boxes(h, 1050.0, out, BC_MAX_BOXES, &gap);
	check(n == 1 && out[0].x == 200 && out[0].y == 120, "halfway between two looks it is halfway between the boxes");
	check(gap == 100.0, "and reports the span it was built from");
	n = bc_dh_boxes(h, 1025.0, out, BC_MAX_BOXES, &gap);
	check(n == 1 && out[0].x == 150, "a quarter of the way is a quarter of the way");

	printf("nothing is dropped\n");
	struct bc_box two[2] = { box(300, 140, 200, 100), box(1000, 600, 100, 100) };
	bc_dh_push(h, 1200.0, two, 2);
	n = bc_dh_boxes(h, 1150.0, out, BC_MAX_BOXES, &gap);
	check(n == 2, "a box that appears between two looks is still drawn");
	int found = 0;
	for (int i = 0; i < n; i++)
		found += out[i].x == 1000;
	check(found == 1, "...where the later look put it");
	struct bc_box one = box(310, 140, 200, 100);
	bc_dh_push(h, 1300.0, &one, 1);
	n = bc_dh_boxes(h, 1250.0, out, BC_MAX_BOXES, &gap);
	found = 0;
	for (int i = 0; i < n; i++)
		found += out[i].x == 1000;
	check(n == 2 && found == 1, "a box that vanishes between two looks is drawn until the later one");

	printf("a screen that sat still and then changed\n");
	bc_dh_reset(h);
	struct bc_box s0 = box(100, 100, 200, 100), s1 = box(900, 500, 200, 100);
	bc_dh_push(h, 0.0, &s0, 1);
	bc_dh_push(h, 2000.0, &s1, 1);
	n = bc_dh_boxes(h, 1000.0, out, BC_MAX_BOXES, &gap);
	check(n == 2, "across a long gap both sides are drawn whole, not slid along it");

	printf("pairing\n");
	bc_dh_reset(h);
	struct bc_box p0[2] = { box(100, 100, 100, 100), box(600, 100, 100, 100) };
	struct bc_box p1[2] = { box(620, 100, 100, 100), box(120, 100, 100, 100) };  /* order swapped */
	bc_dh_push(h, 0.0, p0, 2);
	bc_dh_push(h, 100.0, p1, 2);
	n = bc_dh_boxes(h, 50.0, out, BC_MAX_BOXES, &gap);
	int xs[2] = { 0, 0 };
	for (int i = 0; i < n && i < 2; i++)
		xs[i] = out[i].x;
	check(n == 2 && ((xs[0] == 110 && xs[1] == 610) || (xs[0] == 610 && xs[1] == 110)),
	      "boxes are paired by place, not by order");
	struct bc_box e0 = box(100, 100, 100, 100); e0.effect = 1;
	struct bc_box e1 = box(110, 100, 100, 100); e1.effect = 2;
	bc_dh_reset(h);
	bc_dh_push(h, 0.0, &e0, 1);
	bc_dh_push(h, 100.0, &e1, 1);
	check(bc_dh_boxes(h, 50.0, out, BC_MAX_BOXES, &gap) == 2, "different effects are different subjects");

	printf("angle\n");
	struct bc_box r0 = box(100, 100, 200, 60), r1 = box(100, 100, 200, 60);
	r0.angle = 1.50f; r1.angle = -1.50f;          /* 0.14 rad apart across the half-turn */
	bc_dh_reset(h);
	bc_dh_push(h, 0.0, &r0, 1);
	bc_dh_push(h, 100.0, &r1, 1);
	n = bc_dh_boxes(h, 50.0, out, BC_MAX_BOXES, &gap);
	check(n == 1 && fabsf(fabsf(out[0].angle) - 1.5708f) < 0.05f, "the bar turns the short way round");

	printf("history\n");
	bc_dh_reset(h);
	for (int i = 0; i < 200; i++)
		bc_dh_push(h, 10.0 * i, &a, 1);            /* wraps a 64-slot ring */
	check(bc_dh_newest(h) == 1990.0, "newest survives wrapping");
	check(bc_dh_boxes(h, 1500.0, out, BC_MAX_BOXES, &gap) == 1, "recent frames still bracketed after wrapping");
	bc_dh_push(h, 5.0, &a, 1);
	check(bc_dh_newest(h) == 1990.0, "a record from the past is ignored");

	bc_dh_destroy(h);
	if (failures) {
		printf("\n%d FAILED\n", failures);
		return 1;
	}
	printf("\nAll checks passed.\n");
	return 0;
}
