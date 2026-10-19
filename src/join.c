#include "join.h"

#include <string.h>

/* Area of a rectangle, in a width that cannot overflow on a large display. */
static long area_of(const struct bc_box *b)
{
	if (b->w <= 0 || b->h <= 0)
		return 0;
	return (long)b->w * (long)b->h;
}

/* Area the two cover between them: A + B - overlap, so it is not double
 * counted. Without subtracting it, two nearly identical boxes would look like
 * twice the area and any union would seem cheap. */
static long covered_together(const struct bc_box *p, const struct bc_box *q)
{
	const int ox0 = p->x > q->x ? p->x : q->x;
	const int oy0 = p->y > q->y ? p->y : q->y;
	const int ox1 = (p->x + p->w < q->x + q->w) ? p->x + p->w : q->x + q->w;
	const int oy1 = (p->y + p->h < q->y + q->h) ? p->y + p->h : q->y + q->h;

	long ov = 0;
	if (ox1 > ox0 && oy1 > oy0)
		ov = (long)(ox1 - ox0) * (long)(oy1 - oy0);
	return area_of(p) + area_of(q) - ov;
}

/* Do they touch, allowing `gap` px of slack? */
static int adjacent(const struct bc_box *p, const struct bc_box *q, int gap)
{
	return p->x - gap < q->x + q->w && q->x - gap < p->x + p->w &&
	       p->y - gap < q->y + q->h && q->y - gap < p->y + p->h;
}

/*
 * A rotated box is an eye bar, and an eye bar is hard-bounded to sit inside
 * its own face (see the clamp in bc_main.c). Merging one into an upright
 * region would produce a rectangle that satisfies neither constraint: the
 * union has no angle to inherit, so the bar's orientation -- the whole point
 * of it -- would be silently dropped. They are left alone.
 */
static int joinable(const struct bc_box *b)
{
	return b->angle == 0.0f && b->w > 0 && b->h > 0;
}

int bc_join_boxes(struct bc_box *b, int n, int gap, float waste)
{
	if (n < 2)
		return n;
	if (waste < 0.0f)
		waste = 0.0f;

	/*
	 * Repeat to a fixed point rather than in one pass. A merge creates a
	 * bigger rectangle that may now reach a third box which neither
	 * original reached, and stopping after one pass would leave that third
	 * box drawn on top of the region that now contains it -- which is the
	 * stacking this exists to remove, just less of it.
	 *
	 * The bound on iterations is n: each pass that changes anything removes
	 * at least one box.
	 */
	for (int pass = 0; pass < n; pass++) {
		int merged = 0;

		for (int i = 0; i < n && !merged; i++) {
			if (!joinable(&b[i]))
				continue;
			for (int j = i + 1; j < n && !merged; j++) {
				if (!joinable(&b[j]))
					continue;
				if (!adjacent(&b[i], &b[j], gap))
					continue;

				const int ux0 = b[i].x < b[j].x ? b[i].x : b[j].x;
				const int uy0 = b[i].y < b[j].y ? b[i].y : b[j].y;
				const int ux1 = (b[i].x + b[i].w > b[j].x + b[j].w)
				                ? b[i].x + b[i].w : b[j].x + b[j].w;
				const int uy1 = (b[i].y + b[i].h > b[j].y + b[j].h)
				                ? b[i].y + b[i].h : b[j].y + b[j].h;

				const long uni  = (long)(ux1 - ux0) * (long)(uy1 - uy0);
				const long have = covered_together(&b[i], &b[j]);
				if (have <= 0)
					continue;
				if ((double)(uni - have) > (double)have * waste)
					continue;   /* too much new area for one region */

				/*
				 * The merged region keeps the stronger claim of
				 * the two. Alpha is the max because the join must
				 * not fade a box that was at full strength, and
				 * strength is the max because it is a percentage
				 * of the box's own short side -- the region is
				 * larger than either input, so the smaller
				 * percentage would coarsen less than the box it
				 * replaced.
				 */
				b[i].x = ux0;
				b[i].y = uy0;
				b[i].w = ux1 - ux0;
				b[i].h = uy1 - uy0;
				if (b[j].a > b[i].a)
					b[i].a = b[j].a;
				if (b[j].strength > b[i].strength)
					b[i].strength = b[j].strength;
				/* Text: keep whichever we already had, so a
				 * caption does not vanish into a join. */
				if (!b[i].text[0])
					memcpy(b[i].text, b[j].text, sizeof b[i].text);

				b[j] = b[n - 1];
				n--;
				merged = 1;
			}
		}

		if (!merged)
			break;
	}

	return n;
}
