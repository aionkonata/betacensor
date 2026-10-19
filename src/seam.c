#include "seam.h"

#include <math.h>

static float fmin2(float a, float b) { return a < b ? a : b; }
static float fmax2(float a, float b) { return a > b ? a : b; }

/* Area the two boxes share. */
static float overlap_area(const struct bc_det *p, const struct bc_det *q)
{
	const float x0 = fmax2(p->x, q->x);
	const float y0 = fmax2(p->y, q->y);
	const float x1 = fmin2(p->x + p->w, q->x + q->w);
	const float y1 = fmin2(p->y + p->h, q->y + q->h);
	if (x1 <= x0 || y1 <= y0)
		return 0.0f;
	return (x1 - x0) * (y1 - y0);
}

/*
 * The most these two could overlap, given where they were seen.
 *
 * On an axis the pair is split across, neither box can reach further into the
 * other's tile than the shared strip, so the strip caps that axis. On an axis
 * they share, there is no such cap and the smaller box's own extent is the
 * limit -- which makes this reduce to "the smaller box" for a pair that is not
 * split at all.
 */
static float reachable_area(const struct bc_det *p, const struct bc_det *q,
                            int split_x, int split_y, float ox, float oy)
{
	float w = fmin2(p->w, q->w);
	float h = fmin2(p->h, q->h);
	if (split_x)
		w = fmin2(w, ox);
	if (split_y)
		h = fmin2(h, oy);
	return w * h;
}

int bc_seam_merge(struct bc_det *d, int n, int cols, int full, float ox,
                  float oy, float thresh, int (*similar)(int a, int b))
{
	if (n < 2 || cols < 1)
		return n;
	if (thresh <= 0.0f)
		thresh = 0.4f;

	/*
	 * Repeat to a fixed point: three tiles can meet at a corner, so a box
	 * merged from two of them may then reach the third. Bounded by n --
	 * every pass that changes anything removes one detection.
	 */
	for (int pass = 0; pass < n; pass++) {
		int merged = 0;

		for (int i = 0; i < n && !merged; i++) {
			for (int j = i + 1; j < n && !merged; j++) {
				if (d[i].tile == d[j].tile)
					continue;   /* the model already ran NMS
					             * within a tile */
				if (similar && !similar(d[i].cls, d[j].cls))
					continue;

				/*
				 * The full-frame tile is not in the grid, so it
				 * is not split from anything: it contains every
				 * other tile whole. Treating it as unsplit makes
				 * reachable_area() fall back to the smaller box,
				 * which turns the test into containment -- the
				 * right question for the same object seen large
				 * and small, and the reason a full-frame tile
				 * can be merged in at all.
				 */
				const int cross_scale = full >= 0 &&
				                        (d[i].tile == full ||
				                         d[j].tile == full);
				const int split_x = !cross_scale &&
				                    (d[i].tile % cols)
				                    != (d[j].tile % cols);
				const int split_y = !cross_scale &&
				                    (d[i].tile / cols)
				                    != (d[j].tile / cols);
				/* Nothing shared on a split axis means the tiles
				 * do not actually touch: there is no strip for a
				 * straddling object to have been seen in. */
				if ((split_x && !(ox > 0.0f)) ||
				    (split_y && !(oy > 0.0f)))
					continue;

				const float ref = reachable_area(&d[i], &d[j],
				                                 split_x, split_y,
				                                 ox, oy);
				if (!(ref > 0.0f))
					continue;
				if (overlap_area(&d[i], &d[j]) < thresh * ref)
					continue;

				/* Union: the whole object is at least the two
				 * halves together, and growing a cover is the
				 * safe direction. */
				const float x0 = fmin2(d[i].x, d[j].x);
				const float y0 = fmin2(d[i].y, d[j].y);
				const float x1 = fmax2(d[i].x + d[i].w,
				                       d[j].x + d[j].w);
				const float y1 = fmax2(d[i].y + d[i].h,
				                       d[j].y + d[j].h);

				/*
				 * The more confident label wins. The two halves
				 * only had to be *similar* to get here, so they
				 * may disagree -- and the half that saw more of
				 * the object is the one to believe.
				 */
				if (d[j].score > d[i].score) {
					d[i].cls   = d[j].cls;
					d[i].n_kpt = d[j].n_kpt;
					for (int k = 0; k < d[j].n_kpt &&
					                k < BC_MAX_KPT; k++) {
						d[i].kx[k] = d[j].kx[k];
						d[i].ky[k] = d[j].ky[k];
						d[i].ks[k] = d[j].ks[k];
					}
				}
				d[i].x = x0;
				d[i].y = y0;
				d[i].w = x1 - x0;
				d[i].h = y1 - y0;
				/* Averaged, as the two are one observation of
				 * the same thing seen twice. */
				d[i].score = (d[i].score + d[j].score) * 0.5f;

				d[j] = d[n - 1];
				n--;
				merged = 1;
			}
		}

		if (!merged)
			break;
	}

	return n;
}
