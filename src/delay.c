#include "delay.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct snap {
	double          t;
	int             n;
	struct bc_box  *box;     /* BC_MAX_BOXES */
};

struct bc_dh {
	struct snap *s;
	int          cap;
	int          head;       /* next slot to write */
	int          count;
};

struct bc_dh *bc_dh_create(int slots)
{
	if (slots < 2)
		return NULL;
	struct bc_dh *h = calloc(1, sizeof *h);
	if (!h)
		return NULL;
	h->s = calloc((size_t)slots, sizeof *h->s);
	if (!h->s) {
		free(h);
		return NULL;
	}
	h->cap = slots;
	for (int i = 0; i < slots; i++) {
		h->s[i].box = calloc(BC_MAX_BOXES, sizeof *h->s[i].box);
		if (!h->s[i].box) {
			bc_dh_destroy(h);
			return NULL;
		}
	}
	return h;
}

void bc_dh_destroy(struct bc_dh *h)
{
	if (!h)
		return;
	for (int i = 0; i < h->cap; i++)
		free(h->s[i].box);
	free(h->s);
	free(h);
}

void bc_dh_reset(struct bc_dh *h)
{
	if (h)
		h->head = h->count = 0;
}

/* The i-th oldest record, 0 <= i < count. */
static const struct snap *at(const struct bc_dh *h, int i)
{
	return &h->s[(h->head - h->count + i + h->cap * 2) % h->cap];
}

double bc_dh_newest(const struct bc_dh *h)
{
	return h && h->count ? at(h, h->count - 1)->t : -1e18;
}

void bc_dh_push(struct bc_dh *h, double t, const struct bc_box *boxes, int n)
{
	if (!h || !(t > -1e17))
		return;
	if (h->count && t < bc_dh_newest(h))
		return;
	if (n < 0)
		n = 0;
	if (n > BC_MAX_BOXES)
		n = BC_MAX_BOXES;
	struct snap *s = &h->s[h->head];
	s->t = t;
	s->n = n;
	if (n)
		memcpy(s->box, boxes, (size_t)n * sizeof *boxes);
	h->head = (h->head + 1) % h->cap;
	if (h->count < h->cap)
		h->count++;
}

static float wrap_half(float a)
{
	const float pi = 3.14159265f;
	while (a >  pi / 2) a -= pi;
	while (a <= -pi / 2) a += pi;
	return a;
}

/* How much of the smaller box the intersection is, and whether the pair is
 * close enough to be the same subject moving. */
static int same_subject(const struct bc_box *a, const struct bc_box *b)
{
	if (a->effect != b->effect || a->over != b->over)
		return 0;
	const int x0 = a->x > b->x ? a->x : b->x;
	const int y0 = a->y > b->y ? a->y : b->y;
	const int x1 = a->x + a->w < b->x + b->w ? a->x + a->w : b->x + b->w;
	const int y1 = a->y + a->h < b->y + b->h ? a->y + a->h : b->y + b->h;
	const double inter = x1 > x0 && y1 > y0 ? (double)(x1 - x0) * (y1 - y0) : 0.0;
	const double ua = (double)a->w * a->h, ub = (double)b->w * b->h;
	const double uni = ua + ub - inter;
	if (uni > 0.0 && inter / uni >= 0.10)
		return 1;
	/* Moved by more than its own size between two looks: still the same
	 * subject if it is about the same size and not far. */
	const double ca = hypot(a->x + a->w / 2.0 - (b->x + b->w / 2.0),
	                        a->y + a->h / 2.0 - (b->y + b->h / 2.0));
	const double big = (double)(a->w > a->h ? a->w : a->h);
	const double r = ua > ub ? ub / (ua > 0 ? ua : 1) : ua / (ub > 0 ? ub : 1);
	return ca < 1.5 * big && r > 0.4;
}

static void lerp_box(struct bc_box *out, const struct bc_box *a,
                     const struct bc_box *b, float f)
{
	*out = *a;
	out->x = (int)lroundf(a->x + (b->x - a->x) * f);
	out->y = (int)lroundf(a->y + (b->y - a->y) * f);
	out->w = (int)lroundf(a->w + (b->w - a->w) * f);
	out->h = (int)lroundf(a->h + (b->h - a->h) * f);
	out->angle = wrap_half(a->angle + wrap_half(b->angle - a->angle) * f);
	/* A box present on both sides covers what either side covered: never
	 * smaller than the smaller of the two, which the lerp already is, and
	 * never missing the swept ends -- so grow it to the union when the pair
	 * is not close enough for the lerp to be trusted. */
}

int bc_dh_boxes(const struct bc_dh *h, double t, struct bc_box *out, int max,
                double *gap_ms)
{
	if (gap_ms)
		*gap_ms = 0.0;
	if (!h || h->count == 0)
		return -1;

	/* A: newest record at or before t. B: oldest after it. */
	const struct snap *A = NULL, *B = NULL;
	for (int i = 0; i < h->count; i++) {
		const struct snap *s = at(h, i);
		if (s->t <= t)
			A = s;
		else {
			B = s;
			break;
		}
	}
	if (!B && A && A->t >= t) {
		/* exactly on the newest record */
	} else if (!B) {
		return -1;                       /* nothing at or after t */
	}

	int n = 0;
	if (!A) {                                /* older than all history */
		for (int i = 0; i < B->n && n < max; i++)
			out[n++] = B->box[i];
		return n;
	}
	if (!B) {
		for (int i = 0; i < A->n && n < max; i++)
			out[n++] = A->box[i];
		return n;
	}

	const double span = B->t - A->t;
	if (gap_ms)
		*gap_ms = span;
	const int interp = span > 0.0 && span <= BC_DH_MAX_SPAN_MS;
	const float f = interp ? (float)((t - A->t) / span) : 0.0f;

	unsigned char usedB[BC_MAX_BOXES] = { 0 };
	for (int i = 0; i < A->n && n < max; i++) {
		int best = -1;
		for (int j = 0; j < B->n; j++) {
			if (usedB[j] || !same_subject(&A->box[i], &B->box[j]))
				continue;
			if (best < 0)
				best = j;
			else {
				/* the nearer centre wins */
				const double di = hypot(A->box[i].x - B->box[j].x,
				                        A->box[i].y - B->box[j].y);
				const double db = hypot(A->box[i].x - B->box[best].x,
				                        A->box[i].y - B->box[best].y);
				if (di < db)
					best = j;
			}
		}
		if (best >= 0 && interp) {
			usedB[best] = 1;
			lerp_box(&out[n++], &A->box[i], &B->box[best], f);
		} else {
			out[n++] = A->box[i];   /* unpaired, or too far apart to trust */
		}
	}
	for (int j = 0; j < B->n && n < max; j++)
		if (!usedB[j])
			out[n++] = B->box[j];
	return n;
}
