#ifndef BC_DELAY_H
#define BC_DELAY_H

#include "overlay.h"   /* struct bc_box */

/*
 * The box side of the delayed screen.
 *
 * With the screen delayed, the frame being shown is older than the newest
 * detection, so the boxes for it do not have to be guessed: they are known
 * from both sides. Every time the censor learns something about the screen --
 * a detection arrives, or the change gate establishes that nothing has moved --
 * the boxes it would draw are recorded against the capture time of the frame
 * that was looked at. To show the frame from `t`, take the last record at or
 * before it and the first after it, pair up their boxes and place each pair
 * where it would have been at `t`.
 *
 * Two rules keep this on the right side of "prefer covering too much":
 *
 *   - A frame is *certified* only once something at or after it has been
 *     looked at. Until then there is nothing to bracket it with, and the
 *     caller must not show it.
 *   - A box with no partner is drawn as it is. Interpolation only ever moves a
 *     pair of boxes that both exist; a box present on one side alone is never
 *     dropped, so a detection that appears or vanishes between two looks is
 *     covered for the whole span between them.
 */

struct bc_dh;

struct bc_dh *bc_dh_create(int slots);
void bc_dh_destroy(struct bc_dh *h);
void bc_dh_reset(struct bc_dh *h);

/*
 * Record that the screen as of `t` (ms) is covered by `boxes`. `t` must not go
 * backwards; an out-of-order record is dropped.
 */
void bc_dh_push(struct bc_dh *h, double t, const struct bc_box *boxes, int n);

/* The newest record's time, or a large negative number if there is none. */
double bc_dh_newest(const struct bc_dh *h);

/*
 * The boxes for the frame captured at `t`, into `out` (room for `max`).
 * Returns the count, or -1 when `t` is not certified yet -- nothing at or after
 * it has been looked at.
 *
 * `gap_ms` is the time between the two records it was built from; 0 when `t`
 * fell exactly on one. Pairs further apart than BC_DH_MAX_SPAN_MS are not
 * interpolated -- a gap that long is a screen that sat still and then changed,
 * and moving a box along it would invent a motion nobody saw -- so both sides
 * are drawn whole instead.
 */
#define BC_DH_MAX_SPAN_MS 1500.0
int bc_dh_boxes(const struct bc_dh *h, double t, struct bc_box *out, int max,
                double *gap_ms);

#endif /* BC_DELAY_H */
