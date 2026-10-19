#ifndef BC_HOLD_H
#define BC_HOLD_H

/*
 * Region stability -- "is the picture under this cover the same as it was?"
 *
 * The question exists because of a loop the program cannot otherwise escape on
 * a compositor that will not hide the overlay from screen capture (DESIGN.md
 * §3[7]): the detector sees our own censor bar, finds no nudity under it,
 * drops the box, the content is exposed again, and the bar comes back. Left
 * alone that oscillates at the detection rate and periodically *reveals* what
 * the program exists to hide.
 *
 * The fix is not to trust the absence:
 *
 *   Absence of detection inside a region we are actively covering is not
 *   evidence of absence. We know exactly where we drew, so we know exactly
 *   where we are blind.
 *
 * So each inference, the region under every live box is compared against the
 * same region of the previous inferred frame. Unchanged means we are looking
 * at our own bar, have learned nothing, and the box is held rather than aged.
 * Only once the content underneath actually moves does the sticky timer get to
 * expire it.
 *
 * Its own translation unit because it is pure arithmetic over two tensors, and
 * that is the part worth testing without a compositor, a GPU or a screen.
 */

/*
 * Mean absolute difference between the same rectangle of two preprocessed
 * tensors, over every channel.
 *
 * Tensors are planar (NCHW): `ch` planes of w*h, which is the layout the
 * preprocess shader writes. Coordinates are in tensor pixels and are clamped
 * to the tensor, so a box that hangs off the edge measures the part that is
 * inside rather than reading past the buffer.
 *
 * Returns a value in the same units as the tensor (normalised 0..1 here), or
 * -1.0f when there is no overlap to measure -- which is deliberately *not*
 * zero. Zero means "identical", and identical is what causes a hold; an
 * unanswerable question must never hold a box, or a box off the edge of the
 * screen would become permanent.
 */
float bc_hold_region_mad(const float *cur, const float *prev,
                         int w, int h, int ch,
                         int x0, int y0, int x1, int y1);

#endif /* BC_HOLD_H */
