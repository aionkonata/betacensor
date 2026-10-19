#ifndef BC_SEAM_H
#define BC_SEAM_H

#include "detect.h"

/*
 * Rejoin detections that a tile boundary cut in half.
 *
 * When the screen is split into tiles, an object lying across a seam is
 * *truncated* in each tile: one tile sees its left half, the next its right.
 * Two things follow, and both are wrong on screen:
 *
 *   - Two boxes are drawn where there is one object, with a join down the
 *     middle of it.
 *   - Ordinary IoU cannot fix it. The two halves barely overlap -- only inside
 *     the shared strip -- so their IoU is *low*, and an IoU-threshold merge
 *     keeps both. IoU fails hardest exactly where it is needed.
 *
 * So the test is not IoU. The intersection is compared against the largest
 * intersection the pair *could* have had: each is only visible within the
 * shared strip, so on a split axis neither can contribute more than the strip's
 * width to the overlap. A pair that agrees as much as it possibly can inside
 * the strip scores 1, however small a fraction of either box that is.
 *
 * `d` must already be in screen coordinates, so boxes from different tiles are
 * comparable. `cols` is the grid width, used to tell which axis a pair is
 * split across; `ox`/`oy` are the shared strip widths in screen pixels from
 * bc_preproc_overlap_px(). `similar` decides whether two class ids are the same
 * physical thing -- a model that flips gender or covered/exposed across a seam
 * must still merge.
 *
 * `full` is the full-frame tile's index (bc_preproc_full_tile(), -1 if there is
 * not one). A pair involving it is a different situation from a seam: neither
 * box is truncated, they are one object seen at two scales, and the full-frame
 * tile has no row or column so the split arithmetic would be meaningless for
 * it. Such a pair is judged by plain containment instead -- how much of the
 * smaller box lies inside the larger.
 *
 * Returns the new count. Merging can only grow a box, so it never uncovers.
 */
int bc_seam_merge(struct bc_det *d, int n, int cols, int full, float ox,
                  float oy, float thresh, int (*similar)(int a, int b));

#endif /* BC_SEAM_H */
