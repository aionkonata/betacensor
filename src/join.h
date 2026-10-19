#ifndef BC_JOIN_H
#define BC_JOIN_H

#include "overlay.h"

/*
 * Area joining (Pure Vision's CensorAreaJoiningConfig, PUREVISION.md).
 *
 * Overlapping covers are merged into one region instead of being drawn on top
 * of each other. With more than a few classes switched on, detections overlap
 * constantly -- a torso, a chest and a belly are three boxes over one body --
 * and drawing all three stacks edges, seams and effect boundaries inside a
 * region that is conceptually one thing. The GUI already admits this about the
 * tape effect: it "looks best with only a few classes switched on -- with most
 * of them enabled the bars overlap into a mish-mash".
 *
 * Merging is to the union rectangle, so a joined region always covers at least
 * what the boxes it replaced covered. It can never uncover anything, which is
 * what makes this safe to apply unconditionally under §2.
 *
 * That same property is why it needs a bound. Union is transitive and greedy:
 * left alone, a chain of barely-touching boxes joins end to end and the result
 * is a rectangle spanning most of the screen -- covering, but useless, and the
 * exact failure the user reported from a different cause. `waste` bounds it:
 * a join is only taken when the union does not add more than that fraction of
 * new area over what the two boxes already covered between them.
 *
 * Returns the new box count. Boxes are rewritten in place.
 *
 * `gap`   px of slack; boxes this close but not quite touching still join.
 * `waste` maximum fraction of *new* area a join may add. 0.35 means a join
 *         may cover at most 35% more than the two boxes did together.
 *         Larger merges more eagerly; 0 joins only where the union adds
 *         nothing at all.
 */
int bc_join_boxes(struct bc_box *b, int n, int gap, float waste);

#endif /* BC_JOIN_H */
