#ifndef BC_FRAMERING_H
#define BC_FRAMERING_H

/*
 * A short history of captured frames, kept on the GPU (DESIGN.md §0 -- these are
 * texture-to-texture copies; nothing crosses PCIe).
 *
 * This exists to serve the screen-latency design: rather than predicting where
 * content moved between capture and display, we *display the frame the boxes
 * were computed from*. The censor bar then sits on exactly the pixels the
 * detector saw, which is exact where a tracker is an estimate.
 *
 * The capture texture cannot be retained directly -- it aliases a DMA-BUF the
 * compositor owns and recycles under us -- so each frame is copied into a ring
 * slot that we own.
 */

struct bc_framering;

/*
 * `n` slots of w*h RGBA8. Cost is n*w*h*4 of VRAM: at 2560x1440 that is 14 MB a
 * slot, so keep n small. Returns NULL on failure (reason is logged).
 */
struct bc_framering *bc_framering_create(int w, int h, int n);
void bc_framering_destroy(struct bc_framering *r);

/*
 * Copy `src` into the ring under sequence number `seq`, captured at `t` ms,
 * evicting the oldest.
 * Returns 0 on success.
 */
int bc_framering_push(struct bc_framering *r, unsigned src, unsigned long seq,
                      double t);

/*
 * The texture holding frame `seq`, or 0 if it has been evicted or never
 * existed. Callers must handle 0 rather than assume a hit: the ring is
 * deliberately small and a slow detector will outrun it.
 */
unsigned bc_framering_get(const struct bc_framering *r, unsigned long seq);

/*
 * The newest frame captured at or before `t_max` (ms, the caller's clock), or 0
 * if there is none. Fills `seq` and `t` with that frame's number and capture
 * time when given. This is how the delayed screen picks what to show: the frame
 * from `delay` ago, whatever the capture rate happened to be.
 */
unsigned bc_framering_at(const struct bc_framering *r, double t_max,
                         unsigned long *seq, double *t);

/*
 * Verify the most recent push reproduced the source faithfully, by comparing a
 * single pixel. Returns 0 if they match, -1 otherwise, and logs the channel
 * mapping it saw.
 *
 * This exists because the obvious implementation of the copy -- and the first
 * one written -- was glCopyImageSubData, which moves raw texels and does *not*
 * convert formats: the capture texture is BGRA-in-memory, so every frame came
 * out with red and blue swapped. That is invisible from the counters and only
 * shows up as a blue screen, so it gets an assertion rather than trust.
 *
 * Reads two pixels back to the CPU, so debug only; nothing is stored.
 */
int bc_framering_selftest(const struct bc_framering *r, unsigned src);

/* Slot count, and how many frames are currently held. */
int bc_framering_capacity(const struct bc_framering *r);
int bc_framering_count(const struct bc_framering *r);

#endif /* BC_FRAMERING_H */
