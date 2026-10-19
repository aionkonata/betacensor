#ifndef BC_PREPROCESS_H
#define BC_PREPROCESS_H

#include <stddef.h>

/*
 * Turns the captured screen texture into the tensor the detector wants, on the
 * GPU (DESIGN.md §3[4]).
 *
 * A compute shader letterboxes 2560x1440 down to the network's input size,
 * converts to planar float NCHW in [0,1], and writes it straight into an SSBO.
 * The full frame is never read back -- only the (much smaller) tensor is, and
 * only because the CPU execution provider needs it in host memory. With a GPU
 * execution provider the SSBO can be handed over directly and §0 holds
 * completely.
 */

/*
 * Most tiles per axis, and so the most inferences a frame can cost. The +1 is
 * the optional full-frame tile, which is not part of the grid and sits after
 * it -- without it a 4x4 grid asking for one would overrun every per-tile
 * array here.
 */
#define BC_MAX_GRID  4
#define BC_MAX_TILES (BC_MAX_GRID * BC_MAX_GRID + 1)

struct bc_preproc;

/*
 * Split the source into a cols x rows grid of tiles and preprocess each one
 * into its own tensor, laid out consecutively in the output buffer.
 *
 * Why: a single tile has to fit the whole screen into the network's input, so
 * everything is scaled down by whichever axis is worst and small things stop
 * being resolvable. A grid scales down less, so the detector sees more detail,
 * at the cost of one inference per tile. 1x1 is the plain letterbox and the
 * default -- this is off unless asked for, because inference is already the
 * scarcest thing in the loop.
 *
 * `overlap` is the fraction of a tile shared with its neighbour. It is not
 * optional dressing: a detection that straddles a seam is cut in half by it,
 * and the overlap is what gives each half enough context to still be found.
 * See bc_preproc_overlap_px() and the seam merge that consumes it.
 *
 * `full_frame` adds one more tile after the grid holding the whole screen at
 * the plain letterbox scale. The grid trades large objects for small ones: an
 * object filling the screen is split across every tile, and each fragment on
 * its own no longer looks like the object, so all of them score under the floor
 * and nothing is reported at all. The overlap cannot help -- the object is
 * wider than the strip. Only a view that still contains the whole thing can
 * find it, which is what this tile is. Costs one further inference per frame.
 */
int    bc_preproc_set_grid(struct bc_preproc *p, int cols, int rows,
                           float overlap, int full_frame);
int    bc_preproc_tiles(const struct bc_preproc *p);

/*
 * Replace the grid with `n` arbitrary source rectangles, one tile each, given
 * as x,y,w,h in source pixels. Each is fitted into the network input keeping
 * aspect, exactly as the grid letterboxes.
 *
 * This is what a top-down model needs: it runs on a crop of one object rather
 * than on the frame. A tile was only ever a scale and an offset, so a crop is
 * the same transform with different numbers, and bc_preproc_tile_to_screen()
 * inverts it unchanged.
 *
 * Rectangles survive a source-size change, unlike a grid, because the caller
 * derived them from detections at a particular size and only the caller can
 * say what they should become.
 */
int    bc_preproc_set_rects(struct bc_preproc *p, const float *rects, int n);

/*
 * Per-channel normalisation applied after the 0..1 fetch: (c - mean) / std.
 *
 * Defaults to identity, which is the plain 0..1 every YOLO export here wants.
 * Pass ImageNet's figures for a torchvision-derived model -- getting this wrong
 * does not fail, it just moves every prediction.
 */
void   bc_preproc_set_norm(struct bc_preproc *p, const float mean[3],
                           const float std[3]);

/*
 * Index of the full-frame tile, or -1 when there is not one. The seam merge
 * needs it: this tile has no position in the grid, so the row/column arithmetic
 * that decides whether two detections straddle a seam is meaningless for it.
 */
int    bc_preproc_full_tile(const struct bc_preproc *p);

/* Floats across every tile: what the output buffer must hold. */
size_t bc_preproc_total_floats(const struct bc_preproc *p);

/* Map network coordinates from a given tile back to screen pixels. */
void bc_preproc_tile_to_screen(const struct bc_preproc *p, int tile,
                               float nx, float ny, float *sx, float *sy);

/* Width of the shared strip between neighbouring tiles, in screen pixels;
 * 0 on an axis that is not split. */
void bc_preproc_overlap_px(const struct bc_preproc *p, float *ox, float *oy);

/*
 * Fraction of the tensor that differs from the previous frame's by more than
 * `eps`, and remember this frame as the new previous. Must follow a
 * bc_preproc_run().
 *
 * Runs entirely on the GPU: only the resulting count crosses the bus, four
 * bytes instead of the whole tensor. The caller used to answer this by reading
 * the tensor back and looping over it on the CPU, which is why the readback
 * happened on *every* frame rather than only on the ones actually inferred --
 * at a 3x2 grid, 29 MB a frame to discover that nothing had moved.
 *
 * Returns 1.0 ("all of it") when there is no previous frame yet and on any GL
 * error, so an unanswerable question always sends the caller to inference
 * rather than letting it gate.
 */
float bc_preproc_change_frac(struct bc_preproc *p, float eps);

/* net_w/net_h must be multiples of 32 (YOLO stride). */
struct bc_preproc *bc_preproc_create(int net_w, int net_h);
void               bc_preproc_destroy(struct bc_preproc *p);

/* Number of floats in the output tensor (3 * net_w * net_h). */
size_t bc_preproc_floats(const struct bc_preproc *p);

/*
 * Run the shader over `tex` (a GL_TEXTURE_2D of src_w x src_h).
 * Returns 0 on success. Does not block on completion.
 */
int bc_preproc_run(struct bc_preproc *p, unsigned tex, int src_w, int src_h);

/*
 * Copy the tensor to host memory. Blocks until the shader has finished, so
 * this is the stage that costs wall-clock time; `dst` must hold
 * bc_preproc_floats() floats.
 */
int bc_preproc_readback(struct bc_preproc *p, float *dst);

/*
 * Adopt this frame as the one the gate compares the next against.
 *
 * Split out of bc_preproc_change_frac() so the reference can be the last frame
 * actually *inferred* rather than the last one merely looked at. Comparing
 * consecutive frames means a screen that drifts slowly never trips the
 * threshold on any single step, so the gate can report "unchanged" for as long
 * as the drift lasts while the picture is replaced entirely underneath it. The
 * question worth asking is whether the screen still matches the frame the boxes
 * currently on it were computed from, which is this.
 *
 * The direction of the change matters: holding the reference further back can
 * only make the gate fire more often, never less, so the failure mode is a
 * wasted inference rather than an uncensored screen.
 */
void bc_preproc_commit(struct bc_preproc *p);

/*
 * Forget the reference frame, so the next bc_preproc_change_frac() answers "all
 * of it". For when a committed frame turns out not to have been inferred after
 * all: an unmeasured reference would gate against something no box came from.
 */
void bc_preproc_gate_reset(struct bc_preproc *p);

/*
 * Zero-copy handoff to a CUDA execution provider (DESIGN.md §0).
 *
 * bc_preproc_readback() is the one place the tensor crosses PCIe, and with the
 * CUDA EP it crosses straight back: ORT copies the host tensor into VRAM again
 * for a buffer that never needed to leave. At a 3x2 grid plus the full-frame
 * tile that is ~20 MB each way per inferred frame, to hand the GPU something it
 * already had.
 *
 * Registering the SSBO with CUDA lets ORT read it in place. Returns 0 if the
 * buffer is now usable that way, non-zero otherwise -- there is no failure to
 * handle beyond staying on the readback path, which still works.
 *
 * Ownership, which is the whole difficulty: while the buffer is mapped it
 * belongs to CUDA, and *any* GL access to it -- the next dispatch, the gate's
 * copy into `prev`, a readback -- is undefined. The memcpy in bc_detect_submit()
 * used to keep the inference isolated from the frames that followed it; without
 * it the caller must not preprocess again until the inference reading the
 * buffer has finished. bc_preproc_gpu_release() hands it back, and must be
 * called at a point where that is known -- see the bc_detect_busy() check in
 * bc_main.c, which is where it is known.
 */
int  bc_preproc_gpu_register(struct bc_preproc *p);

/*
 * Is the tensor buffer actually registered? Registration is best-effort, so
 * this is not the same question as "was it asked for": both ends have to agree
 * before the caller can skip the readback.
 */
int  bc_preproc_zero_copy(const struct bc_preproc *p);

/*
 * Map the tensor for CUDA and return a device pointer to tile 0, or NULL. The
 * tiles follow it consecutively, bc_preproc_floats() apart, exactly as in the
 * host buffer. Blocks until the shader has finished, as the readback does.
 */
const void *bc_preproc_device_ptr(struct bc_preproc *p);

/* Give the buffer back to GL. Safe to call when it was never mapped. */
void bc_preproc_gpu_release(struct bc_preproc *p);

/*
 * Undo the letterbox: map a point in network space back to screen pixels.
 * Detections come out in network coordinates and have to be placed on screen.
 */
void bc_preproc_to_screen(const struct bc_preproc *p,
                          float nx, float ny, float *sx, float *sy);

/* Forward direction: screen pixels -> network coordinates. */
void bc_preproc_to_net(const struct bc_preproc *p,
                       float sx, float sy, float *nx, float *ny);

/* Network input dimensions. */
int bc_preproc_net_w(const struct bc_preproc *p);
int bc_preproc_net_h(const struct bc_preproc *p);

/*
 * Scale factor applied to source pixels (for converting box sizes). Tiles no
 * longer share one: this is the grid's, which every tile but the full-frame one
 * uses. Anything working in a particular tile's coordinates wants
 * bc_preproc_tile_to_screen() instead.
 */
float bc_preproc_scale(const struct bc_preproc *p);

/*
 * DEBUG ONLY. Write a readback tensor out as a PPM so the letterbox, channel
 * order and normalisation can be checked against the screen it came from.
 * Undoes the planar NCHW layout, so it also verifies that.
 */
int bc_preproc_dump_ppm(const struct bc_preproc *p, const float *tensor,
                        const char *path);

#endif /* BC_PREPROCESS_H */
