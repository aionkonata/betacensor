#ifndef BC_CAPTURE_H
#define BC_CAPTURE_H

#include <stdint.h>

struct bc_gpu;
struct bc_capture;

/*
 * Connect to the portal's PipeWire remote and negotiate a DMA-BUF stream on
 * `node_id`, importing each frame into `gpu`'s texture.
 *
 * Takes ownership of pw_fd. Returns NULL on failure.
 *
 * Runs on a pw_loop rather than a pw_main_loop so the caller can drive it from
 * its own loop: the joined pipeline must pump PipeWire, run GL work and present
 * the overlay from one thread, because the GL context is only current there.
 */
struct bc_capture *bc_capture_start(int pw_fd, uint32_t node_id,
                                    struct bc_gpu *gpu);
void bc_capture_stop(struct bc_capture *c);

/*
 * Pump pending PipeWire events without blocking.
 * Returns 1 if a new frame was imported, 0 if nothing arrived, -1 on error.
 */
int bc_capture_poll(struct bc_capture *c);

/* Total frames imported since start, for rate accounting. */
uint64_t bc_capture_frames(const struct bc_capture *c);

/*
 * Phase 1 convenience: block until one frame arrives, import it, dump a PPM.
 * Returns 0 on success. Takes ownership of pw_fd.
 */
int bc_capture_one(int pw_fd, uint32_t node_id, struct bc_gpu *gpu,
                   const char *out_path);

/*
 * When the last imported frame was composited (CLOCK_MONOTONIC ns), or 0 if
 * the server sent no header. Used to tell a frame captured while the overlay
 * was blanked from one captured with covers on screen.
 */
int64_t bc_capture_frame_pts(const struct bc_capture *c);

#endif /* BC_CAPTURE_H */
