#ifndef BC_GPU_H
#define BC_GPU_H

#include <stdint.h>

#define BC_MAX_PLANES 4

/* One captured DMA-BUF, as handed to us by PipeWire. */
struct bc_dmabuf {
	int      n_planes;
	int      fd[BC_MAX_PLANES];
	uint32_t offset[BC_MAX_PLANES];
	uint32_t stride[BC_MAX_PLANES];
	uint64_t modifier;      /* DRM format modifier, or DRM_FORMAT_MOD_INVALID */
	uint32_t fourcc;        /* DRM fourcc */
	int      width;
	int      height;
};

struct bc_gpu;

/* Headless EGL + GL context. Returns NULL on failure (reason is logged). */
struct bc_gpu *bc_gpu_create(void);

/*
 * Attach to an EGLDisplay somebody else owns, with their context already
 * current. Used by the joined pipeline: EGLImages and textures cannot cross
 * EGLDisplays, so capture must import onto the *same* display the overlay
 * renders with, not the headless one bc_gpu_create() opens.
 *
 * Does not create or destroy the display or the context.
 */
struct bc_gpu *bc_gpu_create_shared(void *egl_display);
void           bc_gpu_destroy(struct bc_gpu *g);

/* The texture the last import aliases, and its dimensions. 0 if none yet. */
unsigned bc_gpu_texture(const struct bc_gpu *g);
int      bc_gpu_width(const struct bc_gpu *g);
int      bc_gpu_height(const struct bc_gpu *g);

/*
 * Modifiers this driver will accept for `fourcc`, for offering to PipeWire
 * during format negotiation. Returns the count written, or negative on error.
 */
int bc_gpu_query_modifiers(struct bc_gpu *g, uint32_t fourcc,
                           uint64_t *out, int max);

/*
 * Import a DMA-BUF as a GL texture and make it the current source. Zero-copy:
 * this aliases the VRAM the compositor already allocated (DESIGN.md §0).
 * Returns 0 on success.
 *
 * `key` identifies the underlying buffer -- PipeWire hands the same small set
 * of buffers round in rotation, so the EGLImage and texture are cached against
 * it and a repeat sighting costs no EGL calls at all. That is not only cheaper
 * (an eglCreateImage/eglDestroyImage pair per frame at 180 Hz is real driver
 * work); it is what makes the texture's lifetime match the buffer's instead of
 * the frame's. Pass the key back to bc_gpu_forget() when the buffer goes away.
 *
 * The import is re-done if the producer changes the layout under a key.
 */
int bc_gpu_import(struct bc_gpu *g, const void *key, const struct bc_dmabuf *buf);

/*
 * Drop the cached import for `key`. Call when the producer takes the buffer
 * back: the EGLImage aliases memory that is about to stop being ours to read.
 * If it was the current source, there is no current source afterwards and
 * bc_gpu_texture() returns 0 until the next import.
 */
void bc_gpu_forget(struct bc_gpu *g, const void *key);

/*
 * PHASE 1 VERIFICATION ONLY.
 *
 * Reads the imported texture back to the CPU and writes a PPM. This is a
 * deliberate violation of §0 — it exists to prove the import produced correct
 * pixels, and must not survive into the steady-state pipeline.
 */
int bc_gpu_dump_ppm(struct bc_gpu *g, const char *path);

#endif /* BC_GPU_H */
