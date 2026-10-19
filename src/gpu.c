#include "gpu.h"
#include "util.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <drm_fourcc.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Loaded at runtime; typedef locally so we don't depend on GL/glext.h. */
typedef void (*gl_egl_image_target_2d_fn)(GLenum target, void *image);

/*
 * PipeWire negotiates 8 buffers here (see capture.c) and cycles them. 16 slots
 * leaves room for a renegotiation that adds the new set before removing the
 * old, which is the only way the count ever exceeds what was asked for.
 */
#define BC_MAX_IMPORTS 16

/*
 * One buffer's import, held for as long as the producer says the buffer lives
 * rather than for one frame.
 *
 * The layout is kept alongside so a producer that re-lays-out a buffer under
 * the same identity is noticed rather than sampled through a stale EGLImage.
 */
struct bc_import {
	const void *key;            /* NULL marks a free slot */
	EGLImage    img;
	GLuint      tex;
	int         width, height;

	uint32_t    fourcc;
	uint64_t    modifier;
	int         n_planes;
	uint32_t    offset[BC_MAX_PLANES];
	uint32_t    stride[BC_MAX_PLANES];
};

struct bc_gpu {
	EGLDisplay dpy;
	EGLContext ctx;
	int        owns_display;   /* 0 when attached to somebody else's display */

	struct bc_import  imp[BC_MAX_IMPORTS];
	struct bc_import *cur;     /* points into imp[]; NULL before the first frame */

	gl_egl_image_target_2d_fn        image_target_2d;
	PFNEGLQUERYDMABUFMODIFIERSEXTPROC query_modifiers;
};

static int has_ext(const char *exts, const char *name)
{
	if (!exts)
		return 0;
	size_t n = strlen(name);
	for (const char *p = exts; (p = strstr(p, name)); p += n) {
		if ((p == exts || p[-1] == ' ') && (p[n] == ' ' || p[n] == 0))
			return 1;
	}
	return 0;
}

static EGLDisplay open_device_display(void)
{
	const char *client_exts = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);

	if (!has_ext(client_exts, "EGL_EXT_platform_device")) {
		WARN("EGL_EXT_platform_device missing; falling back to default display");
		return eglGetDisplay(EGL_DEFAULT_DISPLAY);
	}

	PFNEGLQUERYDEVICESEXTPROC query_devices =
		(PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
	PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
		(PFNEGLGETPLATFORMDISPLAYEXTPROC)
			eglGetProcAddress("eglGetPlatformDisplayEXT");

	if (!query_devices || !get_platform_display)
		return eglGetDisplay(EGL_DEFAULT_DISPLAY);

	EGLDeviceEXT devices[16];
	EGLint n = 0;
	if (!query_devices(16, devices, &n) || n <= 0) {
		WARN("eglQueryDevicesEXT found no devices");
		return eglGetDisplay(EGL_DEFAULT_DISPLAY);
	}

	LOG("EGL: %d device(s) available, using device 0", n);
	return get_platform_display(EGL_PLATFORM_DEVICE_EXT, devices[0], NULL);
}

/*
 * Extension checks and entry points that every bc_gpu needs, whoever owns the
 * display. Requires eglInitialize to have run and a context to be current.
 */
static int gpu_finish_init(struct bc_gpu *g)
{
	const char *exts = eglQueryString(g->dpy, EGL_EXTENSIONS);

	/*
	 * These two are the whole premise of §0. If either is missing on this
	 * driver, the zero-copy design does not hold and we want to know here
	 * rather than three layers deeper.
	 */
	if (!has_ext(exts, "EGL_EXT_image_dma_buf_import")) {
		ERR("EGL_EXT_image_dma_buf_import missing — zero-copy import impossible");
		return -1;
	}
	if (!has_ext(exts, "EGL_EXT_image_dma_buf_import_modifiers"))
		WARN("EGL_EXT_image_dma_buf_import_modifiers missing; "
		     "will only be able to offer DRM_FORMAT_MOD_INVALID");

	g->image_target_2d = (gl_egl_image_target_2d_fn)
		eglGetProcAddress("glEGLImageTargetTexture2DOES");
	if (!g->image_target_2d) {
		ERR("glEGLImageTargetTexture2DOES unavailable");
		return -1;
	}

	g->query_modifiers = (PFNEGLQUERYDMABUFMODIFIERSEXTPROC)
		eglGetProcAddress("eglQueryDmaBufModifiersEXT");

	/* Textures are per-import now; there is no single one to create here. */
	return 0;
}

struct bc_gpu *bc_gpu_create_shared(void *egl_display)
{
	struct bc_gpu *g = calloc(1, sizeof *g);
	if (!g)
		return NULL;

	g->dpy          = (EGLDisplay)egl_display;
	g->ctx          = EGL_NO_CONTEXT;   /* not ours; do not destroy */
	g->owns_display = 0;

	if (g->dpy == EGL_NO_DISPLAY) {
		ERR("bc_gpu_create_shared given no display");
		goto fail;
	}
	if (eglGetCurrentContext() == EGL_NO_CONTEXT) {
		ERR("bc_gpu_create_shared needs a current GL context");
		goto fail;
	}
	if (gpu_finish_init(g) < 0)
		goto fail;

	return g;
fail:
	bc_gpu_destroy(g);
	return NULL;
}

unsigned bc_gpu_texture(const struct bc_gpu *g)
{
	return (g && g->cur) ? g->cur->tex : 0;
}
int bc_gpu_width(const struct bc_gpu *g)
{
	return (g && g->cur) ? g->cur->width : 0;
}
int bc_gpu_height(const struct bc_gpu *g)
{
	return (g && g->cur) ? g->cur->height : 0;
}

struct bc_gpu *bc_gpu_create(void)
{
	struct bc_gpu *g = calloc(1, sizeof *g);
	if (!g)
		return NULL;

	g->owns_display = 1;
	g->dpy = open_device_display();
	if (g->dpy == EGL_NO_DISPLAY) {
		ERR("no EGL display");
		goto fail;
	}

	EGLint major = 0, minor = 0;
	if (!eglInitialize(g->dpy, &major, &minor)) {
		ERR("eglInitialize failed (0x%04x)", eglGetError());
		goto fail;
	}
	LOG("EGL %d.%d initialised", major, minor);

	const char *exts = eglQueryString(g->dpy, EGL_EXTENSIONS);

	if (!eglBindAPI(EGL_OPENGL_API)) {
		ERR("eglBindAPI(EGL_OPENGL_API) failed");
		goto fail;
	}

	int no_config   = has_ext(exts, "EGL_KHR_no_config_context");
	int surfaceless = has_ext(exts, "EGL_KHR_surfaceless_context");
	if (!surfaceless)
		WARN("EGL_KHR_surfaceless_context missing; makeCurrent may fail");

	EGLConfig cfg = EGL_NO_CONFIG_KHR;
	if (!no_config) {
		/* Pick any config; we never present through it in Phase 1. */
		EGLint attrs[] = {
			EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
			EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
			EGL_NONE
		};
		EGLint num = 0;
		if (!eglChooseConfig(g->dpy, attrs, &cfg, 1, &num) || num < 1) {
			ERR("eglChooseConfig found nothing usable");
			goto fail;
		}
	}

	EGLint ctx_attrs[] = {
		EGL_CONTEXT_MAJOR_VERSION, 4,
		EGL_CONTEXT_MINOR_VERSION, 3,   /* 4.3 = compute shaders, §4 */
		EGL_NONE
	};
	g->ctx = eglCreateContext(g->dpy, cfg, EGL_NO_CONTEXT, ctx_attrs);
	if (g->ctx == EGL_NO_CONTEXT) {
		ERR("eglCreateContext failed (0x%04x)", eglGetError());
		goto fail;
	}

	if (!eglMakeCurrent(g->dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, g->ctx)) {
		ERR("eglMakeCurrent failed (0x%04x)", eglGetError());
		goto fail;
	}

	LOG("GL vendor:   %s", (const char *)glGetString(GL_VENDOR));
	LOG("GL renderer: %s", (const char *)glGetString(GL_RENDERER));
	LOG("GL version:  %s", (const char *)glGetString(GL_VERSION));

	if (gpu_finish_init(g) < 0)
		goto fail;
	return g;

fail:
	bc_gpu_destroy(g);
	return NULL;
}

void bc_gpu_destroy(struct bc_gpu *g)
{
	if (!g)
		return;
	if (g->dpy != EGL_NO_DISPLAY) {
		for (int i = 0; i < BC_MAX_IMPORTS; i++) {
			if (g->imp[i].img)
				eglDestroyImage(g->dpy, g->imp[i].img);
			if (g->imp[i].tex)
				glDeleteTextures(1, &g->imp[i].tex);
		}
		/* Only tear down the display/context when we created them. */
		if (g->owns_display) {
			if (g->ctx != EGL_NO_CONTEXT)
				eglDestroyContext(g->dpy, g->ctx);
			eglTerminate(g->dpy);
		}
	}
	free(g);
}

int bc_gpu_query_modifiers(struct bc_gpu *g, uint32_t fourcc,
                           uint64_t *out, int max)
{
	if (!g->query_modifiers || max <= 0)
		return 0;

	EGLint n = 0;
	if (!g->query_modifiers(g->dpy, (EGLint)fourcc, 0, NULL, NULL, &n) || n <= 0)
		return 0;
	if (n > max)
		n = max;

	EGLBoolean *external = calloc((size_t)n, sizeof *external);
	if (!external)
		return 0;

	if (!g->query_modifiers(g->dpy, (EGLint)fourcc, n,
	                        (EGLuint64KHR *)out, external, &n))
		n = 0;

	free(external);
	return n;
}

static int clamp_planes(int n)
{
	return n > BC_MAX_PLANES ? BC_MAX_PLANES : n;
}

/* Does a cached import still describe the buffer the producer just handed us? */
static int layout_matches(const struct bc_import *s, const struct bc_dmabuf *b)
{
	if (s->width != b->width || s->height != b->height ||
	    s->fourcc != b->fourcc || s->modifier != b->modifier ||
	    s->n_planes != clamp_planes(b->n_planes))
		return 0;
	for (int p = 0; p < s->n_planes; p++)
		if (s->offset[p] != b->offset[p] || s->stride[p] != b->stride[p])
			return 0;
	return 1;
}

static void drop_import(struct bc_gpu *g, struct bc_import *s)
{
	if (g->cur == s)
		g->cur = NULL;
	if (s->img)
		eglDestroyImage(g->dpy, s->img);
	if (s->tex)
		glDeleteTextures(1, &s->tex);
	memset(s, 0, sizeof *s);
}

void bc_gpu_forget(struct bc_gpu *g, const void *key)
{
	if (!g || !key)
		return;
	for (int i = 0; i < BC_MAX_IMPORTS; i++)
		if (g->imp[i].key == key)
			drop_import(g, &g->imp[i]);
}

int bc_gpu_import(struct bc_gpu *g, const void *key, const struct bc_dmabuf *b)
{
	struct bc_import *slot = NULL, *spare = NULL;

	/* The key is what gives a slot its identity; without one every frame
	 * would claim a fresh slot and the cache would leak them. */
	if (!g || !key || !b)
		return -1;

	for (int i = 0; i < BC_MAX_IMPORTS; i++) {
		if (g->imp[i].key == key) {
			slot = &g->imp[i];
			break;
		}
		if (!g->imp[i].key && !spare)
			spare = &g->imp[i];
	}

	/*
	 * Seen before and unchanged: point at it and stop. This is the common
	 * case -- the producer cycles a handful of buffers for the whole run --
	 * so the steady state costs no EGL calls at all.
	 */
	if (slot && layout_matches(slot, b)) {
		g->cur = slot;
		return 0;
	}
	if (slot)
		drop_import(g, slot);   /* same buffer, new layout: redo it */
	else
		slot = spare;

	/*
	 * Full. Only reachable if the producer replaced its buffer set without
	 * ever taking the old one back, which it is not supposed to do -- but
	 * failing here would latch the capture dead for the rest of the run
	 * (bc_main.c treats a failed import as a lost stream), and a stale
	 * cache entry is not worth that. Evict anything that is not the frame
	 * being read and carry on.
	 */
	if (!slot) {
		for (int i = 0; i < BC_MAX_IMPORTS; i++) {
			if (&g->imp[i] != g->cur) {
				WARN("import cache full; evicting a stale buffer");
				drop_import(g, &g->imp[i]);
				slot = &g->imp[i];
				break;
			}
		}
	}
	if (!slot)
		return -1;

	EGLAttrib a[64];
	int i = 0;

	a[i++] = EGL_WIDTH;                   a[i++] = b->width;
	a[i++] = EGL_HEIGHT;                  a[i++] = b->height;
	a[i++] = EGL_LINUX_DRM_FOURCC_EXT;    a[i++] = (EGLAttrib)b->fourcc;

	/* Plane attribute tokens are contiguous per plane, but spelling them
	 * out is clearer than pointer arithmetic on enum values. */
	static const EGLAttrib fd_tok[4] = {
		EGL_DMA_BUF_PLANE0_FD_EXT, EGL_DMA_BUF_PLANE1_FD_EXT,
		EGL_DMA_BUF_PLANE2_FD_EXT, EGL_DMA_BUF_PLANE3_FD_EXT
	};
	static const EGLAttrib off_tok[4] = {
		EGL_DMA_BUF_PLANE0_OFFSET_EXT, EGL_DMA_BUF_PLANE1_OFFSET_EXT,
		EGL_DMA_BUF_PLANE2_OFFSET_EXT, EGL_DMA_BUF_PLANE3_OFFSET_EXT
	};
	static const EGLAttrib pitch_tok[4] = {
		EGL_DMA_BUF_PLANE0_PITCH_EXT, EGL_DMA_BUF_PLANE1_PITCH_EXT,
		EGL_DMA_BUF_PLANE2_PITCH_EXT, EGL_DMA_BUF_PLANE3_PITCH_EXT
	};
	static const EGLAttrib mod_lo_tok[4] = {
		EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT,
		EGL_DMA_BUF_PLANE2_MODIFIER_LO_EXT, EGL_DMA_BUF_PLANE3_MODIFIER_LO_EXT
	};
	static const EGLAttrib mod_hi_tok[4] = {
		EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT,
		EGL_DMA_BUF_PLANE2_MODIFIER_HI_EXT, EGL_DMA_BUF_PLANE3_MODIFIER_HI_EXT
	};

	int planes = b->n_planes > BC_MAX_PLANES ? BC_MAX_PLANES : b->n_planes;
	for (int p = 0; p < planes; p++) {
		a[i++] = fd_tok[p];    a[i++] = b->fd[p];
		a[i++] = off_tok[p];   a[i++] = b->offset[p];
		a[i++] = pitch_tok[p]; a[i++] = b->stride[p];

		/*
		 * Passing the modifier is not optional on NVIDIA — buffers come
		 * back tiled/compressed, and omitting it yields garbled output
		 * rather than an error. DESIGN.md §3[2].
		 */
		if (b->modifier != DRM_FORMAT_MOD_INVALID) {
			a[i++] = mod_lo_tok[p];
			a[i++] = (EGLAttrib)(b->modifier & 0xffffffffu);
			a[i++] = mod_hi_tok[p];
			a[i++] = (EGLAttrib)(b->modifier >> 32);
		}
	}
	a[i++] = EGL_NONE;

	EGLImage img = eglCreateImage(g->dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT,
	                              (EGLClientBuffer)NULL, a);
	if (img == EGL_NO_IMAGE) {
		ERR("eglCreateImage failed (0x%04x) — fourcc 0x%08x mod 0x%016llx "
		    "planes %d", eglGetError(), b->fourcc,
		    (unsigned long long)b->modifier, planes);
		return -1;
	}

	/*
	 * Clear anything the renderer left behind first. GL errors are sticky:
	 * glGetError reports the oldest one recorded since it was last called,
	 * not the result of the call in front of it. Without this drain, a bad
	 * argument anywhere else in the frame surfaced here as a failed import,
	 * which is read as the capture stream dying -- and that is terminal.
	 * The import is the wrong place to learn about somebody else's mistake,
	 * so say so out loud and carry on rather than failing the frame.
	 */
	for (GLenum stale; (stale = glGetError()) != GL_NO_ERROR; )
		ERR("GL error 0x%04x raised before the DMA-BUF import — not a "
		    "capture fault; look at the last thing drawn", stale);

	GLuint tex = 0;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	g->image_target_2d(GL_TEXTURE_2D, (void *)img);

	GLenum e = glGetError();
	if (e != GL_NO_ERROR) {
		ERR("glEGLImageTargetTexture2DOES failed (0x%04x)", e);
		glDeleteTextures(1, &tex);
		eglDestroyImage(g->dpy, img);
		return -1;
	}

	slot->key      = key;
	slot->img      = img;
	slot->tex      = tex;
	slot->width    = b->width;
	slot->height   = b->height;
	slot->fourcc   = b->fourcc;
	slot->modifier = b->modifier;
	slot->n_planes = planes;
	for (int p = 0; p < planes; p++) {
		slot->offset[p] = b->offset[p];
		slot->stride[p] = b->stride[p];
	}
	g->cur = slot;
	return 0;
}

int bc_gpu_dump_ppm(struct bc_gpu *g, const char *path)
{
	if (!g->cur) {
		ERR("nothing imported to dump");
		return -1;
	}

	const int w = g->cur->width, h = g->cur->height;
	size_t px = (size_t)w * (size_t)h;
	unsigned char *rgba = malloc(px * 4);
	if (!rgba)
		return -1;

	glBindTexture(GL_TEXTURE_2D, g->cur->tex);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);

	GLenum e = glGetError();
	if (e != GL_NO_ERROR) {
		ERR("glGetTexImage failed (0x%04x)", e);
		free(rgba);
		return -1;
	}

	FILE *f = fopen(path, "wb");
	if (!f) {
		ERR("cannot open %s", path);
		free(rgba);
		return -1;
	}

	fprintf(f, "P6\n%d %d\n255\n", w, h);
	/* GL hands back components already in RGBA order regardless of the
	 * underlying DRM fourcc, so no swizzle is needed here. */
	for (size_t p = 0; p < px; p++)
		fwrite(&rgba[p * 4], 1, 3, f);

	fclose(f);
	free(rgba);

	/* Cheap sanity signal: an all-black dump usually means the modifier
	 * was wrong or the frame was not yet rendered. */
	return 0;
}
