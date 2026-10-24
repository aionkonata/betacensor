#include "preprocess.h"
#include "util.h"

#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>

#include <stdlib.h>
#include <string.h>

/*
 * CUDA is optional at build time: the toolkit is not a dependency of the
 * project, and without it every path below falls back to the readback that has
 * always been there. See the CUDA_* block in the Makefile.
 */
#ifdef BC_HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif

struct bc_preproc {
	int    net_w, net_h;
	int    src_w, src_h;

	/*
	 * Tiling. 1x1 is the whole screen in one tile, which is the original
	 * behaviour exactly: the scale formula and the centring below both
	 * reduce to the plain letterbox at cols = rows = 1.
	 *
	 * `tiles` counts the full-frame tile, `grid` does not, and the extra
	 * tile is always the last one. Scale is per tile because that tile sees
	 * the screen at a different one -- at 3x2 on 2560x1440 the grid runs at
	 * 0.5067 and the full frame at 0.125.
	 */
	int    cols, rows, grid, tiles;
	int    full;           /* is the last tile the full frame? */
	/* Tiles came from bc_preproc_set_rects() and are arbitrary crops, not a
	 * grid. layout() must not recompute them when the source size changes:
	 * the caller owns them, and re-deriving a grid would silently replace
	 * every face box with a screen tile. */
	int    rect_mode;
	float  overlap;        /* fraction of a tile shared with its neighbour */
	float  scale[BC_MAX_TILES];   /* source pixels -> network pixels */
	float  pad_x[BC_MAX_TILES], pad_y[BC_MAX_TILES];

	GLuint prog;
	GLuint ssbo;
	size_t floats;         /* per tile */

	GLint  u_net, u_src_size, u_scale, u_pad, u_base;
	GLint  u_mean, u_invstd;
	/* Normalisation applied after the 0..1 fetch. Identity by default, so
	 * every existing caller preprocesses exactly as before. */
	float  mean[3], invstd[3];

	/*
	 * Change detection, kept on the GPU.
	 *
	 * The gate only needs one number -- how much of the frame differs from
	 * the last one -- and computing it on the CPU meant dragging the whole
	 * tensor across PCIe every frame to answer it. Measured at a 3x2 grid,
	 * 668 readbacks of 29 MB each supported 13 inferences. The comparison
	 * is a reduction, so it belongs where the data already is: `prev` holds
	 * the last frame in VRAM and `cnt` is the four bytes that come back.
	 */
	GLuint diff_prog;
	GLuint prev;
	GLuint cnt;
	int    has_prev;
	GLint  u_n, u_eps;

	/*
	 * The SSBO as CUDA sees it. `dev` is non-NULL exactly while the buffer
	 * is mapped, which is exactly while GL must not touch it.
	 */
	int    want_cuda;
	int    cuda_failed;    /* asked once, refused; do not ask again per frame */
#ifdef BC_HAVE_CUDA
	struct cudaGraphicsResource *res;
	void  *dev;
#endif
};

/*
 * Registration is against the buffer *name*, and re-specifying its storage
 * underneath CUDA is not allowed -- so a grid change, which reallocates, has to
 * drop the registration and take it again at the new size. Kept here so
 * alloc_ssbo() can bracket itself with the two.
 */
static void cuda_unregister(struct bc_preproc *p)
{
#ifdef BC_HAVE_CUDA
	if (p->dev) {
		cudaGraphicsUnmapResources(1, &p->res, 0);
		p->dev = NULL;
	}
	if (p->res) {
		cudaGraphicsUnregisterResource(p->res);
		p->res = NULL;
	}
#else
	(void)p;
#endif
}

static int cuda_register(struct bc_preproc *p)
{
#ifdef BC_HAVE_CUDA
	if (!p->want_cuda || p->res)
		return p->res ? 0 : -1;
	/*
	 * cudaGraphicsRegisterFlagsReadOnly: the tensor is written by GL and
	 * only ever read by CUDA. Saying so lets the driver skip preserving
	 * anything CUDA might have written back, and makes a stray write from
	 * the inference side an error rather than silent corruption of the next
	 * frame's input.
	 */
	const cudaError_t e = cudaGraphicsGLRegisterBuffer(
		&p->res, p->ssbo, cudaGraphicsRegisterFlagsReadOnly);
	if (e != cudaSuccess) {
		ERR("CUDA cannot see the tensor buffer (%s) -- staying on the "
		    "readback path", cudaGetErrorString(e));
		p->res = NULL;
		p->want_cuda = 0;
		return -1;
	}
	return 0;
#else
	(void)p;
	return -1;
#endif
}

/*
 * Letterbox + normalise + planar-ise in one pass.
 *
 * Writes NCHW because that is what the YOLO export expects; doing the
 * transpose here costs nothing (it is just the output index) whereas doing it
 * on the CPU after readback would be a per-element pass over ~184k floats.
 *
 * The 114/255 fill matches ultralytics' letterbox grey. Using black instead
 * would put a hard edge at the padding boundary that the network has never
 * seen in training.
 */
static const char *CS_SRC =
"#version 430\n"
"layout(local_size_x = 8, local_size_y = 8) in;\n"
"layout(binding = 0) uniform sampler2D uSrc;\n"
"layout(std430, binding = 0) buffer Tensor { float t[]; };\n"
"uniform ivec2 uNet;\n"
"uniform vec2  uSrcSize;\n"
"uniform float uScale;\n"
"uniform vec2  uPad;\n"
"uniform int   uBase;   /* first float of this tile's slice of the SSBO */\n"
"uniform vec3  uMean;   /* per-channel, subtracted after the 0..1 fetch */\n"
"uniform vec3  uInvStd; /* reciprocal, so the shader multiplies */\n"
"void main() {\n"
"    ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
"    if (p.x >= uNet.x || p.y >= uNet.y) return;\n"
"    vec2 sp = (vec2(p) + 0.5 - uPad) / uScale;\n"
"    vec3 c;\n"
"    if (sp.x < 0.0 || sp.y < 0.0 || sp.x >= uSrcSize.x || sp.y >= uSrcSize.y)\n"
"        c = vec3(114.0 / 255.0);\n"
"    else\n"
"        c = texture(uSrc, sp / uSrcSize).rgb;\n"
"    c = (c - uMean) * uInvStd;\n"
"    int n = uNet.x * uNet.y;\n"
"    int i = uBase + p.y * uNet.x + p.x;\n"
"    t[i]         = c.r;\n"
"    t[n + i]     = c.g;\n"
"    t[2 * n + i] = c.b;\n"
"}\n";

/*
 * How many elements differ from the previous frame, by more than uEps.
 *
 * Counted per workgroup in shared memory and added to the global total once,
 * rather than one atomic per differing element: on a frame that changed
 * completely that would be millions of atomics on one address, which is slower
 * than the readback this replaces.
 */
static const char *CS_DIFF_SRC =
"#version 430\n"
"layout(local_size_x = 256) in;\n"
"layout(std430, binding = 0) readonly buffer Cur  { float cur[]; };\n"
"layout(std430, binding = 1) readonly buffer Prev { float prv[]; };\n"
"layout(std430, binding = 2) buffer Count { uint total; };\n"
"uniform uint  uN;\n"
"uniform float uEps;\n"
"shared uint local_count;\n"
"void main() {\n"
"    if (gl_LocalInvocationIndex == 0u) local_count = 0u;\n"
"    barrier();\n"
"    uint i = gl_GlobalInvocationID.x;\n"
"    if (i < uN && abs(cur[i] - prv[i]) > uEps)\n"
"        atomicAdd(local_count, 1u);\n"
"    barrier();\n"
"    if (gl_LocalInvocationIndex == 0u && local_count > 0u)\n"
"        atomicAdd(total, local_count);\n"
"}\n";

static GLuint compile_compute(const char *src)
{
	GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(sh, 1, &src, NULL);
	glCompileShader(sh);

	GLint ok = 0;
	glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[2048] = { 0 };
		glGetShaderInfoLog(sh, sizeof log - 1, NULL, log);
		ERR("compute shader compile failed:\n%s", log);
		glDeleteShader(sh);
		return 0;
	}

	GLuint prog = glCreateProgram();
	glAttachShader(prog, sh);
	glLinkProgram(prog);
	glDeleteShader(sh);

	glGetProgramiv(prog, GL_LINK_STATUS, &ok);
	if (!ok) {
		char log[2048] = { 0 };
		glGetProgramInfoLog(prog, sizeof log - 1, NULL, log);
		ERR("compute program link failed:\n%s", log);
		glDeleteProgram(prog);
		return 0;
	}
	return prog;
}

static int alloc_ssbo(struct bc_preproc *p)
{
	const size_t total = p->floats * (size_t)p->tiles;

	cuda_unregister(p);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, p->ssbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER,
	             (GLsizeiptr)(total * sizeof(float)), NULL, GL_DYNAMIC_READ);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, p->prev);
	glBufferData(GL_SHADER_STORAGE_BUFFER,
	             (GLsizeiptr)(total * sizeof(float)), NULL, GL_DYNAMIC_COPY);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	/* Whatever was in `prev` described a different geometry. */
	p->has_prev = 0;

	GLenum e = glGetError();
	if (e != GL_NO_ERROR) {
		ERR("could not allocate a %zu-float tensor buffer for %d tile(s) "
		    "(0x%04x)", total, p->tiles, e);
		return -1;
	}

	cuda_register(p);
	return 0;
}

struct bc_preproc *bc_preproc_create(int net_w, int net_h)
{
	if (net_w <= 0 || net_h <= 0 || (net_w % 32) || (net_h % 32)) {
		ERR("network input %dx%d must be positive multiples of 32",
		    net_w, net_h);
		return NULL;
	}

	struct bc_preproc *p = calloc(1, sizeof *p);
	if (!p)
		return NULL;

	p->net_w  = net_w;
	p->net_h  = net_h;
	p->floats = (size_t)3 * (size_t)net_w * (size_t)net_h;
	p->cols = p->rows = p->grid = p->tiles = 1;
	p->overlap = 0.0f;
	/* Identity normalisation: the shader's (c - 0) * 1 is the plain 0..1
	 * fetch every YOLO export here expects. */
	for (int i = 0; i < 3; i++) {
		p->mean[i]   = 0.0f;
		p->invstd[i] = 1.0f;
	}

	p->prog = compile_compute(CS_SRC);
	if (!p->prog) {
		free(p);
		return NULL;
	}

	p->u_net      = glGetUniformLocation(p->prog, "uNet");
	p->u_src_size = glGetUniformLocation(p->prog, "uSrcSize");
	p->u_scale    = glGetUniformLocation(p->prog, "uScale");
	p->u_pad      = glGetUniformLocation(p->prog, "uPad");
	p->u_base     = glGetUniformLocation(p->prog, "uBase");
	p->u_mean     = glGetUniformLocation(p->prog, "uMean");
	p->u_invstd   = glGetUniformLocation(p->prog, "uInvStd");

	p->diff_prog = compile_compute(CS_DIFF_SRC);
	if (!p->diff_prog) {
		glDeleteProgram(p->prog);
		free(p);
		return NULL;
	}
	p->u_n   = glGetUniformLocation(p->diff_prog, "uN");
	p->u_eps = glGetUniformLocation(p->diff_prog, "uEps");

	glGenBuffers(1, &p->ssbo);
	glGenBuffers(1, &p->prev);
	glGenBuffers(1, &p->cnt);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, p->cnt);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), NULL,
	             GL_DYNAMIC_READ);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	if (alloc_ssbo(p) != 0) {
		bc_preproc_destroy(p);
		return NULL;
	}

	LOG("preprocess: %dx%d tensor, %zu floats (%.2f MB)",
	    net_w, net_h, p->floats,
	    (double)(p->floats * sizeof(float)) / (1024.0 * 1024.0));
	return p;
}

int bc_preproc_set_grid(struct bc_preproc *p, int cols, int rows, float overlap,
                        int full_frame)
{
	if (!p)
		return -1;
	if (cols < 1) cols = 1;
	if (rows < 1) rows = 1;
	if (cols > BC_MAX_GRID) cols = BC_MAX_GRID;
	if (rows > BC_MAX_GRID) rows = BC_MAX_GRID;
	/*
	 * An overlap of 1 would mean every tile is the same tile, and the
	 * spacing divides by (cols-1) worth of span, so the useful range stops
	 * well short of that. Below zero the tiles would leave a gap between
	 * them -- a strip of screen no tile looks at, i.e. content that cannot
	 * be censored because it is never examined.
	 */
	if (overlap < 0.0f) overlap = 0.0f;
	if (overlap > 0.9f) overlap = 0.9f;

	/*
	 * A 1x1 grid *is* the full frame, so the extra tile would be an exact
	 * duplicate of tile 0 -- the same pixels inferred twice for the same
	 * answer. Drop it rather than let the setting silently double the cost
	 * of the default configuration.
	 */
	full_frame = full_frame && (cols > 1 || rows > 1);

	if (cols == p->cols && rows == p->rows && overlap == p->overlap &&
	    full_frame == p->full)
		return 0;

	p->cols = cols;
	p->rows = rows;
	p->grid = cols * rows;
	p->full = full_frame;
	p->tiles = p->grid + (full_frame ? 1 : 0);
	p->overlap = overlap;
	if (alloc_ssbo(p) != 0)
		return -1;

	/* Force a re-layout on the next run. */
	p->src_w = p->src_h = 0;
	return 0;
}

int bc_preproc_tiles(const struct bc_preproc *p) { return p ? p->tiles : 1; }
int bc_preproc_cols(const struct bc_preproc *p) { return p ? p->cols : 1; }

int bc_preproc_full_tile(const struct bc_preproc *p)
{
	return (p && p->full) ? p->grid : -1;
}

size_t bc_preproc_total_floats(const struct bc_preproc *p)
{
	return p ? p->floats * (size_t)p->tiles : 0;
}

void bc_preproc_destroy(struct bc_preproc *p)
{
	if (!p)
		return;
	cuda_unregister(p);
	if (p->ssbo)
		glDeleteBuffers(1, &p->ssbo);
	if (p->prev)
		glDeleteBuffers(1, &p->prev);
	if (p->cnt)
		glDeleteBuffers(1, &p->cnt);
	if (p->prog)
		glDeleteProgram(p->prog);
	if (p->diff_prog)
		glDeleteProgram(p->diff_prog);
	free(p);
}

size_t bc_preproc_floats(const struct bc_preproc *p)
{
	return p ? p->floats : 0;
}

float bc_preproc_scale(const struct bc_preproc *p)
{
	return p ? p->scale[0] : 1.0f;
}

/*
 * Work out the scale and each tile's offset for a given source size.
 *
 * The scale is the largest at which the grid still covers the whole source:
 * `cols` tiles laid across it share `overlap` of a tile with each neighbour, so
 * together they span net_w * (cols - (cols-1)*overlap) network pixels, and the
 * same down the other axis. Whichever axis runs out first sets the scale, so
 * aspect is preserved and the slack on the other axis becomes padding -- which
 * at 1x1 is exactly the old letterbox.
 *
 * Tiles are spaced evenly with the first and last flush against the edges, so
 * nothing outside the grid is ever left unexamined. The overlap is then a
 * consequence of the spacing rather than something applied separately.
 */
static void layout(struct bc_preproc *p, int src_w, int src_h)
{
	p->src_w = src_w;
	p->src_h = src_h;

	const float span_x = (float)p->cols - (float)(p->cols - 1) * p->overlap;
	const float span_y = (float)p->rows - (float)(p->rows - 1) * p->overlap;
	const float sx = (float)p->net_w * span_x / (float)src_w;
	const float sy = (float)p->net_h * span_y / (float)src_h;
	const float gscale = sx < sy ? sx : sy;

	/* The source at that scale, in network pixels. */
	const float sw = (float)src_w * gscale;
	const float sh = (float)src_h * gscale;

	for (int r = 0; r < p->rows; r++) {
		for (int c = 0; c < p->cols; c++) {
			const int t = r * p->cols + c;
			p->scale[t] = gscale;
			/*
			 * A tile's pad is the negation of its origin: the shader
			 * reads (pixel - pad)/scale, so a tile starting at
			 * origin ox in the scaled source needs pad = -ox.
			 * When the scaled source is *narrower* than one tile
			 * there is nothing to split, and the axis falls back to
			 * centring -- the plain letterbox.
			 */
			if (p->cols > 1 && sw > (float)p->net_w)
				p->pad_x[t] = -(float)c * (sw - (float)p->net_w)
				              / (float)(p->cols - 1);
			else
				p->pad_x[t] = ((float)p->net_w - sw) * 0.5f;

			if (p->rows > 1 && sh > (float)p->net_h)
				p->pad_y[t] = -(float)r * (sh - (float)p->net_h)
				              / (float)(p->rows - 1);
			else
				p->pad_y[t] = ((float)p->net_h - sh) * 0.5f;
		}
	}

	/*
	 * The scale a single tile would have used: the whole source inside one
	 * network input, whichever axis runs out first, the slack centred. This
	 * is both the yardstick the grid's gain is measured against and the
	 * full-frame tile's own layout -- they are the same thing, which is the
	 * point. Note it is what the grid formula reduces to at 1x1.
	 */
	const float one = (float)p->net_w / (float)src_w;
	const float two = (float)p->net_h / (float)src_h;
	const float base = one < two ? one : two;

	if (p->full) {
		const int t = p->grid;
		p->scale[t] = base;
		p->pad_x[t] = ((float)p->net_w - (float)src_w * base) * 0.5f;
		p->pad_y[t] = ((float)p->net_h - (float)src_h * base) * 0.5f;
	}

	if (p->tiles == 1) {
		LOG("letterbox: %dx%d -> %dx%d scale=%.4f pad=(%.1f,%.1f)",
		    src_w, src_h, p->net_w, p->net_h, p->scale[0],
		    p->pad_x[0], p->pad_y[0]);
	} else {
		/*
		 * The gain is the whole point of sectioning, so state it rather
		 * than leaving the user to infer it from the tile count. It is
		 * the ratio to the scale a single tile would have used: how much
		 * bigger the content is in the detector's eyes.
		 */
		LOG("sections: %dx%d grid, %.0f%% overlap, scale=%.4f "
		    "(%.2fx the detail of one tile)%s, %d inferences per frame",
		    p->cols, p->rows, p->overlap * 100.0f, gscale,
		    gscale / base, p->full ? " + full frame" : "", p->tiles);
	}
}

void bc_preproc_set_norm(struct bc_preproc *p, const float mean[3],
                         const float std[3])
{
	for (int i = 0; i < 3; i++) {
		p->mean[i]   = mean ? mean[i] : 0.0f;
		const float s = std ? std[i] : 1.0f;
		p->invstd[i] = (s > 1e-6f || s < -1e-6f) ? 1.0f / s : 1.0f;
	}
}

int bc_preproc_set_rects(struct bc_preproc *p, const float *rects, int n)
{
	if (n < 0 || n > BC_MAX_TILES)
		return -1;

	/*
	 * Arbitrary crops reuse the tile machinery wholesale. A tile is only
	 * ever a scale and an offset -- the shader reads (pixel - pad)/scale --
	 * so a face box is the same transform with different numbers, and
	 * bc_preproc_tile_to_screen() inverts it without knowing the difference.
	 *
	 * The grid fields are set to a degenerate 1x1 so nothing downstream
	 * mistakes these for a grid: the seam merge and the detail figure both
	 * read cols/rows, and neither means anything here.
	 */
	p->cols = p->rows = 1;
	p->grid = p->tiles = n;
	p->full = 0;
	p->rect_mode = 1;
	p->overlap = 0.0f;

	for (int i = 0; i < n; i++) {
		const float rw = rects[i * 4 + 2];
		const float rh = rects[i * 4 + 3];
		if (!(rw > 0.0f) || !(rh > 0.0f))
			return -1;
		/*
		 * Fit the crop inside the input, keeping aspect -- the same
		 * letterbox the grid uses, so a non-square face box is padded
		 * rather than stretched. A top-down landmark model is trained
		 * on aspect-preserved crops; stretching would move every point.
		 */
		const float sx = (float)p->net_w / rw;
		const float sy = (float)p->net_h / rh;
		const float s  = sx < sy ? sx : sy;
		p->scale[i] = s;
		p->pad_x[i] = -rects[i * 4 + 0] * s +
		              ((float)p->net_w - rw * s) * 0.5f;
		p->pad_y[i] = -rects[i * 4 + 1] * s +
		              ((float)p->net_h - rh * s) * 0.5f;
	}

	if (alloc_ssbo(p) != 0)
		return -1;
	return 0;
}

int bc_preproc_run(struct bc_preproc *p, unsigned tex, int src_w, int src_h)
{
	if (!p || !tex || src_w <= 0 || src_h <= 0)
		return -1;

	if (p->rect_mode) {
		p->src_w = src_w;
		p->src_h = src_h;
	} else if (src_w != p->src_w || src_h != p->src_h) {
		layout(p, src_w, src_h);
	}

	glUseProgram(p->prog);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, p->ssbo);

	glUniform2i(p->u_net, p->net_w, p->net_h);
	glUniform2f(p->u_src_size, (float)src_w, (float)src_h);

	/*
	 * One dispatch per tile. They differ in where they read from (the pad
	 * and, for the full-frame tile, the scale) and where they write to (the
	 * base), so the shader is the same one the single-tile path has always
	 * used.
	 */
	for (int t = 0; t < p->tiles; t++) {
		glUniform3fv(p->u_mean, 1, p->mean);
		glUniform3fv(p->u_invstd, 1, p->invstd);
		glUniform1f(p->u_scale, p->scale[t]);
		glUniform2f(p->u_pad, p->pad_x[t], p->pad_y[t]);
		glUniform1i(p->u_base, (int)((size_t)t * p->floats));
		glDispatchCompute((GLuint)((p->net_w + 7) / 8),
		                  (GLuint)((p->net_h + 7) / 8), 1);
	}

	GLenum e = glGetError();
	if (e != GL_NO_ERROR) {
		ERR("preprocess dispatch failed (0x%04x)", e);
		return -1;
	}
	return 0;
}

float bc_preproc_change_frac(struct bc_preproc *p, float eps)
{
	if (!p)
		return 1.0f;

	const size_t n = p->floats * (size_t)p->tiles;

	/*
	 * Nothing to compare against yet, so the honest answer is "all of it".
	 * Returning 0 would read as "unchanged" and gate the very first
	 * inference, leaving the screen uncensored until something moved.
	 */
	if (!p->has_prev)
		return 1.0f;

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	const GLuint zero = 0;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, p->cnt);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof zero, &zero);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	glUseProgram(p->diff_prog);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, p->ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, p->prev);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, p->cnt);
	glUniform1ui(p->u_n, (GLuint)n);
	glUniform1f(p->u_eps, eps);
	glDispatchCompute((GLuint)((n + 255) / 256), 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

	/* The only thing that crosses the bus: four bytes. */
	GLuint diff = 0;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, p->cnt);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof diff, &diff);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	/*
	 * BC_VERIFY_GATE=1: do it the old way as well and compare.
	 *
	 * Temporary cross-check for the move from a CPU loop to this shader. A
	 * gate that silently over-reports "identical" would stop inference and
	 * leave stale bars on live content, which no other counter would show.
	 */
	if (getenv("BC_VERIFY_GATE")) {
		float *a = malloc(n * sizeof *a), *b = malloc(n * sizeof *b);
		if (a && b) {
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, p->ssbo);
			glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
			                   (GLsizeiptr)(n * sizeof(float)), a);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, p->prev);
			glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
			                   (GLsizeiptr)(n * sizeof(float)), b);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			size_t cpu = 0;
			for (size_t i = 0; i < n; i++) {
				const float dd = a[i] - b[i];
				if (dd > eps || dd < -eps)
					cpu++;
			}
			if (cpu != (size_t)diff)
				ERR("gate mismatch: gpu %u cpu %zu of %zu",
				    diff, cpu, n);
			else
				LOG("gate ok: %u differ of %zu (%.2f%%)",
				    diff, n, 100.0 * (double)diff / (double)n);
		}
		free(a);
		free(b);
	}

	/*
	 * The reference frame is *not* advanced here. See bc_preproc_commit():
	 * the caller advances it when it submits, so the comparison is against
	 * the last frame inferred rather than the last one seen.
	 */
	GLenum e = glGetError();
	if (e != GL_NO_ERROR) {
		/*
		 * Report "everything changed" on error. That makes the caller
		 * infer rather than gate, which is the direction that keeps the
		 * screen covered.
		 */
		ERR("change gate failed (0x%04x) -- treating the frame as new", e);
		return 1.0f;
	}
	return n ? (float)diff / (float)n : 0.0f;
}

int bc_preproc_readback(struct bc_preproc *p, float *dst)
{
	if (!p || !dst)
		return -1;

	/* Make the shader's writes visible to the subsequent read. */
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, p->ssbo);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
	                   (GLsizeiptr)(p->floats * (size_t)p->tiles
	                                * sizeof(float)), dst);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	GLenum e = glGetError();
	if (e != GL_NO_ERROR) {
		ERR("tensor readback failed (0x%04x)", e);
		return -1;
	}
	return 0;
}

void bc_preproc_commit(struct bc_preproc *p)
{
	if (!p)
		return;

	const size_t n = p->floats * (size_t)p->tiles;

	/*
	 * VRAM to VRAM; it never touches the bus. Must happen while the buffer
	 * still belongs to GL -- which is why the caller commits *before*
	 * handing the tensor to CUDA, not after.
	 */
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
	glBindBuffer(GL_COPY_READ_BUFFER, p->ssbo);
	glBindBuffer(GL_COPY_WRITE_BUFFER, p->prev);
	glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
	                    (GLsizeiptr)(n * sizeof(float)));
	glBindBuffer(GL_COPY_READ_BUFFER, 0);
	glBindBuffer(GL_COPY_WRITE_BUFFER, 0);

	GLenum e = glGetError();
	if (e != GL_NO_ERROR) {
		/*
		 * A reference frame that is not what we think it is would make
		 * the gate answer a question nobody asked, so throw it away:
		 * has_prev = 0 sends the next call down the "all of it" path
		 * and back to inference.
		 */
		ERR("could not keep the gate's reference frame (0x%04x)", e);
		p->has_prev = 0;
		return;
	}
	p->has_prev = 1;
}

void bc_preproc_gate_reset(struct bc_preproc *p)
{
	if (p)
		p->has_prev = 0;
}

int bc_preproc_gpu_register(struct bc_preproc *p)
{
	if (!p)
		return -1;
#ifndef BC_HAVE_CUDA
	return -1;
#else
	/*
	 * Idempotent: called again after every grid change, where the buffer has
	 * been reallocated and alloc_ssbo() has already taken the registration
	 * back for us.
	 */
	if (p->res)
		return 0;
	if (p->cuda_failed)
		return -1;
	if (getenv("BC_NO_ZEROCOPY")) {
		LOG("preprocess: BC_NO_ZEROCOPY set -- tensor goes via host memory");
		p->cuda_failed = 1;
		return -1;
	}
	p->want_cuda = 1;
	if (cuda_register(p) != 0) {
		p->cuda_failed = 1;
		return -1;
	}
	LOG("preprocess: tensor handed to CUDA in place (%.2f MB/frame stays "
	    "in VRAM)",
	    (double)(p->floats * (size_t)p->tiles * sizeof(float))
	    / (1024.0 * 1024.0));
	return 0;
#endif
}

int bc_preproc_zero_copy(const struct bc_preproc *p)
{
#ifdef BC_HAVE_CUDA
	return p && p->res != NULL;
#else
	(void)p;
	return 0;
#endif
}

const void *bc_preproc_device_ptr(struct bc_preproc *p)
{
#ifndef BC_HAVE_CUDA
	(void)p;
	return NULL;
#else
	if (!p || !p->res)
		return NULL;
	if (p->dev)
		return p->dev;   /* still mapped from a submission that did not take */

	/*
	 * The dispatch is queued, not finished. cudaGraphicsMapResources orders
	 * CUDA work on the stream it is given against GL, but ORT runs on its
	 * own stream, so that guarantee does not reach the inference. glFinish
	 * is the blunt version and costs nothing we were not already paying:
	 * glGetBufferSubData, the call this replaces, blocked here too.
	 */
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
	glFinish();

	cudaError_t e = cudaGraphicsMapResources(1, &p->res, 0);
	if (e != cudaSuccess) {
		ERR("could not map the tensor for CUDA (%s)", cudaGetErrorString(e));
		return NULL;
	}

	size_t bytes = 0;
	e = cudaGraphicsResourceGetMappedPointer(&p->dev, &bytes, p->res);
	if (e != cudaSuccess || !p->dev) {
		ERR("no device pointer for the tensor (%s)", cudaGetErrorString(e));
		cudaGraphicsUnmapResources(1, &p->res, 0);
		p->dev = NULL;
		return NULL;
	}

	/*
	 * A short buffer would be read past the end by the inference, which is
	 * a fault in someone else's memory rather than a wrong answer. Refuse
	 * instead; the readback path is still there.
	 */
	const size_t want = p->floats * (size_t)p->tiles * sizeof(float);
	if (bytes < want) {
		ERR("mapped tensor is %zu bytes, need %zu", bytes, want);
		cudaGraphicsUnmapResources(1, &p->res, 0);
		p->dev = NULL;
		return NULL;
	}
	return p->dev;
#endif
}

void bc_preproc_gpu_release(struct bc_preproc *p)
{
#ifdef BC_HAVE_CUDA
	if (p && p->dev) {
		cudaGraphicsUnmapResources(1, &p->res, 0);
		p->dev = NULL;
	}
#else
	(void)p;
#endif
}

int bc_preproc_dump_ppm(const struct bc_preproc *p, const float *t,
                        const char *path)
{
	if (!p || !t)
		return -1;

	FILE *f = fopen(path, "wb");
	if (!f) {
		ERR("cannot open %s", path);
		return -1;
	}

	fprintf(f, "P6\n%d %d\n255\n", p->net_w, p->net_h);
	int n = p->net_w * p->net_h;
	for (int i = 0; i < n; i++) {
		for (int c = 0; c < 3; c++) {
			float v = t[(size_t)c * n + i] * 255.0f;
			if (v < 0.0f)   v = 0.0f;
			if (v > 255.0f) v = 255.0f;
			fputc((int)(v + 0.5f), f);
		}
	}
	fclose(f);
	LOG("debug: wrote preprocessed tensor to %s (%dx%d)",
	    path, p->net_w, p->net_h);
	return 0;
}

void bc_preproc_tile_to_screen(const struct bc_preproc *p, int tile,
                               float nx, float ny, float *sx, float *sy)
{
	if (!p)
		return;
	if (tile < 0 || tile >= p->tiles)
		tile = 0;
	if (p->scale[tile] <= 0.0f)
		return;
	*sx = (nx - p->pad_x[tile]) / p->scale[tile];
	*sy = (ny - p->pad_y[tile]) / p->scale[tile];
}

void bc_preproc_to_screen(const struct bc_preproc *p,
                          float nx, float ny, float *sx, float *sy)
{
	bc_preproc_tile_to_screen(p, 0, nx, ny, sx, sy);
}

void bc_preproc_to_net(const struct bc_preproc *p,
                       float sx, float sy, float *nx, float *ny)
{
	if (!p)
		return;
	*nx = sx * p->scale[0] + p->pad_x[0];
	*ny = sy * p->scale[0] + p->pad_y[0];
}

/*
 * Width of the strip two neighbouring tiles share, in *screen* pixels, and 0
 * when there is only one tile on that axis. The seam merge needs this: it is
 * the most two halves of a straddling detection can possibly overlap.
 */
void bc_preproc_overlap_px(const struct bc_preproc *p, float *ox, float *oy)
{
	*ox = *oy = 0.0f;
	if (!p || p->scale[0] <= 0.0f)
		return;
	/* The grid's scale, which every tile but the full-frame one uses -- and
	 * the full-frame tile has no seam with anything. */
	const float gs = p->scale[0];
	if (p->cols > 1) {
		const float sw = (float)p->src_w * gs;
		const float step = (sw - (float)p->net_w) / (float)(p->cols - 1);
		const float ov = (float)p->net_w - step;
		*ox = ov > 0.0f ? ov / gs : 0.0f;
	}
	if (p->rows > 1) {
		const float sh = (float)p->src_h * gs;
		const float step = (sh - (float)p->net_h) / (float)(p->rows - 1);
		const float ov = (float)p->net_h - step;
		*oy = ov > 0.0f ? ov / gs : 0.0f;
	}
}

int bc_preproc_net_w(const struct bc_preproc *p) { return p ? p->net_w : 0; }
int bc_preproc_net_h(const struct bc_preproc *p) { return p ? p->net_h : 0; }
