#include "framering.h"
#include "util.h"

/* glCopyImageSubData/glTexStorage2D are GL 4.3/4.2 core; glext.h only declares
 * them when asked, and this driver's libGL exports them directly. */
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>

#include <stdlib.h>
#include <string.h>

struct slot {
	GLuint        tex;
	unsigned long seq;
	double        t;        /* ms, when the frame was captured */
	int           valid;
};

struct bc_framering {
	struct slot *slots;
	int          n;
	int          next;      /* slot to overwrite */
	int          count;
	int          w, h;
	GLuint       fbo;
	GLuint       prog;
	GLuint       vao;
};

/*
 * The copy is a *render*, not glCopyImageSubData.
 *
 * glCopyImageSubData moves raw texels and does no format conversion. The
 * capture texture is EGL-imported from a DMA-BUF whose DRM fourcc is typically
 * XRGB8888 -- BGRA in memory -- so copying those bytes verbatim into an RGBA8
 * texture swaps red and blue, which shows up as everything turning blue. The
 * pixelate path never hit this because it samples the source, and sampling is
 * where the driver applies the format.
 *
 * So sample it too: bind the destination to an FBO and draw a full-screen
 * triangle. Slightly more work per frame, correct for every fourcc the
 * compositor might hand us, and still entirely on the GPU (§0).
 */
static const char *RING_VS =
"#version 430 core\n"
"void main() {\n"
"    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
"    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
"}\n";

/* No v flip: destination row 0 takes source row 0, so a ring frame has the
 * same orientation as the capture texture and consumers need no special case. */
static const char *RING_FS =
"#version 430 core\n"
"layout(binding = 0) uniform sampler2D uSrc;\n"
"uniform vec2 uSize;\n"
"out vec4 frag;\n"
"void main() {\n"
"    frag = vec4(texture(uSrc, gl_FragCoord.xy / uSize).rgb, 1.0);\n"
"}\n";

static GLuint ring_program(void)
{
	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &RING_VS, NULL);
	glCompileShader(vs);
	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &RING_FS, NULL);
	glCompileShader(fs);

	GLuint p = glCreateProgram();
	glAttachShader(p, vs);
	glAttachShader(p, fs);
	glLinkProgram(p);

	GLint ok = 0;
	glGetProgramiv(p, GL_LINK_STATUS, &ok);
	glDeleteShader(vs);
	glDeleteShader(fs);
	if (!ok) {
		char log[512] = { 0 };
		glGetProgramInfoLog(p, sizeof log, NULL, log);
		ERR("frame ring program link failed: %s", log);
		glDeleteProgram(p);
		return 0;
	}
	return p;
}

struct bc_framering *bc_framering_create(int w, int h, int n)
{
	if (w <= 0 || h <= 0 || n <= 0)
		return NULL;

	struct bc_framering *r = calloc(1, sizeof *r);
	if (!r)
		return NULL;

	/* glCopyImageSubData and glTexStorage2D are core in 4.3, which the
	 * preprocess compute shader already requires; this libGL exports them
	 * directly, as preprocess.c's bare glDispatchCompute call relies on. */

	r->slots = calloc((size_t)n, sizeof *r->slots);
	if (!r->slots) {
		free(r);
		return NULL;
	}
	r->n = n;
	r->w = w;
	r->h = h;

	for (int i = 0; i < n; i++) {
		glGenTextures(1, &r->slots[i].tex);
		glBindTexture(GL_TEXTURE_2D, r->slots[i].tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		/*
		 * Immutable storage: glCopyImageSubData requires it, and it also
		 * pins the format so a driver cannot silently pick something the
		 * copy then rejects.
		 */
		glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);
	}
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenFramebuffers(1, &r->fbo);
	glGenVertexArrays(1, &r->vao);
	r->prog = ring_program();
	if (!r->prog) {
		bc_framering_destroy(r);
		return NULL;
	}

	GLenum e = glGetError();
	if (e != GL_NO_ERROR) {
		ERR("frame ring allocation failed (0x%04x)", e);
		bc_framering_destroy(r);
		return NULL;
	}

	LOG("frame ring: %d slots of %dx%d (%.0f MB)", n, w, h,
	    (double)n * w * h * 4 / (1024.0 * 1024.0));
	return r;
}

void bc_framering_destroy(struct bc_framering *r)
{
	if (!r)
		return;
	for (int i = 0; i < r->n; i++)
		if (r->slots[i].tex)
			glDeleteTextures(1, &r->slots[i].tex);
	if (r->fbo)  glDeleteFramebuffers(1, &r->fbo);
	if (r->vao)  glDeleteVertexArrays(1, &r->vao);
	if (r->prog) glDeleteProgram(r->prog);
	free(r->slots);
	free(r);
}

int bc_framering_push(struct bc_framering *r, unsigned src, unsigned long seq,
                      double t)
{
	if (!r || !src)
		return -1;

	struct slot *s = &r->slots[r->next];

	GLint prev_fbo = 0, prev_vp[4];
	glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prev_fbo);
	glGetIntegerv(GL_VIEWPORT, prev_vp);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, r->fbo);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
	                       GL_TEXTURE_2D, s->tex, 0);
	glViewport(0, 0, r->w, r->h);
	glDisable(GL_SCISSOR_TEST);
	glDisable(GL_BLEND);

	glUseProgram(r->prog);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, src);
	glUniform2f(glGetUniformLocation(r->prog, "uSize"),
	            (float)r->w, (float)r->h);
	glBindVertexArray(r->vao);
	glDrawArrays(GL_TRIANGLES, 0, 3);

	glUseProgram(0);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)prev_fbo);
	glViewport(prev_vp[0], prev_vp[1], prev_vp[2], prev_vp[3]);

	GLenum e = glGetError();
	if (e != GL_NO_ERROR) {
		/*
		 * Leave the slot invalid rather than claiming it holds this
		 * frame: a lookup that returns a texture holding some *other*
		 * frame would put the censor bars on the wrong pixels, which is
		 * worse than reporting a miss.
		 */
		s->valid = 0;
		ERR("frame ring copy failed (0x%04x)", e);
		return -1;
	}

	s->seq   = seq;
	s->t     = t;
	s->valid = 1;
	r->next  = (r->next + 1) % r->n;
	if (r->count < r->n)
		r->count++;
	return 0;
}

unsigned bc_framering_get(const struct bc_framering *r, unsigned long seq)
{
	if (!r)
		return 0;
	for (int i = 0; i < r->n; i++)
		if (r->slots[i].valid && r->slots[i].seq == seq)
			return r->slots[i].tex;
	return 0;
}

unsigned bc_framering_at(const struct bc_framering *r, double t_max,
                         unsigned long *seq, double *t)
{
	if (!r)
		return 0;
	const struct slot *best = NULL;
	for (int i = 0; i < r->n; i++) {
		const struct slot *s = &r->slots[i];
		if (s->valid && s->t <= t_max && (!best || s->t > best->t))
			best = s;
	}
	if (!best)
		return 0;
	if (seq) *seq = best->seq;
	if (t)   *t   = best->t;
	return best->tex;
}

int bc_framering_selftest(const struct bc_framering *r, unsigned src)
{
	if (!r || !src || r->count == 0)
		return -1;

	int last = (r->next - 1 + r->n) % r->n;
	size_t n = (size_t)r->w * r->h * 4;
	unsigned char *a = malloc(n), *b = malloc(n);
	if (!a || !b) { free(a); free(b); return -1; }

	glBindTexture(GL_TEXTURE_2D, src);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, a);
	glBindTexture(GL_TEXTURE_2D, r->slots[last].tex);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, b);
	glBindTexture(GL_TEXTURE_2D, 0);

	/* Centre pixel: a corner is often flat black and would match under any
	 * permutation, proving nothing. */
	size_t o = ((size_t)(r->h / 2) * r->w + r->w / 2) * 4;
	int rc = 0;
	if (a[o] == b[o] && a[o+1] == b[o+1] && a[o+2] == b[o+2]) {
		LOG("frame ring self-test: OK (channels preserved)");
	} else if (a[o] == b[o+2] && a[o+2] == b[o]) {
		ERR("frame ring self-test: RED/BLUE SWAPPED -- the copy is not "
		    "converting the source format");
		rc = -1;
	} else {
		ERR("frame ring self-test: mismatch src=(%u,%u,%u) ring=(%u,%u,%u)",
		    a[o], a[o+1], a[o+2], b[o], b[o+1], b[o+2]);
		rc = -1;
	}
	free(a); free(b);
	return rc;
}

int bc_framering_capacity(const struct bc_framering *r) { return r ? r->n : 0; }
int bc_framering_count(const struct bc_framering *r) { return r ? r->count : 0; }
