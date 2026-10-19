/*
 * The headless pipeline shared by bc-video and bc-image. See offline.h.
 *
 * Everything here used to be private to video.c; it moved out unchanged when
 * the image pass arrived, so that a photo and a video frame are censored by
 * literally the same code.
 */

#include "offline.h"
#include "shaders.h"
#include "font8x16.h"
#include "util.h"

#include <EGL/eglext.h>
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Headless GL                                                         */

/*
 * Draw the decoded frame into the render target, flipping v so the result is
 * oriented the way the overlay's framebuffer is: GL y=0 at the bottom of the
 * picture. Everything downstream -- the top-left-to-bottom-left box conversion,
 * the scissor arithmetic, the shader's own src() flip -- is then the live code
 * unchanged, which is the point.
 */
static const char *BLIT_FS =
"#version 430 core\n"
"layout(binding = 0) uniform sampler2D uSrc;\n"
"uniform vec2 uSize;\n"
"out vec4 frag;\n"
"void main() {\n"
"    vec2 uv = vec2(gl_FragCoord.x / uSize.x, 1.0 - gl_FragCoord.y / uSize.y);\n"
"    frag = vec4(texture(uSrc, uv).rgb, 1.0);\n"
"}\n";

/*
 * The hardening pass (see bc_off_harden). Reads a copy of the frame as drawn
 * and writes it back, inside a cover's scissor, with three things added:
 *
 *   - a shift per block, on the pixelate effect's grid: floor(p / uBlock)
 *     over the same gl_FragCoord, so each mosaic block gets one shift and
 *     its true mean is gone;
 *   - grain per pixel, so a block is no longer one exact value either;
 *   - a tiled BETACENSOR mark at 30 degrees, from the glyph atlas, inked
 *     dark on light content and light on dark so it reads on both.
 *
 * The hash is a PCG step keyed by the seed: cheap, and with no structure
 * visible at the scale of a block.
 */
static const char *HARDEN_FS =
"#version 430 core\n"
"layout(binding = 0) uniform sampler2D uSrc;\n"
"layout(binding = 1) uniform sampler2D uFont;\n"
"uniform float uBlock;\n"
"uniform uint  uSeed;\n"
"uniform float uAmp;\n"
"uniform float uGlyph;\n"
"uniform float uMark;\n"
"out vec4 frag;\n"
"uint pcg(uint v) {\n"
"    uint s = v * 747796405u + 2891336453u;\n"
"    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;\n"
"    return (w >> 22u) ^ w;\n"
"}\n"
"float r11(uvec2 c, uint k) {\n"
"    uint h = pcg(c.x ^ pcg(c.y ^ pcg(k ^ uSeed)));\n"
"    return float(h) / 4294967295.0 * 2.0 - 1.0;\n"
"}\n"
"const int MARK[11] = int[](66, 69, 84, 65, 67, 69, 78, 83, 79, 82, 32);\n"
"float mark(vec2 p) {\n"
"    const float ca = 0.8660254, sa = 0.5;\n"
"    vec2 q = vec2(p.x * ca + p.y * sa, -p.x * sa + p.y * ca) / uGlyph;\n"
"    float row = floor(q.y / 32.0);\n"
"    float gy  = q.y - row * 32.0;\n"
"    if (gy >= 16.0)\n"
"        return 0.0;\n"
"    float xi = mod(q.x + row * 37.0, 88.0);\n"
"    int   ch = MARK[int(xi / 8.0)];\n"
"    int   gx = int(mod(xi, 8.0));\n"
"    return texelFetch(uFont, ivec2((ch - 32) * 8 + gx, 15 - int(gy)), 0).r;\n"
"}\n"
"void main() {\n"
"    vec3  c    = texelFetch(uSrc, ivec2(gl_FragCoord.xy), 0).rgb;\n"
"    uvec2 cell = uvec2(floor(gl_FragCoord.xy / uBlock));\n"
"    float lum  = r11(cell, 1u) * uAmp;\n"
"    vec3  chro = vec3(r11(cell, 2u), r11(cell, 3u), r11(cell, 4u))\n"
"                 * uAmp * 0.5;\n"
"    float fine = r11(uvec2(gl_FragCoord.xy), 5u) * uAmp * 0.35;\n"
"    c = clamp(c + vec3(lum + fine) + chro, 0.0, 1.0);\n"
"    float ink = step(dot(c, vec3(0.299, 0.587, 0.114)), 0.55);\n"
"    c = mix(c, vec3(ink), mark(gl_FragCoord.xy) * uMark);\n"
"    frag = vec4(c, 1.0);\n"
"}\n";

static GLuint compile(GLenum type, const char *src)
{
	GLuint s = glCreateShader(type);
	glShaderSource(s, 1, &src, NULL);
	glCompileShader(s);
	GLint ok = 0;
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[4096];
		glGetShaderInfoLog(s, sizeof log, NULL, log);
		ERR("shader compile failed: %s", log);
		glDeleteShader(s);
		return 0;
	}
	return s;
}

static GLuint program(const char *vs_src, const char *fs_src)
{
	GLuint vs = compile(GL_VERTEX_SHADER, vs_src);
	GLuint fs = compile(GL_FRAGMENT_SHADER, fs_src);
	if (!vs || !fs)
		return 0;
	GLuint p = glCreateProgram();
	glAttachShader(p, vs);
	glAttachShader(p, fs);
	glLinkProgram(p);
	GLint ok = 0;
	glGetProgramiv(p, GL_LINK_STATUS, &ok);
	glDeleteShader(vs);
	glDeleteShader(fs);
	if (!ok) {
		char log[4096];
		glGetProgramInfoLog(p, sizeof log, NULL, log);
		ERR("program link failed: %s", log);
		glDeleteProgram(p);
		return 0;
	}
	return p;
}

static void font_atlas(struct bc_off_gl *g)
{
	static unsigned char atlas[BC_FONT_H][BC_FONT_N * BC_FONT_W];

	for (int c = 0; c < BC_FONT_N; c++)
		for (int y = 0; y < BC_FONT_H; y++)
			for (int x = 0; x < BC_FONT_W; x++)
				atlas[y][c * BC_FONT_W + x] =
				    (BC_FONT[c][y] >> (BC_FONT_W - 1 - x)) & 1
				    ? 0xff : 0x00;

	glGenTextures(1, &g->font_tex);
	glBindTexture(GL_TEXTURE_2D, g->font_tex);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, BC_FONT_N * BC_FONT_W, BC_FONT_H,
	             0, GL_RED, GL_UNSIGNED_BYTE, atlas);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

/* (Re)allocate the two size-dependent textures and attach the target. */
static int alloc_targets(struct bc_off_gl *g, int w, int h)
{
	glBindTexture(GL_TEXTURE_2D, g->src_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, w, h, 0, GL_RGB,
	             GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_2D, g->fbo_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA,
	             GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	/* The hardening pass reads the drawn frame while writing it. */
	glBindTexture(GL_TEXTURE_2D, g->work_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA,
	             GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glBindFramebuffer(GL_FRAMEBUFFER, g->fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
	                       GL_TEXTURE_2D, g->fbo_tex, 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		ERR("render target incomplete at %dx%d", w, h);
		return -1;
	}
	/*
	 * An allocation the driver refused leaves a zero-sized texture behind
	 * that still reports complete on some drivers; the error flag is the
	 * only thing that says so.
	 */
	if (glGetError() != GL_NO_ERROR) {
		ERR("could not allocate a %dx%d render target", w, h);
		return -1;
	}
	g->w = w;
	g->h = h;
	glViewport(0, 0, w, h);
	return 0;
}

/*
 * A context with no window. The device platform is tried first because it is
 * what works on the proprietary NVIDIA driver; the surfaceless platform is the
 * Mesa route. Either way the context is made current against EGL_NO_SURFACE and
 * all rendering goes to an FBO.
 */
int bc_off_gl_init(struct bc_off_gl *g, int w, int h)
{
	memset(g, 0, sizeof *g);
	g->dpy = EGL_NO_DISPLAY;

	PFNEGLQUERYDEVICESEXTPROC query_devices =
	    (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
	PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
	    (PFNEGLGETPLATFORMDISPLAYEXTPROC)
	    eglGetProcAddress("eglGetPlatformDisplayEXT");

	if (query_devices && get_display) {
		EGLDeviceEXT dev[8];
		EGLint n = 0;
		if (query_devices(8, dev, &n)) {
			for (int i = 0; i < n && g->dpy == EGL_NO_DISPLAY; i++) {
				EGLDisplay d = get_display(EGL_PLATFORM_DEVICE_EXT,
				                           dev[i], NULL);
				EGLint maj, min;
				if (d != EGL_NO_DISPLAY && eglInitialize(d, &maj, &min))
					g->dpy = d;
			}
		}
	}
	if (g->dpy == EGL_NO_DISPLAY && get_display) {
		EGLDisplay d = get_display(EGL_PLATFORM_SURFACELESS_MESA,
		                           EGL_DEFAULT_DISPLAY, NULL);
		EGLint maj, min;
		if (d != EGL_NO_DISPLAY && eglInitialize(d, &maj, &min))
			g->dpy = d;
	}
	if (g->dpy == EGL_NO_DISPLAY) {
		ERR("no headless EGL display available");
		return -1;
	}

	if (!eglBindAPI(EGL_OPENGL_API)) {
		ERR("eglBindAPI(OpenGL) failed");
		return -1;
	}

	static const EGLint cfg_attr[] = {
		EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT,
		EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
		EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
		EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
		EGL_NONE
	};
	EGLConfig cfg;
	EGLint    n_cfg = 0;
	if (!eglChooseConfig(g->dpy, cfg_attr, &cfg, 1, &n_cfg) || n_cfg < 1) {
		ERR("no suitable EGL config");
		return -1;
	}

	static const EGLint ctx_attr[] = {
		EGL_CONTEXT_MAJOR_VERSION, 4,
		EGL_CONTEXT_MINOR_VERSION, 3,
		EGL_NONE
	};
	g->ctx = eglCreateContext(g->dpy, cfg, EGL_NO_CONTEXT, ctx_attr);
	if (g->ctx == EGL_NO_CONTEXT) {
		ERR("eglCreateContext failed (0x%04x)", eglGetError());
		return -1;
	}
	if (!eglMakeCurrent(g->dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, g->ctx)) {
		ERR("eglMakeCurrent(surfaceless) failed (0x%04x)", eglGetError());
		return -1;
	}

	g->prog   = program(BC_VS_SRC, BC_FS_SRC);
	g->blit   = program(BC_VS_SRC, BLIT_FS);
	g->harden = program(BC_VS_SRC, HARDEN_FS);
	if (!g->prog || !g->blit || !g->harden)
		return -1;
	g->h_block = glGetUniformLocation(g->harden, "uBlock");
	g->h_seed  = glGetUniformLocation(g->harden, "uSeed");
	g->h_amp   = glGetUniformLocation(g->harden, "uAmp");
	g->h_glyph = glGetUniformLocation(g->harden, "uGlyph");
	g->h_mark  = glGetUniformLocation(g->harden, "uMark");

	g->u_size     = glGetUniformLocation(g->prog, "uSize");
	g->u_block    = glGetUniformLocation(g->prog, "uBlock");
	g->u_alpha    = glGetUniformLocation(g->prog, "uAlpha");
	g->u_effect   = glGetUniformLocation(g->prog, "uEffect");
	g->u_time     = glGetUniformLocation(g->prog, "uTime");
	g->u_box      = glGetUniformLocation(g->prog, "uBox");
	g->u_feather  = glGetUniformLocation(g->prog, "uFeather");
	g->u_angle    = glGetUniformLocation(g->prog, "uAngle");
	g->u_color    = glGetUniformLocation(g->prog, "uColor");
	g->u_text     = glGetUniformLocation(g->prog, "uText");
	g->u_text_len = glGetUniformLocation(g->prog, "uTextLen");
	g->u_text_mode = glGetUniformLocation(g->prog, "uTextMode");
	g->u_text_col  = glGetUniformLocation(g->prog, "uTextCol");
	g->b_size     = glGetUniformLocation(g->blit, "uSize");

	GLint mt = 0;
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &mt);
	g->max_tex = mt > 0 ? mt : 8192;

	glGenVertexArrays(1, &g->vao);
	glGenTextures(1, &g->src_tex);
	glGenTextures(1, &g->fbo_tex);
	glGenTextures(1, &g->work_tex);
	glGenFramebuffers(1, &g->fbo);
	font_atlas(g);

	return bc_off_gl_resize(g, w, h);
}

int bc_off_gl_resize(struct bc_off_gl *g, int w, int h)
{
	if (w == g->w && h == g->h)
		return 0;
	if (w <= 0 || h <= 0 || w > g->max_tex || h > g->max_tex) {
		ERR("%dx%d is larger than this GPU's %d px texture limit",
		    w, h, g->max_tex);
		return -1;
	}
	while (glGetError() != GL_NO_ERROR)
		;
	return alloc_targets(g, w, h);
}

void bc_off_upload(struct bc_off_gl *g, const unsigned char *rgb)
{
	glBindTexture(GL_TEXTURE_2D, g->src_tex);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g->w, g->h, GL_RGB,
	                GL_UNSIGNED_BYTE, rgb);
}

/* ------------------------------------------------------------------ */
/* Drawing                                                             */

/*
 * Mirrors the per-box pass in bc_overlay_present(). It is deliberately the
 * same shape: same block-size derivation, same full-width scissor for tape,
 * same feather for static. Anything that differs here is an effect that would
 * look different in a file than on screen.
 */
/* The effect's block size for a box. One function, because the hardening
 * pass must land on exactly the grid the pixelate effect drew. */
static float block_of(const struct bc_box *b)
{
	int   shortest = b->w < b->h ? b->w : b->h;
	float pct      = b->strength > 0.0f ? b->strength : 8.0f;
	float block    = (float)shortest * pct / 100.0f;
	if (block < 4.0f)   block = 4.0f;
	if (block > 256.0f) block = 256.0f;
	return block;
}

void bc_off_draw(struct bc_off_gl *g, const struct bc_box *b, int n, double t)
{
	glBindFramebuffer(GL_FRAMEBUFFER, g->fbo);

	/* Background: the frame itself. */
	glDisable(GL_SCISSOR_TEST);
	glUseProgram(g->blit);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, g->src_tex);
	glUniform2f(g->b_size, (float)g->w, (float)g->h);
	glBindVertexArray(g->vao);
	glDrawArrays(GL_TRIANGLES, 0, 3);

	glEnable(GL_SCISSOR_TEST);
	glUseProgram(g->prog);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, g->font_tex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, g->src_tex);

	for (int i = 0; i < n; i++) {
		int y = g->h - (b[i].y + b[i].h);
		glScissor(b[i].x, y, b[i].w, b[i].h);

		float block = block_of(&b[i]);

		if (b[i].effect == BC_EFFECT_TAPE)
			glScissor(0, y, g->w, b[i].h);

		float feather = 0.0f;
		if (b[i].effect == BC_EFFECT_NOISE) {
			feather = block * 0.75f;
			if (feather < 6.0f)  feather = 6.0f;
			if (feather > 40.0f) feather = 40.0f;
			int f  = (int)feather + 1;
			int sx = b[i].x - f, sy = y - f;
			int sw = b[i].w + 2 * f, sh = b[i].h + 2 * f;
			if (sx < 0) { sw += sx; sx = 0; }
			if (sy < 0) { sh += sy; sy = 0; }
			if (sx + sw > g->w) sw = g->w - sx;
			if (sy + sh > g->h) sh = g->h - sy;
			glScissor(sx, sy, sw, sh);
		}

		glUniform2f(g->u_size, (float)g->w, (float)g->h);
		glUniform1f(g->u_block, block);
		glUniform1f(g->u_alpha, 1.0f);
		glUniform1i(g->u_effect, b[i].effect);
		glUniform3f(g->u_color, 0.0f, 0.0f, 0.0f);
		glUniform1f(g->u_time, (float)t);
		glUniform4f(g->u_box, (float)b[i].x, (float)y,
		            (float)b[i].w, (float)b[i].h);
		glUniform1f(g->u_feather, feather);
		glUniform1f(g->u_angle, 0.0f);

		GLint codes[64];
		int   nc = 0;
		for (const char *p = b[i].text; *p && nc < 64; p++)
			if ((unsigned char)*p >= 32 && (unsigned char)*p <= 126)
				codes[nc++] = (unsigned char)*p;
		glUniform1iv(g->u_text, nc ? nc : 1,
		             nc ? codes : (GLint[]){ 32 });
		glUniform1i(g->u_text_len, nc);
		glUniform1i(g->u_text_mode, b[i].text_mode);
		glUniform3f(g->u_text_col, b[i].text_rgb[0],
		            b[i].text_rgb[1], b[i].text_rgb[2]);

		glBindVertexArray(g->vao);
		glDrawArrays(GL_TRIANGLES, 0, 3);
	}
	glDisable(GL_SCISSOR_TEST);
}

/*
 * glReadPixels returns bottom-up; encoders want top-down. Flipping on the way
 * out rather than on the way in keeps the shaders' coordinate convention
 * identical to the live path.
 */
void bc_off_read(struct bc_off_gl *g, unsigned char *rgb)
{
	const size_t row = (size_t)g->w * 3;
	glBindFramebuffer(GL_FRAMEBUFFER, g->fbo);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, g->w, g->h, GL_RGB, GL_UNSIGNED_BYTE, rgb);

	unsigned char tmp[3 * 4096];
	for (int y = 0; y < g->h / 2; y++) {
		unsigned char *a = rgb + (size_t)y * row;
		unsigned char *b = rgb + (size_t)(g->h - 1 - y) * row;
		for (size_t o = 0; o < row; o += sizeof tmp) {
			size_t n = row - o < sizeof tmp ? row - o : sizeof tmp;
			memcpy(tmp, a + o, n);
			memcpy(a + o, b + o, n);
			memcpy(b + o, tmp, n);
		}
	}
}

/* ------------------------------------------------------------------ */
/* Hardening                                                           */

int bc_off_effect_samples(int effect)
{
	switch (effect) {
	case BC_EFFECT_PIXELATE: case BC_EFFECT_BLUR: case BC_EFFECT_GLITCH:
	case BC_EFFECT_RGBSHIFT: case BC_EFFECT_HALFTONE: case BC_EFFECT_SOBEL:
	case BC_EFFECT_TRIANGLE: case BC_EFFECT_SHAPE:
		return 1;
	default:
		return 0;
	}
}

void bc_off_harden(struct bc_off_gl *g, const struct bc_box *b, int n,
                   uint32_t seed)
{
	int any = 0;
	for (int i = 0; i < n; i++)
		any |= bc_off_effect_samples(b[i].effect);
	if (!any)
		return;

	glCopyImageSubData(g->fbo_tex, GL_TEXTURE_2D, 0, 0, 0, 0,
	                   g->work_tex, GL_TEXTURE_2D, 0, 0, 0, 0,
	                   g->w, g->h, 1);

	glBindFramebuffer(GL_FRAMEBUFFER, g->fbo);
	glUseProgram(g->harden);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, g->font_tex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, g->work_tex);
	glUniform1ui(g->h_seed, seed);
	/* ~±36/255 of shift per block: past what matching a block's colour
	 * can tolerate, short of turning the cover into confetti. */
	glUniform1f(g->h_amp, 0.14f);
	glUniform1f(g->h_mark, 0.22f);
	glEnable(GL_SCISSOR_TEST);
	glBindVertexArray(g->vao);

	for (int i = 0; i < n; i++) {
		if (!bc_off_effect_samples(b[i].effect))
			continue;
		const int shortest = b[i].w < b[i].h ? b[i].w : b[i].h;
		/* Glyphs sized to the cover, so the mark is a texture over
		 * it rather than one huge word or an invisible speck. */
		float glyph = (float)shortest / 110.0f;
		if (glyph < 1.0f) glyph = 1.0f;
		if (glyph > 8.0f) glyph = 8.0f;
		glScissor(b[i].x, g->h - (b[i].y + b[i].h), b[i].w, b[i].h);
		glUniform1f(g->h_block, block_of(&b[i]));
		glUniform1f(g->h_glyph, glyph);
		glDrawArrays(GL_TRIANGLES, 0, 3);
	}
	glDisable(GL_SCISSOR_TEST);
}

/* splitmix64: the finaliser, used to fold values into a seed. */
static uint64_t mix64(uint64_t x)
{
	x += 0x9e3779b97f4a7c15ull;
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
	x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
	return x ^ (x >> 31);
}

uint64_t bc_off_run_seed(void)
{
	const char *s = getenv("BC_HARDEN_SEED");
	if (s && *s)
		return strtoull(s, NULL, 16);
	uint64_t v = 0;
	FILE *f = fopen("/dev/urandom", "rb");
	if (!f || fread(&v, sizeof v, 1, f) != 1) {
		/*
		 * No kernel randomness is the one case where the pattern
		 * would be guessable; the clock and pid are a poor stand-in
		 * and it is said so rather than hidden.
		 */
		WARN("no /dev/urandom -- the hardening pattern is weaker");
		struct timespec ts;
		clock_gettime(CLOCK_REALTIME, &ts);
		v = (uint64_t)ts.tv_nsec ^ ((uint64_t)ts.tv_sec << 20) ^
		    ((uint64_t)getpid() << 40);
	}
	if (f)
		fclose(f);
	return v;
}

uint32_t bc_off_mix_seed(uint64_t run, uint64_t v)
{
	return (uint32_t)(mix64(run ^ mix64(v)) >> 16);
}

/* ------------------------------------------------------------------ */
/* Badge                                                               */

#define BADGE_TEXT "censored with BetaCensor"

/* A font file for drawtext, or NULL. This ffmpeg may lack fontconfig, so
 * fc-match is asked for a path rather than drawtext for a family. */
static const char *badge_font(void)
{
	static char path[1024];
	static int  tried;
	if (tried)
		return path[0] ? path : NULL;
	tried = 1;
	if (getenv("BC_BADGE_BITMAP"))
		return NULL;
	FILE *f = popen("fc-match -f '%{file}' 'sans-serif:medium' 2>/dev/null",
	                "r");
	if (!f)
		return NULL;
	if (!fgets(path, sizeof path, f))
		path[0] = 0;
	pclose(f);
	path[strcspn(path, "\n")] = 0;
	/* Filter-graph quoting has its own special characters; a path that
	 * needs them takes the bitmap route rather than an escaping bet. */
	if (!path[0] || strpbrk(path, "':\\,;[]=") || access(path, R_OK) != 0)
		path[0] = 0;
	return path[0] ? path : NULL;
}

/* drawtext onto a transparent canvas, then crop to what it drew. */
static int badge_drawtext(struct bc_off_badge *b, int fs, int pad)
{
	const char *font = badge_font();
	if (!font)
		return -1;
	const int cw = fs * 20, ch = fs * 3;
	char cmd[2048];
	snprintf(cmd, sizeof cmd,
	         "ffmpeg -v error -f lavfi -i 'color=c=black@0.0:s=%dx%d,"
	         "format=rgba' -frames:v 1 -vf \"drawtext=fontfile='%s':"
	         "text='" BADGE_TEXT "':fontsize=%d:fontcolor=white:box=1:"
	         "boxcolor=black:boxborderw=%d:x=w-tw-%d:y=h-th-%d\" "
	         "-f rawvideo -pix_fmt rgba - 2>/dev/null",
	         cw, ch, font, fs, pad, pad + 1, pad + 1);
	FILE *f = popen(cmd, "r");
	if (!f)
		return -1;
	const size_t sz = (size_t)cw * ch * 4;
	unsigned char *rgba = malloc(sz);
	const int got = rgba && fread(rgba, 1, sz, f) == sz;
	const int rc = pclose(f);
	if (!got || rc != 0) {
		free(rgba);
		return -1;
	}
	int x0 = cw, y0 = ch, x1 = -1, y1 = -1;
	for (int y = 0; y < ch; y++)
		for (int x = 0; x < cw; x++)
			if (rgba[((size_t)y * cw + x) * 4 + 3]) {
				if (x < x0) x0 = x;
				if (x > x1) x1 = x;
				if (y < y0) y0 = y;
				if (y > y1) y1 = y;
			}
	if (x1 < 0) {
		free(rgba);
		return -1;
	}
	b->w = x1 - x0 + 1;
	b->h = y1 - y0 + 1;
	b->rgb = malloc((size_t)b->w * b->h * 3);
	if (!b->rgb) {
		free(rgba);
		return -1;
	}
	/* Opaque all through -- the text is blended over its own black box --
	 * so alpha can be dropped rather than composited. */
	for (int y = 0; y < b->h; y++)
		for (int x = 0; x < b->w; x++)
			memcpy(b->rgb + ((size_t)y * b->w + x) * 3,
			       rgba + ((size_t)(y + y0) * cw + x + x0) * 4, 3);
	free(rgba);
	b->bitmap = 0;
	return 0;
}

/* The fallback: the 8x16 font at an integer scale. Blockier, same layout. */
static int badge_bitmap(struct bc_off_badge *b, int fs, int pad)
{
	int s = (fs + 8) / 16;
	if (s < 1)
		s = 1;
	const int n = (int)strlen(BADGE_TEXT);
	b->w = n * BC_FONT_W * s + 2 * pad;
	b->h = BC_FONT_H * s + 2 * pad;
	b->rgb = calloc((size_t)b->w * b->h, 3);
	if (!b->rgb)
		return -1;
	for (int i = 0; i < n; i++) {
		const int c = (unsigned char)BADGE_TEXT[i] - BC_FONT_LO;
		if (c < 0 || c >= BC_FONT_N)
			continue;
		for (int gy = 0; gy < BC_FONT_H * s; gy++)
			for (int gx = 0; gx < BC_FONT_W * s; gx++)
				if ((BC_FONT[c][gy / s] >> (BC_FONT_W - 1 - gx / s)) & 1)
					memset(b->rgb + ((size_t)(pad + gy) * b->w +
					       pad + i * BC_FONT_W * s + gx) * 3,
					       0xff, 3);
	}
	b->bitmap = 1;
	return 0;
}

int bc_off_badge_make(struct bc_off_badge *b, int fw, int fh)
{
	if (b->fw == fw && b->fh == fh)
		return b->rgb ? 0 : -1;
	bc_off_badge_free(b);
	b->fw = fw;
	b->fh = fh;

	/* About 3% of the short side: readable on a phone screenshot, quiet
	 * on a 4K frame. Shrunk if it would take most of a narrow image. */
	const int shortest = fw < fh ? fw : fh;
	int fs = (int)(shortest * 0.03f + 0.5f);
	if (fs < 11) fs = 11;
	if (fs > 72) fs = 72;
	for (; fs >= 9; fs = fs * 3 / 4) {
		const int pad = (fs * 2 + 4) / 5;
		if (badge_drawtext(b, fs, pad) != 0 &&
		    badge_bitmap(b, fs, pad) != 0)
			return -1;
		const int margin = (fs * 11 + 10) / 20;
		if (b->w + 2 * margin <= fw * 4 / 5 &&
		    b->h + 2 * margin <= fh / 4) {
			b->x = fw - margin - b->w;
			b->y = fh - margin - b->h;
			return 0;
		}
		free(b->rgb);
		b->rgb = NULL;
	}
	return -1;   /* too small an image for a legible badge */
}

void bc_off_badge_apply(const struct bc_off_badge *b, unsigned char *rgb)
{
	if (!b->rgb)
		return;
	for (int y = 0; y < b->h; y++)
		memcpy(rgb + ((size_t)(b->y + y) * b->fw + b->x) * 3,
		       b->rgb + (size_t)y * b->w * 3, (size_t)b->w * 3);
}

void bc_off_badge_free(struct bc_off_badge *b)
{
	free(b->rgb);
	memset(b, 0, sizeof *b);
}

/* ------------------------------------------------------------------ */

struct bc_det_result bc_off_detect(struct bc_off_gl *g, struct bc_preproc *pp,
                                   struct bc_detector *det, float *tensor,
                                   struct bc_det *dets, int max, int *hung)
{
	struct bc_det_result res = { BC_DET_FAILED, 0, 0.0 };
	*hung = 0;
	if (bc_preproc_run(pp, g->src_tex, g->w, g->h) != 0 ||
	    bc_preproc_readback(pp, tensor) != 0 ||
	    !bc_detect_submit(det, tensor))
		return res;

	res.status = BC_DET_PENDING;
	for (int spin = 0; spin < 100000 && res.status == BC_DET_PENDING;
	     spin++) {
		res = bc_detect_collect(det, dets, max);
		if (res.status == BC_DET_PENDING) {
			struct timespec ts = { 0, 100000 };
			nanosleep(&ts, NULL);
		}
	}
	if (res.status == BC_DET_PENDING)
		*hung = 1;
	return res;
}

/* Takes the effect rather than the config, so the same mapping serves the
 * global effect and a class's pinned one. */
int bc_off_effect(int e)
{
	switch (e) {
	case BC_CFG_FILL:     return BC_EFFECT_FILL;
	case BC_CFG_BLUR:     return BC_EFFECT_BLUR;
	case BC_CFG_NOISE:    return BC_EFFECT_NOISE;
	case BC_CFG_GLITCH:   return BC_EFFECT_GLITCH;
	case BC_CFG_RGBSHIFT: return BC_EFFECT_RGBSHIFT;
	case BC_CFG_TAPE:     return BC_EFFECT_TAPE;
	case BC_CFG_HALFTONE: return BC_EFFECT_HALFTONE;
	case BC_CFG_SOBEL:    return BC_EFFECT_SOBEL;
	case BC_CFG_TRIANGLE: return BC_EFFECT_TRIANGLE;
	case BC_CFG_CLUSTER:  return BC_EFFECT_CLUSTER;
	case BC_CFG_SPLATTER: return BC_EFFECT_SPLATTER;
	case BC_CFG_SHAPE:    return BC_EFFECT_SHAPE;
	case BC_CFG_CAPTION:  return BC_EFFECT_CAPTION;
	case BC_CFG_WALL:     return BC_EFFECT_WALL;
	default:              return BC_EFFECT_PIXELATE;
	}
}

struct bc_box bc_off_box(const struct bc_config *cfg, int cls,
                         int x, int y, int w, int h)
{
	/* This box's own effect, from its class -- the live path does the
	 * same, and an offline render of the same config has to match it. */
	int ce = bc_cfg_class_effect(cfg, cls);
	struct bc_box b = {
		.x = x, .y = y, .w = w, .h = h,
		.effect = bc_off_effect(ce),
		.strength = cfg->strength_by[ce],
		.text_mode = cfg->text_mode,
		.text_rgb = { cfg->text_rgb[0], cfg->text_rgb[1],
		              cfg->text_rgb[2] }
	};
	const char *txt = cfg->caption[0] ? cfg->caption : bc_class_name(cls);
	snprintf(b.text, sizeof b.text, "%s", txt ? txt : "");
	return b;
}

/* Same pairing the live path uses: the model flips between these two for one
 * physical thing, and without it a flipped frame abandons the track. */
int bc_off_classes_similar(int a, int b)
{
	if (a == b)
		return 1;
	const char *na = bc_class_name(a), *nb = bc_class_name(b);
	if (!na || !nb)
		return 0;
	const char *sa = strstr(na, "BREAST"), *sb = strstr(nb, "BREAST");
	return sa && sb && !strcmp(sa, sb);
}

/*
 * Shell-quote for the ffmpeg command lines. Paths come from argv, so they can
 * contain anything; building a command string without this is a command
 * injection waiting for a file named with a semicolon in it.
 *
 * A path that does not fit is refused rather than cut short: the truncated
 * quote is a valid path too, and it names some other file.
 */
int bc_off_shq(char *dst, size_t cap, const char *s)
{
	size_t n = 0;
	if (cap < 3)
		return -1;
	dst[n++] = '\'';
	for (; *s; s++) {
		if (n + 6 >= cap) {
			dst[0] = 0;
			return -1;
		}
		if (*s == '\'') {
			/* close, escaped quote, reopen */
			memcpy(dst + n, "'\\''", 4);
			n += 4;
		} else {
			dst[n++] = *s;
		}
	}
	dst[n++] = '\'';
	dst[n] = 0;
	return 0;
}

double bc_off_now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
