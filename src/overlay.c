#include "overlay.h"
#include "font8x16.h"
#include "shaders.h"

#include <math.h>
#include <time.h>
#include "util.h"

#include <wayland-client.h>
#include <wayland-egl.h>
#include "wlr-layer-shell-unstable-v1-client-protocol.h"
#include "xdg-shell-client-protocol.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>

#include <stdlib.h>
#include <string.h>
#include <unistd.h>   /* close() -- the keymap fd, see kbd_keymap() */

#include "detect.h"   /* BC_MAX_DET */

/* Seconds a reverse-mode cutout survives a gap in detections. */
#define BC_REVERSE_HOLD 0.20

#define MAX_OUTPUTS 8

struct output_ref {
	struct wl_output *wl;
	uint32_t          global_name;
	char              name[64];   /* connector name, e.g. "DP-3" */
	int32_t           scale;
	int32_t           mode_w, mode_h;
};

struct bc_overlay {
	struct wl_display    *dpy;
	struct wl_registry   *registry;
	struct wl_compositor *compositor;
	struct zwlr_layer_shell_v1 *shell;

	struct output_ref outputs[MAX_OUTPUTS];
	int               n_outputs;

	struct xdg_wm_base          *wm_base;

	enum bc_overlay_mode         mode;
	struct wl_surface           *surface;
	struct zwlr_layer_surface_v1 *layer;
	struct xdg_surface          *xdg_surface;
	struct xdg_toplevel         *xdg_toplevel;

	int  logical_w, logical_h;
	int  scale;
	int  buf_w, buf_h;
	/* The output we ended up on: its mode, in physical pixels, and its
	 * name. Kept so the core can check the capture it is being fed
	 * actually describes this screen. */
	int  out_mode_w, out_mode_h;
	char out_name[64];
	/* Peek rate limiting; see bc_overlay_peek(). */
	double        last_peek_ms;
	unsigned long peeks;
	int  configured;
	int  closed;

	struct wl_egl_window *egl_window;
	EGLDisplay egl_dpy;
	EGLContext egl_ctx;
	EGLSurface egl_surf;

	/* Pixelate pass. */
	GLuint   pix_prog;
	GLuint   vao;
	GLint    u_size, u_block, u_alpha, u_effect, u_time, u_box, u_feather, u_angle;
	GLint    u_color;
	GLint    u_auto_col, u_outline, u_out_col, u_shadow, u_shadow_a;
	GLint    u_text, u_text_len, u_text_mode, u_text_col;
	GLint    u_use_mask, u_mask_rect;
	GLuint   font_tex;

	/*
	 * Subject silhouette. mask_obj is the GL texture, kept once allocated;
	 * mask_tex is it or 0, and 0 is what means "cover everything", so
	 * clearing the mask costs no allocation to undo.
	 */
	GLuint   mask_obj;
	int      mask_w, mask_h;
	unsigned mask_tex;
	float    mask_rect[4];   /* u0 v0 u1 v1, screen corners in mask uv */
	int      reverse;
	/* Style for the reverse mask; see bc_overlay_set_mask(). */
	struct bc_box mask;
	/*
	 * Last cutouts, and when they were seen.
	 *
	 * In reverse mode the boxes are holes in an opaque screen, so "no
	 * detections this frame" does not mean "cover a little less" the way it
	 * does normally -- it means the entire display goes solid black, and
	 * the next frame with a detection puts it back. One dropped frame is a
	 * full-screen flash, and at 180 Hz an intermittent drop is a strobe.
	 * Nothing counts it as a fault because the screen was, if anything,
	 * over-covered throughout.
	 *
	 * So a gap shorter than BC_REVERSE_HOLD reuses the previous holes
	 * rather than closing them. This is the one place the §2 bias is
	 * deliberately relaxed: it keeps a hole open for up to 200 ms after the
	 * detection behind it went away. That is bounded, it only ever applies
	 * to a region already judged safe to reveal a moment ago, and the
	 * alternative is a photosensitive hazard.
	 */
	struct bc_box hold[BC_MAX_BOXES];
	int      n_hold;
	double   hold_t;
	/*
	 * Origin for uTime. CLOCK_MONOTONIC counts from boot, so on a machine
	 * that has been up a while it arrives as a seven-digit number, and a
	 * float carries about seven digits: anything the shaders scale it by
	 * lands past the point where consecutive floats are still a fraction
	 * apart. The animated effects then step in visible jumps -- worse the
	 * longer the machine has been up, which is the kind of bug that never
	 * reproduces on a freshly booted one. Counting from our own start keeps
	 * it small.
	 */
	double   t0;
	unsigned src_tex;
	GLuint   bg_prog;       /* full-screen blit of the source, for the delayed screen */
	int      bg_on;
	int      src_w, src_h;

	/*
	 * Drawing-mode input. Bound always, listened to always, but harmless
	 * until bc_overlay_set_input() gives the surface an input region: with
	 * an empty one the compositor sends no pointer or key events here at
	 * all, so `input_on` is a statement about the region rather than a
	 * filter applied after the fact.
	 */
	struct wl_seat     *seat;
	struct wl_pointer  *ptr;
	struct wl_keyboard *kbd;
	int      input_on;
	int      ptr_in;              /* pointer is over this surface */
	int      ptr_x, ptr_y;        /* pixels, same space as bc_box */
	int      keyq[8];
	int      keyq_n;
	int           btn_down;      /* left button held right now */
	unsigned char btn_press;     /* edges since the last read; see ptr_button */
	unsigned char btn_release;
};

/*
 * Queue a key press for the caller to pop.
 *
 * Dropped when full rather than overwriting the oldest: these are Enter and
 * Escape, and the order they were pressed in is the whole meaning. Eight is
 * more than a frame's worth of deliberate keystrokes, so a full queue means
 * the caller has stopped draining, and in that case the honest thing is to
 * lose the newest rather than reorder the ones already waiting.
 */
static void key_push(struct bc_overlay *o, int k)
{
	if (o->keyq_n < (int)(sizeof o->keyq / sizeof o->keyq[0]))
		o->keyq[o->keyq_n++] = k;
}

/*
 * Fullscreen triangle from gl_VertexID -- no vertex buffer. The scissor box
 * clips it to the region being censored.
 */

/*
 * Block-average pixelation on a *screen-aligned* grid.
 *
 * The alignment is the whole trick. We capture our own output, so this shader
 * samples a region it previously wrote. Block-averaging on a fixed grid is
 * idempotent -- re-pixelating an already-pixelated block returns the same
 * value, because the average of a constant is that constant. So the region
 * settles at a stable fixed point instead of washing out toward flat grey the
 * way an unaligned grid or a blur-of-a-blur would.
 *
 * That is what makes the censored area keep tracking the content underneath,
 * which is what the release logic in bc_main.c depends on.
 */
/* Monotonic seconds, for effects that animate. */
static double now_seconds(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}


/*
 * Expand the baked font into a one-row atlas: 95 glyphs side by side, space
 * first, each BC_FONT_W by BC_FONT_H. One row rather than a grid because it
 * makes the lookup a single divide in the shader, and 760x16 is nothing.
 *
 * R8 with NEAREST filtering. Linear would bleed neighbouring glyphs into each
 * other at the cell edges, and these are bitmaps -- there is no sub-pixel
 * information to interpolate toward.
 */
static void build_font_atlas(struct bc_overlay *o)
{
	static unsigned char atlas[BC_FONT_H][BC_FONT_N * BC_FONT_W];

	for (int g = 0; g < BC_FONT_N; g++)
		for (int y = 0; y < BC_FONT_H; y++)
			for (int x = 0; x < BC_FONT_W; x++)
				atlas[y][g * BC_FONT_W + x] =
				    (BC_FONT[g][y] >> (BC_FONT_W - 1 - x)) & 1
				    ? 0xff : 0x00;

	glGenTextures(1, &o->font_tex);
	glBindTexture(GL_TEXTURE_2D, o->font_tex);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, BC_FONT_N * BC_FONT_W, BC_FONT_H,
	             0, GL_RED, GL_UNSIGNED_BYTE, atlas);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);
}

static GLuint build_program(const char *vs_src, const char *fs_src)
{
	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &vs_src, NULL);
	glCompileShader(vs);

	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &fs_src, NULL);
	glCompileShader(fs);

	GLint ok = 0;
	char log[1024];
	glGetShaderiv(vs, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		glGetShaderInfoLog(vs, sizeof log - 1, NULL, log);
		ERR("vertex shader: %s", log);
		goto fail;
	}
	glGetShaderiv(fs, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		glGetShaderInfoLog(fs, sizeof log - 1, NULL, log);
		ERR("fragment shader: %s", log);
		goto fail;
	}

	GLuint prog = glCreateProgram();
	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	glLinkProgram(prog);
	glDeleteShader(vs);
	glDeleteShader(fs);

	glGetProgramiv(prog, GL_LINK_STATUS, &ok);
	if (!ok) {
		glGetProgramInfoLog(prog, sizeof log - 1, NULL, log);
		ERR("pixelate program link: %s", log);
		glDeleteProgram(prog);
		return 0;
	}
	return prog;

fail:
	glDeleteShader(vs);
	glDeleteShader(fs);
	return 0;
}

/* --- wl_output ------------------------------------------------------------ */

static void out_geometry(void *data, struct wl_output *o, int32_t x, int32_t y,
                         int32_t pw, int32_t ph, int32_t subpixel,
                         const char *make, const char *model, int32_t transform)
{
}

static void out_mode(void *data, struct wl_output *o, uint32_t flags,
                     int32_t w, int32_t h, int32_t refresh)
{
	struct output_ref *r = data;
	if (flags & WL_OUTPUT_MODE_CURRENT) {
		r->mode_w = w;
		r->mode_h = h;
	}
}

static void out_done(void *data, struct wl_output *o) { }

static void out_scale(void *data, struct wl_output *o, int32_t factor)
{
	struct output_ref *r = data;
	r->scale = factor;
}

/*
 * wl_output v4 reports the connector name ("DP-3"). This is the only reliable
 * way to pin the overlay to one physical monitor -- registry order and the
 * make/model strings are both unstable across reboots and hotplugs.
 */
static void out_name(void *data, struct wl_output *o, const char *name)
{
	struct output_ref *r = data;
	snprintf(r->name, sizeof r->name, "%s", name ? name : "");
}

static void out_description(void *data, struct wl_output *o, const char *d) { }

static const struct wl_output_listener output_listener = {
	.geometry    = out_geometry,
	.mode        = out_mode,
	.done        = out_done,
	.scale       = out_scale,
	.name        = out_name,
	.description = out_description,
};

/* --- seat, for drawing mode ------------------------------------------------
 *
 * Bound at version 5 and no higher. The listeners below fill in every event a
 * v5 pointer or keyboard can send, and libwayland calls a listener slot
 * without checking it for NULL -- so binding a version whose events this file
 * does not name would crash the moment the compositor sent one. Capping the
 * bind is what keeps that a compile-time-visible property rather than a
 * dependency on which wayland-client happened to be installed.
 */
#define BC_SEAT_VERSION 5

/*
 * Raw evdev key codes. wl_keyboard reports these directly, and for Enter and
 * Escape alone that is exact -- they sit at fixed codes on every layout,
 * because a layout maps codes to *symbols* and these two have none. Anything
 * with a character behind it would need xkbcommon to interpret; nothing here
 * has one.
 */
#define BC_EVDEV_ESC      1
#define BC_EVDEV_ENTER   28
#define BC_EVDEV_KPENTER 96

static void ptr_enter(void *data, struct wl_pointer *p, uint32_t serial,
                      struct wl_surface *surf, wl_fixed_t sx, wl_fixed_t sy)
{
	struct bc_overlay *o = data;
	if (surf != o->surface)
		return;
	o->ptr_in = 1;
	o->ptr_x  = (int)(wl_fixed_to_double(sx) * o->scale);
	o->ptr_y  = (int)(wl_fixed_to_double(sy) * o->scale);
}

static void ptr_leave(void *data, struct wl_pointer *p, uint32_t serial,
                      struct wl_surface *surf)
{
	struct bc_overlay *o = data;
	if (surf == o->surface)
		o->ptr_in = 0;
}

static void ptr_motion(void *data, struct wl_pointer *p, uint32_t t,
                       wl_fixed_t sx, wl_fixed_t sy)
{
	struct bc_overlay *o = data;
	o->ptr_in = 1;
	o->ptr_x  = (int)(wl_fixed_to_double(sx) * o->scale);
	o->ptr_y  = (int)(wl_fixed_to_double(sy) * o->scale);
}

/*
 * The left button, reported as a level and two latched edges.
 *
 * The level is what a freehand stroke follows; the edges are what makes a
 * *click* survive. A click is press and release inside about eighty
 * milliseconds, and the loop that reads this runs a frame at a time -- so
 * sampling the level alone would miss whole clicks whenever both edges landed
 * between two reads. Counting them here means the reader can be as slow as it
 * likes and still see that a click happened.
 *
 * Left button only; the others are left alone so that nothing in this mode
 * happens by brushing a thumb button.
 */
#define BC_BTN_LEFT 0x110

static void ptr_button(void *data, struct wl_pointer *p, uint32_t serial,
                       uint32_t time, uint32_t button, uint32_t state)
{
	struct bc_overlay *o = data;
	if (button != BC_BTN_LEFT)
		return;
	if (state == WL_POINTER_BUTTON_STATE_PRESSED) {
		o->btn_down = 1;
		if (o->btn_press < 255)
			o->btn_press++;
	} else {
		o->btn_down = 0;
		if (o->btn_release < 255)
			o->btn_release++;
	}
}

static void ptr_axis(void *data, struct wl_pointer *p, uint32_t t,
                     uint32_t axis, wl_fixed_t value) { }
static void ptr_frame(void *data, struct wl_pointer *p) { }
static void ptr_axis_source(void *data, struct wl_pointer *p, uint32_t src) { }
static void ptr_axis_stop(void *data, struct wl_pointer *p, uint32_t t,
                          uint32_t axis) { }
static void ptr_axis_discrete(void *data, struct wl_pointer *p, uint32_t axis,
                              int32_t discrete) { }

static const struct wl_pointer_listener pointer_listener = {
	.enter         = ptr_enter,
	.leave         = ptr_leave,
	.motion        = ptr_motion,
	.button        = ptr_button,
	.axis          = ptr_axis,
	.frame         = ptr_frame,
	.axis_source   = ptr_axis_source,
	.axis_stop     = ptr_axis_stop,
	.axis_discrete = ptr_axis_discrete,
};

static void kbd_keymap(void *data, struct wl_keyboard *k, uint32_t format,
                       int32_t fd, uint32_t size)
{
	/* Not read -- see the evdev note above. Closed all the same: the
	 * compositor sends a fresh fd on every keymap change, and leaking one
	 * per change would eventually exhaust the process's descriptors. */
	if (fd >= 0)
		close(fd);
}

static void kbd_enter(void *data, struct wl_keyboard *k, uint32_t serial,
                      struct wl_surface *surf, struct wl_array *keys) { }
static void kbd_leave(void *data, struct wl_keyboard *k, uint32_t serial,
                      struct wl_surface *surf) { }

static void kbd_key(void *data, struct wl_keyboard *k, uint32_t serial,
                    uint32_t time, uint32_t key, uint32_t state)
{
	struct bc_overlay *o = data;
	if (state != WL_KEYBOARD_KEY_STATE_PRESSED)
		return;
	if (key == BC_EVDEV_ENTER || key == BC_EVDEV_KPENTER)
		key_push(o, BC_KEY_ENTER);
	else if (key == BC_EVDEV_ESC)
		key_push(o, BC_KEY_ESC);
}

static void kbd_modifiers(void *data, struct wl_keyboard *k, uint32_t serial,
                          uint32_t dep, uint32_t lat, uint32_t lock,
                          uint32_t group) { }
static void kbd_repeat_info(void *data, struct wl_keyboard *k, int32_t rate,
                            int32_t delay) { }

static const struct wl_keyboard_listener keyboard_listener = {
	.keymap      = kbd_keymap,
	.enter       = kbd_enter,
	.leave       = kbd_leave,
	.key         = kbd_key,
	.modifiers   = kbd_modifiers,
	.repeat_info = kbd_repeat_info,
};

static void seat_caps(void *data, struct wl_seat *s, uint32_t caps)
{
	struct bc_overlay *o = data;

	if ((caps & WL_SEAT_CAPABILITY_POINTER) && !o->ptr) {
		o->ptr = wl_seat_get_pointer(s);
		if (o->ptr)
			wl_pointer_add_listener(o->ptr, &pointer_listener, o);
	} else if (!(caps & WL_SEAT_CAPABILITY_POINTER) && o->ptr) {
		wl_pointer_destroy(o->ptr);
		o->ptr = NULL;
		o->ptr_in = 0;
	}

	if ((caps & WL_SEAT_CAPABILITY_KEYBOARD) && !o->kbd) {
		o->kbd = wl_seat_get_keyboard(s);
		if (o->kbd)
			wl_keyboard_add_listener(o->kbd, &keyboard_listener, o);
	} else if (!(caps & WL_SEAT_CAPABILITY_KEYBOARD) && o->kbd) {
		wl_keyboard_destroy(o->kbd);
		o->kbd = NULL;
	}
}

static void seat_name(void *data, struct wl_seat *s, const char *name) { }

static const struct wl_seat_listener seat_listener = {
	.capabilities = seat_caps,
	.name         = seat_name,
};

void bc_overlay_set_input(struct bc_overlay *o, int on)
{
	if (!o || !o->surface || !o->compositor)
		return;
	if (!!on == o->input_on)
		return;

	if (on) {
		/*
		 * A region covering the whole surface. This is the dangerous
		 * direction -- for as long as it is set, every click on this
		 * output lands here instead of on whatever is underneath -- so
		 * the caller is responsible for a bounded stay, and bc_main.c
		 * gives it one three ways: Escape, a timeout, and the config
		 * flag the GUI can clear without needing our focus.
		 */
		struct wl_region *all = wl_compositor_create_region(o->compositor);
		if (all) {
			wl_region_add(all, 0, 0, o->logical_w, o->logical_h);
			wl_surface_set_input_region(o->surface, all);
			wl_region_destroy(all);
		}
		if (o->layer)
			zwlr_layer_surface_v1_set_keyboard_interactivity(o->layer,
				ZWLR_LAYER_SURFACE_V1_KEYBOARD_INTERACTIVITY_EXCLUSIVE);
	} else {
		struct wl_region *empty = wl_compositor_create_region(o->compositor);
		if (empty) {
			wl_surface_set_input_region(o->surface, empty);
			wl_region_destroy(empty);
		}
		if (o->layer)
			zwlr_layer_surface_v1_set_keyboard_interactivity(o->layer,
				ZWLR_LAYER_SURFACE_V1_KEYBOARD_INTERACTIVITY_NONE);
		o->ptr_in = 0;
		o->keyq_n = 0;
		o->btn_down = 0;
		o->btn_press = o->btn_release = 0;
	}
	/*
	 * Committed here rather than left for the next present(). Giving input
	 * back is the safety-critical half, and it must not wait on a frame
	 * that a stalled render loop might never draw.
	 */
	wl_surface_commit(o->surface);
	wl_display_flush(o->dpy);
	o->input_on = !!on;
}

int bc_overlay_pointer(const struct bc_overlay *o, int *x, int *y)
{
	if (!o || !o->input_on || !o->ptr_in)
		return 0;
	if (x) *x = o->ptr_x;
	if (y) *y = o->ptr_y;
	return 1;
}

int bc_overlay_button(struct bc_overlay *o, int *pressed, int *released)
{
	if (pressed)  *pressed  = o ? o->btn_press   : 0;
	if (released) *released = o ? o->btn_release : 0;
	if (o) {
		o->btn_press = o->btn_release = 0;
		return o->btn_down;
	}
	return 0;
}

int bc_overlay_next_key(struct bc_overlay *o)
{
	if (!o || o->keyq_n <= 0)
		return BC_KEY_NONE;
	const int k = o->keyq[0];
	for (int i = 1; i < o->keyq_n; i++)
		o->keyq[i - 1] = o->keyq[i];
	o->keyq_n--;
	return k;
}

/* --- registry ------------------------------------------------------------- */

static void reg_global(void *data, struct wl_registry *reg, uint32_t name,
                       const char *iface, uint32_t version)
{
	struct bc_overlay *o = data;

	if (!strcmp(iface, wl_compositor_interface.name)) {
		uint32_t v = version < 4 ? version : 4;
		o->compositor = wl_registry_bind(reg, name,
		                                 &wl_compositor_interface, v);
	} else if (!strcmp(iface, zwlr_layer_shell_v1_interface.name)) {
		uint32_t v = version < 4 ? version : 4;
		o->shell = wl_registry_bind(reg, name,
		                            &zwlr_layer_shell_v1_interface, v);
	} else if (!strcmp(iface, xdg_wm_base_interface.name)) {
		uint32_t v = version < 5 ? version : 5;
		o->wm_base = wl_registry_bind(reg, name,
		                              &xdg_wm_base_interface, v);
	} else if (!strcmp(iface, wl_seat_interface.name)) {
		/* One seat is enough: drawing is done by whoever is at the
		 * machine, and a second seat would be a second cursor placing
		 * corners in the same rectangle. */
		if (o->seat)
			return;
		uint32_t v = version < BC_SEAT_VERSION ? version : BC_SEAT_VERSION;
		o->seat = wl_registry_bind(reg, name, &wl_seat_interface, v);
		if (o->seat)
			wl_seat_add_listener(o->seat, &seat_listener, o);
	} else if (!strcmp(iface, wl_output_interface.name)) {
		if (o->n_outputs >= MAX_OUTPUTS)
			return;
		/* v4 needed for the name event. */
		if (version < 4) {
			WARN("wl_output v%u (<4): cannot read connector names",
			     version);
		}
		uint32_t v = version < 4 ? version : 4;
		struct output_ref *r = &o->outputs[o->n_outputs++];
		memset(r, 0, sizeof *r);
		r->scale       = 1;
		r->global_name = name;
		r->wl = wl_registry_bind(reg, name, &wl_output_interface, v);
		wl_output_add_listener(r->wl, &output_listener, r);
	}
}

static void reg_global_remove(void *data, struct wl_registry *r, uint32_t name) { }

static const struct wl_registry_listener registry_listener = {
	.global        = reg_global,
	.global_remove = reg_global_remove,
};

/* --- xdg-shell ------------------------------------------------------------ */

static void wm_base_ping(void *data, struct xdg_wm_base *b, uint32_t serial)
{
	/* Not optional: ignoring pings makes the compositor kill us as hung. */
	xdg_wm_base_pong(b, serial);
}

static const struct xdg_wm_base_listener wm_base_listener = {
	.ping = wm_base_ping,
};

static void xdg_surface_configure(void *data, struct xdg_surface *s,
                                  uint32_t serial)
{
	struct bc_overlay *o = data;
	xdg_surface_ack_configure(s, serial);
	o->configured = 1;
}

static const struct xdg_surface_listener xdg_surface_listener = {
	.configure = xdg_surface_configure,
};

/*
 * The toplevel configure carries the size and arrives *before* the xdg_surface
 * configure that we ack, so record it here and commit on the ack.
 */
static void xdg_toplevel_configure(void *data, struct xdg_toplevel *t,
                                   int32_t w, int32_t h, struct wl_array *states)
{
	struct bc_overlay *o = data;
	if (w > 0 && h > 0) {
		o->logical_w = w;
		o->logical_h = h;
	}
}

static void xdg_toplevel_close(void *data, struct xdg_toplevel *t)
{
	struct bc_overlay *o = data;
	o->closed = 1;
}

static void xdg_toplevel_configure_bounds(void *data, struct xdg_toplevel *t,
                                          int32_t w, int32_t h) { }
static void xdg_toplevel_wm_capabilities(void *data, struct xdg_toplevel *t,
                                         struct wl_array *caps) { }

static const struct xdg_toplevel_listener xdg_toplevel_listener = {
	.configure        = xdg_toplevel_configure,
	.close            = xdg_toplevel_close,
	.configure_bounds = xdg_toplevel_configure_bounds,
	.wm_capabilities  = xdg_toplevel_wm_capabilities,
};

/* --- layer surface -------------------------------------------------------- */

static void layer_configure(void *data, struct zwlr_layer_surface_v1 *ls,
                            uint32_t serial, uint32_t w, uint32_t h)
{
	struct bc_overlay *o = data;

	zwlr_layer_surface_v1_ack_configure(ls, serial);
	o->logical_w  = (int)w;
	o->logical_h  = (int)h;
	o->configured = 1;
}

static void layer_closed(void *data, struct zwlr_layer_surface_v1 *ls)
{
	struct bc_overlay *o = data;
	o->closed = 1;
}

static const struct zwlr_layer_surface_v1_listener layer_listener = {
	.configure = layer_configure,
	.closed    = layer_closed,
};

/* --- EGL ------------------------------------------------------------------ */

static int egl_setup(struct bc_overlay *o)
{
	if (!eglBindAPI(EGL_OPENGL_API)) {
		ERR("eglBindAPI(OPENGL) failed");
		return -1;
	}

	o->egl_dpy = eglGetPlatformDisplay(EGL_PLATFORM_WAYLAND_KHR, o->dpy, NULL);
	if (o->egl_dpy == EGL_NO_DISPLAY) {
		ERR("eglGetPlatformDisplay(WAYLAND) failed");
		return -1;
	}

	EGLint major, minor;
	if (!eglInitialize(o->egl_dpy, &major, &minor)) {
		ERR("eglInitialize failed");
		return -1;
	}
	LOG("EGL %d.%d on the Wayland display", major, minor);

	/*
	 * Phase 3 note. EGLImages and GL textures cannot be shared across two
	 * different EGLDisplays, and bc_gpu currently opens a *headless*
	 * EGL_PLATFORM_DEVICE_EXT display while this opens a Wayland one. So
	 * the joined pipeline must run both on a single EGLDisplay -- this one
	 * -- which only works if the Wayland display can import DMA-BUFs too.
	 * Check now rather than discovering it halfway through Phase 3.
	 */
	const char *exts = eglQueryString(o->egl_dpy, EGL_EXTENSIONS);
	int has_import = exts && strstr(exts, "EGL_EXT_image_dma_buf_import");
	int has_mods   = exts && strstr(exts,
	                                "EGL_EXT_image_dma_buf_import_modifiers");
	LOG("wayland EGLDisplay: dma_buf_import=%s modifiers=%s",
	    has_import ? "yes" : "NO", has_mods ? "yes" : "NO");
	if (!has_import || !has_mods)
		WARN("Phase 3 will need both on a single shared EGLDisplay");

	/*
	 * ALPHA_SIZE 8 is not optional: without a real alpha channel the
	 * compositor treats the surface as opaque and the overlay becomes a
	 * black rectangle covering the monitor.
	 */
	static const EGLint cfg_attrs[] = {
		EGL_SURFACE_TYPE,    EGL_WINDOW_BIT,
		EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
		EGL_RED_SIZE,        8,
		EGL_GREEN_SIZE,      8,
		EGL_BLUE_SIZE,       8,
		EGL_ALPHA_SIZE,      8,
		EGL_NONE
	};
	EGLConfig cfg;
	EGLint n = 0;
	if (!eglChooseConfig(o->egl_dpy, cfg_attrs, &cfg, 1, &n) || n < 1) {
		ERR("no EGL config with an alpha channel");
		return -1;
	}

	static const EGLint ctx_attrs[] = {
		EGL_CONTEXT_MAJOR_VERSION, 4,
		EGL_CONTEXT_MINOR_VERSION, 3,
		EGL_CONTEXT_OPENGL_PROFILE_MASK,
			EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
		EGL_NONE
	};
	o->egl_ctx = eglCreateContext(o->egl_dpy, cfg, EGL_NO_CONTEXT, ctx_attrs);
	if (o->egl_ctx == EGL_NO_CONTEXT) {
		ERR("eglCreateContext(GL 4.3 core) failed");
		return -1;
	}

	o->egl_window = wl_egl_window_create(o->surface, o->buf_w, o->buf_h);
	if (!o->egl_window) {
		ERR("wl_egl_window_create failed");
		return -1;
	}

	o->egl_surf = eglCreateWindowSurface(o->egl_dpy, cfg,
	                                     (EGLNativeWindowType)o->egl_window,
	                                     NULL);
	if (o->egl_surf == EGL_NO_SURFACE) {
		ERR("eglCreateWindowSurface failed");
		return -1;
	}

	if (!eglMakeCurrent(o->egl_dpy, o->egl_surf, o->egl_surf, o->egl_ctx)) {
		ERR("eglMakeCurrent failed");
		return -1;
	}

	LOG("GL %s", (const char *)glGetString(GL_VERSION));

	o->pix_prog = build_program(BC_VS_SRC, BC_FS_SRC);
	if (!o->pix_prog) {
		ERR("could not build the pixelate program");
		return -1;
	}
	{
		static const char *BG_FS =
"#version 430 core\n"
"layout(binding = 0) uniform sampler2D uSrc;\n"
"uniform vec2 uSize;\n"
"uniform float uA;\n"
"out vec4 frag;\n"
"void main() {\n"
"    vec2 uv = vec2(gl_FragCoord.x / uSize.x, 1.0 - gl_FragCoord.y / uSize.y);\n"
"    frag = vec4(texture(uSrc, uv).rgb * uA, uA);\n"
"}\n";
		o->bg_prog = build_program(BC_VS_SRC, BG_FS);
		if (!o->bg_prog) {
			ERR("could not build the background blit program");
			return -1;
		}
	}
	o->u_size  = glGetUniformLocation(o->pix_prog, "uSize");
	o->u_block = glGetUniformLocation(o->pix_prog, "uBlock");
	o->u_alpha = glGetUniformLocation(o->pix_prog, "uAlpha");
	o->u_effect = glGetUniformLocation(o->pix_prog, "uEffect");
	o->u_time  = glGetUniformLocation(o->pix_prog, "uTime");
	o->u_box     = glGetUniformLocation(o->pix_prog, "uBox");
	o->u_feather = glGetUniformLocation(o->pix_prog, "uFeather");
	o->u_angle   = glGetUniformLocation(o->pix_prog, "uAngle");
	o->u_color   = glGetUniformLocation(o->pix_prog, "uColor");
	o->u_auto_col = glGetUniformLocation(o->pix_prog, "uAutoCol");
	o->u_outline  = glGetUniformLocation(o->pix_prog, "uOutline");
	o->u_out_col  = glGetUniformLocation(o->pix_prog, "uOutCol");
	o->u_shadow   = glGetUniformLocation(o->pix_prog, "uShadow");
	o->u_shadow_a = glGetUniformLocation(o->pix_prog, "uShadowA");
	o->u_text     = glGetUniformLocation(o->pix_prog, "uText");
	o->u_text_len = glGetUniformLocation(o->pix_prog, "uTextLen");
	o->u_text_mode = glGetUniformLocation(o->pix_prog, "uTextMode");
	o->u_text_col  = glGetUniformLocation(o->pix_prog, "uTextCol");
	o->u_use_mask  = glGetUniformLocation(o->pix_prog, "uUseMask");
	o->u_mask_rect = glGetUniformLocation(o->pix_prog, "uMaskRect");
	build_font_atlas(o);

	o->t0 = now_seconds();

	/* Core profile requires a bound VAO even when drawing without buffers. */
	glGenVertexArrays(1, &o->vao);

	return 0;
}

void bc_overlay_set_reverse(struct bc_overlay *o, int on)
{
	if (o)
		o->reverse = on;
}

void bc_overlay_set_mask(struct bc_overlay *o, const struct bc_box *style)
{
	if (!o || !style)
		return;
	o->mask = *style;
}

void bc_overlay_set_source(struct bc_overlay *o, unsigned tex, int w, int h)
{
	if (!o)
		return;
	o->src_tex = tex;
	o->src_w   = w;
	o->src_h   = h;
}

/* --- public --------------------------------------------------------------- */

struct bc_overlay *bc_overlay_create(const char *output_name)
{
	return bc_overlay_create_ex(output_name, BC_OVERLAY_LAYER);
}

struct bc_overlay *bc_overlay_create_ex(const char *output_name,
                                        enum bc_overlay_mode mode)
{
	struct bc_overlay *o = calloc(1, sizeof *o);
	if (!o)
		return NULL;
	o->scale = 1;
	o->mode  = mode;

	o->dpy = wl_display_connect(NULL);
	if (!o->dpy) {
		ERR("wl_display_connect failed (is WAYLAND_DISPLAY set?)");
		goto fail;
	}

	o->registry = wl_display_get_registry(o->dpy);
	wl_registry_add_listener(o->registry, &registry_listener, o);

	/* First roundtrip: globals. Second: per-output name/scale/mode events. */
	wl_display_roundtrip(o->dpy);
	wl_display_roundtrip(o->dpy);

	if (!o->compositor) {
		ERR("compositor does not expose wl_compositor");
		goto fail;
	}
	if (mode == BC_OVERLAY_LAYER && !o->shell) {
		ERR("compositor does not expose zwlr_layer_shell_v1");
		goto fail;
	}
	if (mode == BC_OVERLAY_TOPLEVEL && !o->wm_base) {
		ERR("compositor does not expose xdg_wm_base");
		goto fail;
	}

	struct output_ref *target = NULL;
	for (int i = 0; i < o->n_outputs; i++) {
		struct output_ref *r = &o->outputs[i];
		LOG("output %d: %-8s %dx%d scale=%d", i,
		    r->name[0] ? r->name : "(unnamed)",
		    r->mode_w, r->mode_h, r->scale);
		if (output_name && !strcmp(r->name, output_name))
			target = r;
	}

	if (output_name && !target) {
		ERR("no output named '%s' -- refusing to guess", output_name);
		goto fail;
	}
	/*
	 * With several outputs and no name, refuse rather than take the first.
	 *
	 * Guessing here is fail-open, and quietly so. The overlay goes on one
	 * monitor while the portal streams whichever the user picked in the
	 * consent dialog; when those differ, the censor draws bars on a screen
	 * it is not watching and leaves the screen it *is* watching uncovered.
	 * Nothing about that looks wrong from inside the program -- it reports
	 * detections, draws boxes and says SUCCESS.
	 *
	 * Caught by drawing over the wrong monitor on the author's own machine
	 * within a minute of the default changing to "first listed". A single
	 * output stays zero-configuration, which is the case this default was
	 * added for; more than one has to be named. BC_OUTPUT_ANY=1 restores
	 * the guess for anyone who wants it.
	 */
	if (!target && o->n_outputs > 1 && !getenv("BC_OUTPUT_ANY")) {
		ERR("%d outputs and none named -- refusing to guess which one "
		    "to censor", o->n_outputs);
		ERR("naming the wrong one covers a screen nobody is watching "
		    "and leaves the watched one bare, so this is not a default "
		    "worth having");
		for (int i = 0; i < o->n_outputs; i++)
			ERR("  betacensor live %s", o->outputs[i].name);
		goto fail;
	}
	if (!target && o->n_outputs > 0)
		target = &o->outputs[0];
	if (!target) {
		ERR("no outputs at all");
		goto fail;
	}
	o->out_mode_w = target->mode_w;
	o->out_mode_h = target->mode_h;
	snprintf(o->out_name, sizeof o->out_name, "%s", target->name);
	LOG("targeting output '%s'", target->name);
	o->scale = target->scale > 0 ? target->scale : 1;

	o->surface = wl_compositor_create_surface(o->compositor);
	if (!o->surface) {
		ERR("wl_compositor_create_surface failed");
		goto fail;
	}

	if (mode == BC_OVERLAY_LAYER) {
		o->layer = zwlr_layer_shell_v1_get_layer_surface(
			o->shell, o->surface, target->wl,
			ZWLR_LAYER_SHELL_V1_LAYER_OVERLAY, "betacensor");
		if (!o->layer) {
			ERR("get_layer_surface failed");
			goto fail;
		}
		zwlr_layer_surface_v1_add_listener(o->layer, &layer_listener, o);

		/* Anchoring all four edges gets us the full output. */
		zwlr_layer_surface_v1_set_anchor(o->layer,
			ZWLR_LAYER_SURFACE_V1_ANCHOR_TOP |
			ZWLR_LAYER_SURFACE_V1_ANCHOR_BOTTOM |
			ZWLR_LAYER_SURFACE_V1_ANCHOR_LEFT |
			ZWLR_LAYER_SURFACE_V1_ANCHOR_RIGHT);
		zwlr_layer_surface_v1_set_size(o->layer, 0, 0);
		/* -1: ignore panels' exclusive zones, cover the whole output. */
		zwlr_layer_surface_v1_set_exclusive_zone(o->layer, -1);
		zwlr_layer_surface_v1_set_keyboard_interactivity(o->layer,
			ZWLR_LAYER_SURFACE_V1_KEYBOARD_INTERACTIVITY_NONE);
	} else {
		xdg_wm_base_add_listener(o->wm_base, &wm_base_listener, o);

		o->xdg_surface = xdg_wm_base_get_xdg_surface(o->wm_base,
		                                             o->surface);
		if (!o->xdg_surface) {
			ERR("get_xdg_surface failed");
			goto fail;
		}
		xdg_surface_add_listener(o->xdg_surface, &xdg_surface_listener, o);

		o->xdg_toplevel = xdg_surface_get_toplevel(o->xdg_surface);
		if (!o->xdg_toplevel) {
			ERR("get_toplevel failed");
			goto fail;
		}
		xdg_toplevel_add_listener(o->xdg_toplevel,
		                          &xdg_toplevel_listener, o);

		/*
		 * The title and app_id are what make this findable in the task
		 * manager, which is the only place KWin offers "Hide from
		 * Screencast". Without them the whole point is lost.
		 */
		xdg_toplevel_set_title(o->xdg_toplevel, "BetaCensor Overlay");
		xdg_toplevel_set_app_id(o->xdg_toplevel, "betacensor");

		/*
		 * Note: this window will not draw over *another* fullscreen
		 * window. Measured -- neither the Layer window rule (Overlay, On
		 * Screen Display, Critical Notification all tried) nor dropping
		 * our own fullscreen request changes it, while a layer-shell
		 * OVERLAY surface covers a fullscreen browser fine. Accepted
		 * trade: capture exclusion is worth more here. BC_LAYER=1 swaps
		 * the priorities back.
		 */
		xdg_toplevel_set_fullscreen(o->xdg_toplevel, target->wl);
	}

	/*
	 * An *empty* input region is what makes the overlay click-through.
	 * Without this the surface would swallow every click on the monitor,
	 * which on the OVERLAY layer means locking the user out of their
	 * desktop -- the single most dangerous failure mode in this design.
	 */
	struct wl_region *empty = wl_compositor_create_region(o->compositor);
	wl_surface_set_input_region(o->surface, empty);
	wl_region_destroy(empty);

	/* Commit with no buffer, then wait for the compositor to size us. */
	wl_surface_commit(o->surface);
	wl_display_roundtrip(o->dpy);

	if (!o->configured) {
		ERR("surface was never configured");
		goto fail;
	}
	/* A fullscreen toplevel may be configured 0x0 if the compositor defers
	 * to us; fall back to the output's own mode. */
	if (o->logical_w <= 0 || o->logical_h <= 0) {
		o->logical_w = target->mode_w;
		o->logical_h = target->mode_h;
	}
	if (o->logical_w <= 0 || o->logical_h <= 0) {
		ERR("compositor configured a %dx%d surface",
		    o->logical_w, o->logical_h);
		goto fail;
	}

	o->buf_w = o->logical_w * o->scale;
	o->buf_h = o->logical_h * o->scale;
	wl_surface_set_buffer_scale(o->surface, o->scale);

	LOG("%s surface configured: %dx%d logical, %dx%d buffer (scale %d)",
	    mode == BC_OVERLAY_LAYER ? "layer" : "toplevel",
	    o->logical_w, o->logical_h, o->buf_w, o->buf_h, o->scale);
	if (mode == BC_OVERLAY_TOPLEVEL)
		LOG("toplevel mode: needs a KWin rule on window class "
		    "\"betacensor\" forcing Layer=Overlay and "
		    "\"Hide from screenshots and screen recordings\" (see DESIGN.md §3[7])");
	/*
	 * Say what goes wrong, not just what is required. Having the rule
	 * present is not the same as having it applied, and when it is not
	 * applied nothing fails loudly -- the censor bars simply enter the
	 * captured frame and feed back into the detector, which reads as
	 * flickering and jumpy censoring with no obvious cause. That cost a
	 * long debugging session, so the failure mode is now written down
	 * where it will be seen.
	 */
	if (mode == BC_OVERLAY_TOPLEVEL) {
		LOG("  IMPORTANT: leave \"Match whole window class\" UNCHECKED. "
		    "On Wayland KWin derives the class from the app_id and the "
		    "whole-class form does not match, silently disabling every "
		    "property of the rule at once");
		LOG("  if the rule is not applied the detector sees its own bars "
		    "and the censoring will flicker; to check, pause a video and "
		    "compare 'change gate' with bars drawn against a BC_CONF=0.99 "
		    "run with none -- the two should agree (they read 4.8%% vs "
		    "79.9%% when the rule was not matching)");
	} else {
		/*
		 * The layer-shell path has no such rule to get wrong, and
		 * printing KWin instructions there sends someone off to look
		 * for a settings page their compositor does not have. It has
		 * the opposite problem: nothing can exclude this surface from
		 * the capture, so the feedback loop is real and the occlusion
		 * hold is what stands in for the rule.
		 */
		LOG("  layer-shell: no compositor here can hide this surface "
		    "from the capture, so the detector will see its own bars; "
		    "the occlusion hold is on to compensate (DESIGN.md §3[7])");
	}

	if (egl_setup(o) < 0)
		goto fail;

	return o;

fail:
	bc_overlay_destroy(o);
	return NULL;
}

int bc_overlay_peek(struct bc_overlay *o, double now_ms)
{
	if (!o)
		return 0;
	/*
	 * The clamp. Note it is checked before anything is drawn and updated
	 * only on success, so a caller that asks every frame gets exactly one
	 * peek per interval rather than a burst.
	 */
	/*
	 * Interval, with a hard floor.
	 *
	 * BC_PEEK_MS tunes how often the screen is uncovered to look at it.
	 * The default is deliberately far slower than the floor: a still
	 * picture does not change between one second and the next, so peeking
	 * twice a second buys nothing and costs a visible flash plus that
	 * fraction of the time uncovered. Faster is for content that moves.
	 *
	 * The floor is not tunable. Below it the uncovering enters the
	 * photosensitive-seizure band, and no configuration should be able to
	 * put a user there.
	 */
	static double interval = -1.0;
	if (interval < 0.0) {
		const char *e = getenv("BC_PEEK_MS");
		interval = e ? atof(e) : 2000.0;
		if (interval < BC_PEEK_MIN_MS)
			interval = BC_PEEK_MIN_MS;
	}
	if (o->last_peek_ms > 0.0 && now_ms - o->last_peek_ms < interval)
		return 0;
	if (bc_overlay_present(o, NULL, 0) != 0)
		return 0;
	o->last_peek_ms = now_ms;
	o->peeks++;
	return 1;
}

unsigned long bc_overlay_peek_count(const struct bc_overlay *o)
{
	return o ? o->peeks : 0;
}

int bc_overlay_width(const struct bc_overlay *o)  { return o ? o->buf_w : 0; }
int bc_overlay_height(const struct bc_overlay *o) { return o ? o->buf_h : 0; }

/*
 * The *output's* mode, not the surface buffer. The two differ under scaling,
 * and it is the mode that can be compared against a capture stream to ask
 * "is this the screen I am drawing on?".
 */
void bc_overlay_output_mode(const struct bc_overlay *o, int *w, int *h)
{
	if (w) *w = o ? o->out_mode_w : 0;
	if (h) *h = o ? o->out_mode_h : 0;
}

const char *bc_overlay_output_name(const struct bc_overlay *o)
{
	return (o && o->out_name[0]) ? o->out_name : "(unknown)";
}

void *bc_overlay_egl_display(const struct bc_overlay *o)
{
	return o ? o->egl_dpy : NULL;
}

/*
 * Every scissor in this file goes through here.
 *
 * A box arrives from a Kalman filter that can coast off the screen, so its
 * rectangle is not guaranteed to be sane: partly off the output, or -- when a
 * lost track's size state runs away -- inverted. GL answers a negative width
 * or height with GL_INVALID_VALUE, and that error then sits in the context
 * until the *next* glGetError() anywhere in the program picks it up. The next
 * one belongs to the DMA-BUF import, which reported it as a failed
 * glEGLImageTargetTexture2DOES, declared the capture stream lost and failed
 * the whole program closed. One bad rectangle, and censorship stopped until
 * restart -- blamed on the one part of the pipeline that was working.
 *
 * An inverted rectangle is normalised rather than dropped: it still says where
 * the detection was, and covering its absolute extent is the §2 direction.
 * Clipping to the output only removes area that was never on screen.
 *
 * Returns 0 when nothing is left to draw, in which case no scissor is set and
 * the caller must skip the box -- drawing with the previous box's scissor
 * still active would paint one detection's effect over another's rectangle.
 */
/*
 * The scissor for a box, in GL's bottom-up coordinates, grown by `pad` and --
 * this is the point -- by however much its rotation needs.
 *
 * The shader draws a properly rotated rectangle, but the scissor was always
 * the box's *unrotated* extent. A rotated rect clipped by an axis-aligned one
 * leaves their intersection, so the bar rendered as a rhombus with its corners
 * cut off rather than as a turned rectangle -- and lost coverage at exactly
 * the ends, where an eye bar can least afford it. The shader comment said the
 * scissor was the rotated rect's bounding box; it was not.
 *
 * A rectangle of half-size (hw, hh) turned by a has AABB half-extents
 *
 *   ex = hw|cos a| + hh|sin a|,   ey = hw|sin a| + hh|cos a|
 *
 * which is the same relation the eye-bar containment test uses, run in the
 * direction that grows the bound instead of shrinking the box.
 */
static void rot_scissor_box(const struct bc_box *b, int buf_h, int pad,
                            int *x, int *y, int *w, int *h)
{
	const float hw = b->w * 0.5f, hh = b->h * 0.5f;
	const float ca = fabsf(cosf(b->angle)), sa = fabsf(sinf(b->angle));
	const float ex = hw * ca + hh * sa;
	const float ey = hw * sa + hh * ca;
	const float cx = b->x + hw;
	/* Into GL's bottom-up frame before growing, so the pad is symmetric. */
	const float cy = (float)buf_h - (b->y + hh);
	*x = (int)(cx - ex) - pad;
	*y = (int)(cy - ey) - pad;
	*w = (int)(2.0f * ex) + 2 * pad + 1;
	*h = (int)(2.0f * ey) + 2 * pad + 1;
}

static int set_scissor(const struct bc_overlay *o, int x, int y, int w, int h)
{
	if (w < 0) { x += w; w = -w; }
	if (h < 0) { y += h; h = -h; }
	if (x < 0) { w += x; x = 0; }
	if (y < 0) { h += y; y = 0; }
	if (x + w > o->buf_w) w = o->buf_w - x;
	if (y + h > o->buf_h) h = o->buf_h - y;
	if (w <= 0 || h <= 0)
		return 0;
	glScissor(x, y, w, h);
	return 1;
}

void bc_overlay_set_silhouette(struct bc_overlay *o, const unsigned char *px,
                               int w, int h,
                               float u0, float v0, float u1, float v1)
{
	if (!o)
		return;

	if (!px || w <= 0 || h <= 0) {
		/* No silhouette: every box goes back to covering its whole
		 * rectangle. The texture is kept -- clearing the flag is what
		 * matters, and the next upload reuses it. */
		o->mask_tex = 0;
		return;
	}

	if (!o->mask_obj) {
		glGenTextures(1, &o->mask_obj);
		glBindTexture(GL_TEXTURE_2D, o->mask_obj);
		/* Nearest: the mask is a hard in/out test that the shader
		 * thresholds anyway, and interpolating it would only soften the
		 * edge inward -- which is coverage lost. */
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		o->mask_w = o->mask_h = 0;
	}

	glBindTexture(GL_TEXTURE_2D, o->mask_obj);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	if (w != o->mask_w || h != o->mask_h) {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0,
		             GL_RED, GL_UNSIGNED_BYTE, px);
		o->mask_w = w;
		o->mask_h = h;
	} else {
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
		                GL_RED, GL_UNSIGNED_BYTE, px);
	}

	o->mask_tex     = o->mask_obj;
	o->mask_rect[0] = u0;
	o->mask_rect[1] = v0;
	o->mask_rect[2] = u1;
	o->mask_rect[3] = v1;
}

void bc_overlay_set_background(struct bc_overlay *o, int on)
{
	if (o)
		o->bg_on = on;
}

int bc_overlay_present(struct bc_overlay *o, const struct bc_box *boxes, int n)
{
	if (!o)
		return -1;

	if (wl_display_dispatch_pending(o->dpy) < 0) {
		ERR("wayland connection lost");
		return -1;
	}
	if (o->closed) {
		LOG("compositor closed the layer surface");
		return 1;
	}

	glViewport(0, 0, o->buf_w, o->buf_h);

	/* Fully transparent everywhere we are not censoring. */
	glDisable(GL_SCISSOR_TEST);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	/*
	 * Premultiplied source-over. Needed only by the fade-out path: at
	 * alpha 1 this is identical to overwriting, so the normal opaque case
	 * is unaffected.
	 */
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	/*
	 * The delayed screen: the source frame, opaque, over the whole output and
	 * beneath every box. Under KWin an opaque surface used to stop the
	 * compositor repainting what was behind it, which starved the capture; with
	 * the overlay hidden from the screencast it does not (measured: capture
	 * rate 130/s against 141/s with nothing drawn, 0% identical frames).
	 */
	if (o->bg_on && o->bg_prog && o->src_tex) {
		glUseProgram(o->bg_prog);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, o->src_tex);
		glUniform2f(glGetUniformLocation(o->bg_prog, "uSize"),
		            (float)o->buf_w, (float)o->buf_h);
		glUniform1f(glGetUniformLocation(o->bg_prog, "uA"), 1.0f);
		glBindVertexArray(o->vao);
		glDrawArrays(GL_TRIANGLES, 0, 3);
	}

	/*
	 * Reverse mode. The mask is one box covering the whole output, drawn
	 * with the same effect and strength as the boxes would have been --
	 * Beta Blocker calls this "match style" -- and the detections are then
	 * cleared back out of it below.
	 *
	 * With no detections there is nothing to reveal, so the screen is
	 * simply masked. That is also what happens if the detector dies, which
	 * is the safe direction for this mode: fewer cutouts, not more.
	 *
	 * The style comes from o->mask and never from the boxes. Taking it from
	 * boxes[0] tied the mask's *appearance* to whether any detection existed
	 * this frame, so a single frame with none repainted the whole screen
	 * solid black and the next frame put it back -- a strobe at display
	 * rate, uncounted because the screen stayed covered throughout.
	 */
	struct bc_box full;
	const struct bc_box *list = boxes;
	int count = n;
	/*
	 * Which boxes get punched out. Normally just this frame's; in reverse
	 * mode a brief gap reuses the last set instead of blanking the screen.
	 * See o->hold.
	 */
	/*
	 * Boxes marked `over` are covers, not holes: they are kept out of the
	 * cutouts and drawn after them. Everything else is punched out as
	 * before, so a frame with no such box takes exactly the old path.
	 */
	struct bc_box over[BC_MAX_BOXES];
	struct bc_box cutbuf[BC_MAX_BOXES];
	int n_over = 0, n_keep = 0;
	for (int i = 0; i < n && i < BC_MAX_BOXES; i++) {
		if (boxes[i].over) {
			if (n_over < BC_MAX_BOXES)
				over[n_over++] = boxes[i];
		} else if (n_keep < BC_MAX_BOXES) {
			cutbuf[n_keep++] = boxes[i];
		}
	}

	const struct bc_box *cut = n_over ? cutbuf : boxes;
	int n_cut = n_over ? n_keep : n;
	if (o->reverse) {
		const double now = now_seconds();
		if (n_cut > 0) {
			o->n_hold = n_cut < BC_MAX_BOXES ? n_cut : BC_MAX_BOXES;
			memcpy(o->hold, cut,
			       (size_t)o->n_hold * sizeof *o->hold);
			o->hold_t = now;
		} else if (n == 0 && o->n_hold > 0 &&
		           now - o->hold_t < BC_REVERSE_HOLD) {
			/*
			 * Only across a real gap in detections -- `n == 0` --
			 * and not merely because this frame had no holes in it.
			 * Since a class pinned to an effect became a cover
			 * rather than a hole, "nothing to cut out" also happens
			 * on a frame that detected plenty and covered all of
			 * it. Reviving the previous frame's holes there would
			 * punch the mask open over content this frame says to
			 * cover, which is the one direction §6 forbids.
			 */
			cut   = o->hold;
			n_cut = o->n_hold;
		} else {
			o->n_hold = 0;
		}
	}

	int reverse = o->reverse && n_cut > 0;
	if (o->reverse) {
		full = o->mask;
		full.x = 0; full.y = 0;
		full.w = o->buf_w; full.h = o->buf_h;
		full.a = 1.0f;
		list  = &full;
		count = 1;
	}

	glEnable(GL_SCISSOR_TEST);
	/*
	 * Two passes, with the cutouts between them. Reverse mode is
	 * mask -> holes -> anything marked `over`, and the third phase cannot
	 * be folded into the first: it has to land on top of content the
	 * cutouts have not revealed yet. In normal mode there is no second
	 * pass -- an `over` box is already in `list`, in the right order.
	 */
	const struct bc_box *plist = list;
	int pcount = count;
	for (int pass = 0; pass < 2; pass++) {
	if (pass == 1) {
		/*
		 * On o->reverse, not on `reverse`: the covers are drawn whether
		 * or not anything was cut out. Gated on `reverse` they vanished
		 * exactly when every detected class was pinned, since that is a
		 * frame with covers and no holes at all.
		 */
		if (!o->reverse || n_over == 0)
			break;
		plist  = over;
		pcount = n_over;
	}
	for (int i = 0; i < pcount; i++) {
		const struct bc_box *b = &plist[i];
		/* GL's origin is bottom-left; bc_box is top-left. */
		int y = o->buf_h - (b->y + b->h);
		if (b->angle != 0.0f) {
			/* A rotated rect needs its axis-aligned bound as the
			 * scissor; the shader rejects the corners. */
			float c = fabsf(cosf(b->angle)), s2 = fabsf(sinf(b->angle));
			float hw = b->w * 0.5f, hh = b->h * 0.5f;
			float ex = hw * c + hh * s2, ey = hw * s2 + hh * c;
			float cx = b->x + hw, cy = (float)y + hh;
			int   sx = (int)(cx - ex), sy = (int)(cy - ey);
			int   sw = (int)(2.0f * ex) + 2, sh = (int)(2.0f * ey) + 2;
			if (!set_scissor(o, sx, sy, sw, sh))
				continue;
		} else {
			{
				int sx, sy, sw, sh;
				rot_scissor_box(b, o->buf_h, 0, &sx, &sy, &sw, &sh);
				if (!set_scissor(o, sx, sy, sw, sh))
					continue;
			}
		}

		/*
		 * Which effects need the captured screen. Noise does not sample
		 * it at all, so it stays available when the capture is stale or
		 * gone -- and fill is the fallback for the ones that do, which
		 * is the §6 rule: never show stale content in the name of an
		 * effect.
		 */
		int samples_src = (b->effect == BC_EFFECT_PIXELATE ||
		                   b->effect == BC_EFFECT_BLUR     ||
		                   b->effect == BC_EFFECT_GLITCH   ||
		                   b->effect == BC_EFFECT_RGBSHIFT ||
		                   b->effect == BC_EFFECT_HALFTONE ||
		                   b->effect == BC_EFFECT_SOBEL    ||
		                   b->effect == BC_EFFECT_TRIANGLE ||
		                   b->effect == BC_EFFECT_SHAPE);
		int self_contained = (b->effect == BC_EFFECT_NOISE    ||
		                      b->effect == BC_EFFECT_TAPE     ||
		                      b->effect == BC_EFFECT_CLUSTER  ||
		                      b->effect == BC_EFFECT_SPLATTER ||
		                      b->effect == BC_EFFECT_CAPTION  ||
		                      b->effect == BC_EFFECT_WALL);
		/*
		 * A rotated box must go through the shader whatever the effect:
		 * glClear only knows axis-aligned rectangles.
		 */
		/*
		 * An outline, a shadow or a sampled colour all have to go
		 * through the shader: glClear fills one flat rectangle and can
		 * express none of them.
		 */
		const float decor = (b->outline > 0.0f ? b->outline : 0.0f) +
		                    (b->shadow  > 0.0f ? b->shadow  : 0.0f);
		/*
		 * A silhouette-clipped box has to go through the shader whatever
		 * the effect: the clip is a per-fragment test and glClear fills
		 * the whole scissor unconditionally. Without this a masked solid
		 * fill would silently ignore its mask and draw the rectangle.
		 */
		const int masked = b->use_mask && o->mask_tex;
		int shaded = self_contained || (samples_src && o->src_tex) ||
		             b->angle != 0.0f || decor > 0.0f || masked ||
		             (b->auto_color && o->src_tex);
		if (!shaded) {
			/* Solid fill, and the fail-closed path when we have no
			 * source texture yet: still cover, just opaquely.
			 * glClear ignores blending, so it writes the faded
			 * value directly -- which is what we want, and why the
			 * colour must already be premultiplied. */
			glClearColor(b->r * b->a, b->g * b->a, b->b * b->a, b->a);
			glClear(GL_COLOR_BUFFER_BIT);
			continue;
		}

		/*
		 * Block size as a percentage of the box's short side (§5), so a
		 * small box is not censored with one giant block and a large one
		 * is not left legible. Clamped to a sane pixel range.
		 */
		int   shortest = b->w < b->h ? b->w : b->h;
		float pct      = b->strength > 0.0f ? b->strength : 8.0f;
		float block    = (float)shortest * pct / 100.0f;
		if (block < 4.0f)   block = 4.0f;
		if (block > 256.0f) block = 256.0f;

		/*
		 * Tape is drawn as a bar spanning the full width of the output
		 * rather than fitted to the box: a strip of tape stops looking
		 * like tape the moment it is exactly the size of what it hides.
		 * Widening only ever covers more, so the detection stays covered.
		 */
		if (b->effect == BC_EFFECT_TAPE)
			if (!set_scissor(o, 0, y, o->buf_w, b->h))
				continue;

		/*
		 * Static gets a soft edge; a hard rectangle of noise reads as a
		 * sticker laid on the screen. The scissor is grown to make room
		 * for the ramp, so the covered area only ever gets larger.
		 */
		float feather = (b->effect == BC_EFFECT_NOISE)
		                ? (block * 0.75f < 6.0f ? 6.0f
		                   : (block * 0.75f > 40.0f ? 40.0f : block * 0.75f))
		                : 0.0f;
		/*
		 * The outline and shadow are drawn outside the box, so the
		 * scissor has to make room for them or they are clipped away at
		 * the box edge -- which looks like the outline simply not
		 * working on three sides.
		 */
		{
			const int f = (feather > 0.0f || decor > 0.0f)
			            ? (int)(feather + decor) + 1 : 0;
			int sx, sy, sw, sh;
			rot_scissor_box(b, o->buf_h, f, &sx, &sy, &sw, &sh);
			if (!set_scissor(o, sx, sy, sw, sh))
				continue;
		}

		glUseProgram(o->pix_prog);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, o->src_tex);
		glUniform2f(o->u_size, (float)o->buf_w, (float)o->buf_h);
		glUniform1f(o->u_block, block);
		glUniform1f(o->u_alpha, b->a);
		/* Fail-closed: an effect that needs the screen, without a screen
		 * to read, degrades to a solid bar rather than showing stale
		 * pixels. Reached only when the box is rotated -- the
		 * axis-aligned case took the glClear path above. */
		glUniform1i(o->u_effect,
		            (samples_src && !o->src_tex) ? BC_EFFECT_FILL
		                                         : b->effect);
		glUniform3f(o->u_color, b->r, b->g, b->b);
		/* Fail-closed again: sampling the surroundings needs the screen,
		 * and without one the bar keeps its configured colour rather
		 * than reading whatever is stale in the texture. */
		glUniform1i(o->u_auto_col, (b->auto_color && o->src_tex) ? 1 : 0);
		glUniform1f(o->u_outline, b->outline);
		glUniform3f(o->u_out_col, b->outline_rgb[0], b->outline_rgb[1],
		            b->outline_rgb[2]);
		glUniform1f(o->u_shadow, b->shadow);
		glUniform1f(o->u_shadow_a, b->shadow_a);
		/*
		 * Caption text as code points. Non-printable characters are
		 * dropped rather than substituted: the shader blanks anything
		 * outside its range anyway, and a box of replacement glyphs
		 * would read as a bug.
		 */
		{
			GLint codes[64];
			int   n = 0;
			for (const char *p = b->text; *p && n < 64; p++)
				if ((unsigned char)*p >= 32 &&
				    (unsigned char)*p <= 126)
					codes[n++] = (unsigned char)*p;
			glUniform1iv(o->u_text, n ? n : 1, n ? codes
			                                     : (GLint[]){ 32 });
			glUniform1i(o->u_text_len, n);
		}
		glUniform1i(o->u_text_mode, b->text_mode);
		glUniform3f(o->u_text_col, b->text_rgb[0], b->text_rgb[1],
		            b->text_rgb[2]);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, o->font_tex);
		if (masked) {
			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, o->mask_tex);
		}
		glActiveTexture(GL_TEXTURE0);
		glUniform1i(o->u_use_mask, masked ? 1 : 0);
		glUniform4f(o->u_mask_rect, o->mask_rect[0], o->mask_rect[1],
		            o->mask_rect[2], o->mask_rect[3]);
		glUniform1f(o->u_time, (float)(now_seconds() - o->t0));
		glUniform4f(o->u_box, (float)b->x, (float)y,
		            (float)b->w, (float)b->h);
		glUniform1f(o->u_feather, feather);
		glUniform1f(o->u_angle, b->angle);
		glBindVertexArray(o->vao);
		glDrawArrays(GL_TRIANGLES, 0, 3);
	}

	/*
	 * Punch the detections out of the mask. glClear ignores blending and
	 * writes the value directly, so this leaves genuinely transparent
	 * holes rather than compositing toward transparent.
	 */
	if (pass == 0 && reverse) {
		glUseProgram(0);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		for (int i = 0; i < n_cut; i++) {
			const struct bc_box *b = &cut[i];
			int y = o->buf_h - (b->y + b->h);
			/*
			 * Deliberately the unrotated box, unlike every other
			 * scissor here. In reverse mode the scissor is what gets
			 * *cleared*, so it reveals rather than covers -- and
			 * glClear cannot express a rotated rect, so widening this
			 * to the rotated bounding box would uncover the corners
			 * the rotation was supposed to keep covered.
			 */
			if (!set_scissor(o, b->x, y, b->w, b->h))
				continue;
			glClear(GL_COLOR_BUFFER_BIT);
		}
	}
	}   /* pass */

	glDisable(GL_SCISSOR_TEST);
	glDisable(GL_BLEND);
	glUseProgram(0);

	/*
	 * BC_DEBUG_MASK=1: what did we actually draw?
	 *
	 * The overlay is hidden from screen capture by a compositor rule, which
	 * also hides it from every tool that could be used to check it -- so
	 * short of looking at the monitor there is otherwise no way to tell a
	 * black screen from a pixelated one, and "the whole screen went black"
	 * is exactly the class of bug that matters most here. Reads back one
	 * frame in sixty and reports coverage and brightness.
	 *
	 * Numbers only, never an image: this buffer holds the user's screen.
	 */
	const char *debug_mask = getenv("BC_DEBUG_MASK");
	if (debug_mask) {
		/*
		 * Every frame, and free: a flash *is* a change in the cutouts,
		 * and one lasting a frame or two is invisible to the sampled
		 * readback below -- which reads 1 frame in 60, so at 180 Hz it
		 * sees three a second and can report a rock-steady screen
		 * through a visible strobe.
		 */
		static int prev_cut = -1;
		if (n_cut != prev_cut) {
			LOG("cutouts %d -> %d", prev_cut, n_cut);
			prev_cut = n_cut;
		}

		/* BC_DEBUG_MASK=<n>: read back every nth frame (default 60). */
		int every = atoi(debug_mask);
		if (every < 1)
			every = 60;
		static int nth;
		if (++nth % every == 0) {
			const int W = o->buf_w, H = o->buf_h;
			unsigned char *px = malloc((size_t)W * H * 4);
			if (px) {
				glReadPixels(0, 0, W, H, GL_RGBA,
				             GL_UNSIGNED_BYTE, px);
				long black = 0, opaque = 0;
				double lum = 0;
				for (long i = 0; i < (long)W * H; i++) {
					int r = px[i*4], g = px[i*4+1],
					    b = px[i*4+2], a = px[i*4+3];
					if (a > 250) opaque++;
					if (a > 250 && r < 8 && g < 8 && b < 8)
						black++;
					lum += (r + g + b) / 3.0;
				}
				LOG("mask probe: opaque %.1f%%  pure-black %.1f%%  "
				    "mean luma %.1f", 100.0 * opaque / ((double)W*H),
				    100.0 * black / ((double)W*H), lum / ((double)W*H));
				free(px);
			}
		}
	}

	if (!eglSwapBuffers(o->egl_dpy, o->egl_surf)) {
		ERR("eglSwapBuffers failed (0x%04x)", eglGetError());
		return -1;
	}
	return 0;
}

void bc_overlay_destroy(struct bc_overlay *o)
{
	if (!o)
		return;

	if (o->egl_dpy != EGL_NO_DISPLAY && o->egl_dpy != NULL) {
		if (o->vao)
			glDeleteVertexArrays(1, &o->vao);
		if (o->font_tex)
			glDeleteTextures(1, &o->font_tex);
		if (o->mask_obj)
			glDeleteTextures(1, &o->mask_obj);
		if (o->pix_prog)
			glDeleteProgram(o->pix_prog);
		eglMakeCurrent(o->egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE,
		               EGL_NO_CONTEXT);
		if (o->egl_surf) eglDestroySurface(o->egl_dpy, o->egl_surf);
		if (o->egl_ctx)  eglDestroyContext(o->egl_dpy, o->egl_ctx);
		eglTerminate(o->egl_dpy);
	}
	if (o->egl_window) wl_egl_window_destroy(o->egl_window);
	if (o->ptr)  wl_pointer_destroy(o->ptr);
	if (o->kbd)  wl_keyboard_destroy(o->kbd);
	if (o->seat) wl_seat_destroy(o->seat);

	if (o->xdg_toplevel) xdg_toplevel_destroy(o->xdg_toplevel);
	if (o->xdg_surface)  xdg_surface_destroy(o->xdg_surface);
	if (o->wm_base)      xdg_wm_base_destroy(o->wm_base);
	if (o->layer)      zwlr_layer_surface_v1_destroy(o->layer);
	if (o->surface)    wl_surface_destroy(o->surface);
	if (o->shell)      zwlr_layer_shell_v1_destroy(o->shell);
	if (o->compositor) wl_compositor_destroy(o->compositor);

	for (int i = 0; i < o->n_outputs; i++)
		if (o->outputs[i].wl)
			wl_output_destroy(o->outputs[i].wl);

	if (o->registry) wl_registry_destroy(o->registry);
	if (o->dpy) {
		wl_display_flush(o->dpy);
		wl_display_disconnect(o->dpy);
	}
	free(o);
}
