#include "capture.h"
#include "gpu.h"
#include "util.h"

#include <pipewire/pipewire.h>
#include <spa/param/video/format-utils.h>
#include <spa/param/video/type-info.h>
#include <spa/utils/result.h>
#include <drm_fourcc.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

#define MAX_MODIFIERS 64

struct bc_capture {
	struct pw_loop      *loop;
	struct pw_context   *context;
	struct pw_core      *core;
	struct pw_stream    *stream;
	struct spa_hook      stream_listener;
	int                  entered;

	struct spa_video_info_raw info;
	int                  have_modifier;
	uint64_t             modifier;

	struct bc_gpu       *gpu;
	/*
	 * When the frame we last imported was composited, from the buffer
	 * header, in CLOCK_MONOTONIC ns. Measured 2.4-14.6 ms behind the wall
	 * clock on this machine: a captured frame always describes a moment
	 * already past, which is why "was the overlay up when this was taken"
	 * cannot be answered by looking at the clock now.
	 */
	int64_t              frame_pts;
	int                  new_frame;   /* set by on_process, cleared by poll */
	uint64_t             frames;
	int                  failed;

	/*
	 * The buffer whose pixels the pipeline is currently reading.
	 *
	 * Queueing a buffer back tells the compositor it may draw into it
	 * again, so the frame we are still sampling has to stay dequeued. It
	 * used to be released at the end of on_process, while the EGLImage, the
	 * detector's preprocess pass and every one of the ~180 presents per
	 * second that followed all went on reading it -- so the overlay was
	 * compositing against whatever the compositor happened to be painting
	 * there next. Holding one buffer of the eight costs nothing and makes
	 * the read valid for as long as it lasts.
	 */
	struct pw_buffer    *held;
};

/* SPA video format -> DRM fourcc. SPA names are byte-order-reversed. */
static uint32_t spa_to_fourcc(uint32_t spa_format)
{
	switch (spa_format) {
	case SPA_VIDEO_FORMAT_BGRx: return DRM_FORMAT_XRGB8888;
	case SPA_VIDEO_FORMAT_BGRA: return DRM_FORMAT_ARGB8888;
	case SPA_VIDEO_FORMAT_RGBx: return DRM_FORMAT_XBGR8888;
	case SPA_VIDEO_FORMAT_RGBA: return DRM_FORMAT_ABGR8888;
	default:                    return DRM_FORMAT_INVALID;
	}
}

/*
 * Build one EnumFormat offering `spa_format`, advertising every modifier the
 * driver told us it can import (DESIGN.md §3[2]).
 *
 * DONT_FIXATE is required: it tells the server we are offering a *set* of
 * modifiers and expect it to pick one. Without it, negotiation with a
 * modifier-aware compositor fails.
 */
static struct spa_pod *build_format(struct spa_pod_builder *b,
                                    uint32_t spa_format,
                                    const uint64_t *mods, int n_mods)
{
	struct spa_pod_frame f[2];

	spa_pod_builder_push_object(b, &f[0],
	                            SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat);
	spa_pod_builder_add(b,
		SPA_FORMAT_mediaType,    SPA_POD_Id(SPA_MEDIA_TYPE_video),
		SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
		SPA_FORMAT_VIDEO_format, SPA_POD_Id(spa_format),
		0);

	if (n_mods > 0) {
		spa_pod_builder_prop(b, SPA_FORMAT_VIDEO_modifier,
		                     SPA_POD_PROP_FLAG_MANDATORY |
		                     SPA_POD_PROP_FLAG_DONT_FIXATE);
		spa_pod_builder_push_choice(b, &f[1], SPA_CHOICE_Enum, 0);
		spa_pod_builder_long(b, (int64_t)mods[0]); /* default */
		for (int i = 0; i < n_mods; i++)
			spa_pod_builder_long(b, (int64_t)mods[i]);
		spa_pod_builder_pop(b, &f[1]);
	}

	spa_pod_builder_add(b,
		SPA_FORMAT_VIDEO_size, SPA_POD_CHOICE_RANGE_Rectangle(
			&SPA_RECTANGLE(2560, 1440),
			&SPA_RECTANGLE(1, 1),
			&SPA_RECTANGLE(8192, 8192)),
		SPA_FORMAT_VIDEO_framerate, SPA_POD_CHOICE_RANGE_Fraction(
			&SPA_FRACTION(60, 1),
			&SPA_FRACTION(0, 1),
			&SPA_FRACTION(240, 1)),
		0);

	return spa_pod_builder_pop(b, &f[0]);
}

/*
 * One EnumFormat naming exactly one modifier, fixated.
 *
 * The counterpart to build_format(): that one says "here is a set, you pick",
 * this one says "I have picked, here it is". The difference is the absence of
 * DONT_FIXATE and the value being a plain long rather than a choice. Size and
 * framerate are pinned to what the server already chose, so this renegotiation
 * settles the modifier and nothing else.
 */
/* Non-static, and prefixed, so tests/capture_test.c can drive them. */
struct spa_pod *bc_capture_build_format_fixed(struct spa_pod_builder *b,
                                              const struct spa_video_info_raw *info,
                                              uint64_t modifier)
{
	struct spa_pod_frame f;

	spa_pod_builder_push_object(b, &f,
	                            SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat);
	spa_pod_builder_add(b,
		SPA_FORMAT_mediaType,    SPA_POD_Id(SPA_MEDIA_TYPE_video),
		SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
		SPA_FORMAT_VIDEO_format, SPA_POD_Id(info->format),
		0);

	if (modifier != DRM_FORMAT_MOD_INVALID) {
		spa_pod_builder_prop(b, SPA_FORMAT_VIDEO_modifier,
		                     SPA_POD_PROP_FLAG_MANDATORY);
		spa_pod_builder_long(b, (int64_t)modifier);
	}

	spa_pod_builder_add(b,
		SPA_FORMAT_VIDEO_size,      SPA_POD_Rectangle(&info->size),
		SPA_FORMAT_VIDEO_framerate, SPA_POD_Fraction(&info->framerate),
		0);

	return spa_pod_builder_pop(b, &f);
}

/*
 * Which of the server's offered modifiers to take.
 *
 * The list arrives in the server's preference order and has already been
 * intersected with what we advertised, so the first entry is the best answer
 * available -- except that DRM_FORMAT_MOD_INVALID (the implicit-modifier path)
 * is a last resort we appended ourselves, and taking it when a real modifier
 * is on offer throws away the tiling information the import needs.
 *
 * BC_MODIFIER=<hex> forces one and BC_NO_MODIFIERS=1 forces the implicit path,
 * because this is precisely the decision that cannot be tested from here: the
 * modifier set is a property of the tester's driver.
 */
uint64_t bc_capture_choose_modifier(const uint64_t *mods, uint32_t n)
{
	const char *env = getenv("BC_MODIFIER");
	if (env) {
		uint64_t want = strtoull(env, NULL, 0);
		for (uint32_t i = 0; i < n; i++)
			if (mods[i] == want)
				return want;
		WARN("BC_MODIFIER=0x%llx is not on offer -- ignoring",
		     (unsigned long long)want);
	}
	if (getenv("BC_NO_MODIFIERS"))
		return DRM_FORMAT_MOD_INVALID;

	for (uint32_t i = 0; i < n; i++)
		if (mods[i] != DRM_FORMAT_MOD_INVALID)
			return mods[i];

	return DRM_FORMAT_MOD_INVALID;
}

static void on_param_changed(void *data, uint32_t id, const struct spa_pod *param)
{
	struct bc_capture *c = data;

	if (!param || id != SPA_PARAM_Format)
		return;

	struct spa_video_info info = { 0 };
	if (spa_format_parse(param, &info.media_type, &info.media_subtype) < 0)
		return;
	if (info.media_type != SPA_MEDIA_TYPE_video ||
	    info.media_subtype != SPA_MEDIA_SUBTYPE_raw)
		return;
	if (spa_format_video_raw_parse(param, &info.info.raw) < 0)
		return;

	c->info = info.info.raw;

	const struct spa_pod_prop *mp =
		spa_pod_find_prop(param, NULL, SPA_FORMAT_VIDEO_modifier);

	/*
	 * The modifier may still be a *set*, not an answer.
	 *
	 * This is the middle step of the DMA-BUF negotiation: we offer a choice
	 * flagged DONT_FIXATE, the server hands the surviving set back, and we
	 * are expected to name one and re-announce before asking for buffers.
	 * Skipping it -- replying to the set with ParamBuffers, which is what
	 * this did -- is a protocol violation that KWin happens to tolerate: it
	 * settles on the default itself and allocates anyway. Measured here,
	 * KWin offers 17 modifiers and never once complained, which is exactly
	 * why the bug survived. A wlroots compositor on Mesa answers the same
	 * shortcut with "error alloc buffers: Invalid argument" on the producer
	 * side, several steps from the actual mistake.
	 *
	 * So: if the set comes back, pick one and re-announce. The server then
	 * sends another Format, fixated, and that is the one that gets buffers.
	 */
	if (mp && (mp->flags & SPA_POD_PROP_FLAG_DONT_FIXATE)) {
		uint32_t n_vals = 0, choice = 0;
		const struct spa_pod *vals =
			spa_pod_get_values(&mp->value, &n_vals, &choice);
		const uint64_t *mods = (const uint64_t *)SPA_POD_BODY_CONST(vals);

		if (!vals || n_vals == 0) {
			ERR("server offered an empty modifier set");
			c->failed = 1;
			return;
		}

		uint64_t picked = bc_capture_choose_modifier(mods, n_vals);
		LOG("format: %dx%d spa_fmt=%u -- server offered %u modifier(s), "
		    "fixating on 0x%016llx",
		    c->info.size.width, c->info.size.height, c->info.format,
		    n_vals, (unsigned long long)picked);

		uint8_t buf[1024];
		struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buf, sizeof buf);
		const struct spa_pod *params[1];
		params[0] = bc_capture_build_format_fixed(&b, &c->info, picked);
		pw_stream_update_params(c->stream, params, 1);
		return;
	}

	c->have_modifier = mp != NULL;
	if (mp)
		c->modifier = c->info.modifier;

	LOG("format: %dx%d spa_fmt=%u modifier=%s0x%016llx",
	    c->info.size.width, c->info.size.height, c->info.format,
	    c->have_modifier ? "" : "(none) ",
	    (unsigned long long)(c->have_modifier ? c->modifier
	                                          : DRM_FORMAT_MOD_INVALID));

	/*
	 * Demand DMA-BUF buffers. Refusing MemFd/MemPtr here is deliberate:
	 * §0 says a silent fall back to shared memory is worse than a hard
	 * failure, because it would quietly cost 14.7 MB per frame over PCIe.
	 *
	 * `blocks` is a *range*, not 1. A block is a plane, and how many planes
	 * a buffer has is a property of the modifier the server just chose:
	 * NVIDIA's are single-plane, but an AMD modifier with DCC carries a
	 * metadata plane alongside the colour one. Demanding exactly one made
	 * the request unsatisfiable for those, which the producer reports as
	 * the same "Invalid argument" as everything else. The import path
	 * already reads n_datas and handles up to BC_MAX_PLANES.
	 */
	uint8_t buf[1024];
	struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buf, sizeof buf);
	const struct spa_pod *params[2];

	params[0] = spa_pod_builder_add_object(&b,
		SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
		SPA_PARAM_BUFFERS_buffers,  SPA_POD_CHOICE_RANGE_Int(8, 2, 16),
		SPA_PARAM_BUFFERS_blocks,
			SPA_POD_CHOICE_RANGE_Int(1, 1, BC_MAX_PLANES),
		SPA_PARAM_BUFFERS_dataType,
			SPA_POD_CHOICE_FLAGS_Int(1 << SPA_DATA_DmaBuf));

	/*
	 * Ask for the frame header, which carries the timestamp the frame was
	 * composited at. A server attaches only the metadata a client requests,
	 * so without this every buffer arrives anonymous -- and "when was this
	 * frame taken" is the question that decides whether a captured frame
	 * shows the screen with our overlay on it or without.
	 */
	params[1] = spa_pod_builder_add_object(&b,
		SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
		SPA_PARAM_META_type, SPA_POD_Id(SPA_META_Header),
		SPA_PARAM_META_size, SPA_POD_Int(sizeof(struct spa_meta_header)));

	pw_stream_update_params(c->stream, params, 2);
}

static void on_process(void *data)
{
	struct bc_capture *c = data;

	/*
	 * Drain to the newest buffer. If we have fallen behind, older frames are
	 * stale, and censoring against a stale frame puts boxes where the content
	 * no longer is -- import the latest and drop the rest.
	 */
	struct pw_buffer *newest = NULL, *b;
	while ((b = pw_stream_dequeue_buffer(c->stream))) {
		if (newest)
			pw_stream_queue_buffer(c->stream, newest);
		newest = b;
	}
	if (!newest)
		return;

	struct spa_buffer *sb = newest->buffer;
	if (sb->n_datas < 1)
		goto drop;

	/*
	 * BC_TRACE_PTS=n: report the frame timestamp for the first n frames,
	 * against CLOCK_MONOTONIC now. Whether this field is populated, and in
	 * which clock, decides whether a captured frame can be matched to the
	 * overlay frame it was taken over -- see DESIGN.md §11.
	 */
	{
		struct spa_meta_header *h0 = spa_buffer_find_meta_data(
			sb, SPA_META_Header, sizeof *h0);
		c->frame_pts = h0 ? (int64_t)h0->pts : 0;
	}
	{
		static int pts_left = -1;
		if (pts_left < 0) {
			const char *e = getenv("BC_TRACE_PTS");
			pts_left = e ? atoi(e) : 0;
		}
		if (pts_left > 0) {
			pts_left--;
			struct spa_meta_header *hd = spa_buffer_find_meta_data(
				sb, SPA_META_Header, sizeof *hd);
			struct timespec ts;
			clock_gettime(CLOCK_MONOTONIC, &ts);
			const int64_t now_ns = (int64_t)ts.tv_sec * 1000000000
			                     + ts.tv_nsec;
			if (!hd) {
				LOG("pts: no SPA_META_Header on this buffer");
			} else {
				LOG("pts: %" PRId64 " ns, now %" PRId64
				    ", age %.2f ms, seq %" PRIu64 ", flags 0x%x",
				    (int64_t)hd->pts, now_ns,
				    (double)(now_ns - (int64_t)hd->pts) / 1e6,
				    (uint64_t)hd->seq, hd->flags);
			}
		}
	}

	if (sb->datas[0].type != SPA_DATA_DmaBuf) {
		ERR("server gave us buffer type %u, not DmaBuf — refusing "
		    "(see DESIGN.md §0)", sb->datas[0].type);
		c->failed = 1;
		goto drop;
	}

	struct bc_dmabuf dbuf = { 0 };
	dbuf.n_planes = (int)sb->n_datas > BC_MAX_PLANES
	                ? BC_MAX_PLANES : (int)sb->n_datas;
	dbuf.width    = (int)c->info.size.width;
	dbuf.height   = (int)c->info.size.height;
	dbuf.fourcc   = spa_to_fourcc(c->info.format);
	dbuf.modifier = c->have_modifier ? c->modifier : DRM_FORMAT_MOD_INVALID;

	if (dbuf.fourcc == DRM_FORMAT_INVALID) {
		ERR("unmapped SPA video format %u", c->info.format);
		c->failed = 1;
		goto drop;
	}

	for (int i = 0; i < dbuf.n_planes; i++) {
		dbuf.fd[i]     = (int)sb->datas[i].fd;
		dbuf.offset[i] = sb->datas[i].chunk->offset;
		dbuf.stride[i] = (uint32_t)sb->datas[i].chunk->stride;
	}

	/*
	 * Keyed on the spa_buffer, so the EGLImage and texture are made once
	 * per buffer rather than once per frame. The producer cycles the same
	 * eight, so after the first pass this is a lookup and nothing else.
	 */
	if (bc_gpu_import(c->gpu, sb, &dbuf) < 0) {
		c->failed = 1;
		goto drop;
	}

	if (c->frames == 0)
		LOG("frame: %dx%d planes=%d fourcc=0x%08x mod=0x%016llx stride=%u",
		    dbuf.width, dbuf.height, dbuf.n_planes, dbuf.fourcc,
		    (unsigned long long)dbuf.modifier, dbuf.stride[0]);

	c->frames++;
	c->new_frame = 1;

	/*
	 * Release the frame we have been reading until now -- not the one just
	 * taken, which everything downstream is about to read.
	 *
	 * By the time control returns here the caller has finished with the
	 * previous frame: bc_capture_poll() runs on the same thread as the GL
	 * work, so that frame's preprocess pass and its present were both
	 * issued before this callback started. What remains is the GPU's own
	 * queue, and one whole inter-frame period is ample for it -- the
	 * zero-copy submission path glFinish()es inside that window anyway.
	 */
	if (c->held)
		pw_stream_queue_buffer(c->stream, c->held);
	c->held = newest;
	return;

drop:
	pw_stream_queue_buffer(c->stream, newest);
}

/*
 * The producer is taking a buffer back. Whatever we cached against it aliases
 * memory that is about to stop being ours to read, so it goes now.
 *
 * There is deliberately no add_buffer counterpart: offset and stride live in
 * the chunk, which the producer only fills in once it has actually put a frame
 * in the buffer, so an import done at add_buffer time would describe a layout
 * that is not set yet. Importing lazily on first sight costs one EGLImage per
 * buffer and reads the layout the producer really used.
 */
static void on_remove_buffer(void *data, struct pw_buffer *buffer)
{
	struct bc_capture *c = data;

	/* Once removed it is no longer ours to queue back. */
	if (c->held == buffer)
		c->held = NULL;
	if (buffer->buffer)
		bc_gpu_forget(c->gpu, buffer->buffer);
}

static void on_state_changed(void *data, enum pw_stream_state old,
                             enum pw_stream_state state, const char *error)
{
	struct bc_capture *c = data;

	LOG("stream state: %s -> %s%s%s",
	    pw_stream_state_as_string(old), pw_stream_state_as_string(state),
	    error ? " : " : "", error ? error : "");

	if (state == PW_STREAM_STATE_ERROR)
		c->failed = 1;
}

static const struct pw_stream_events stream_events = {
	PW_VERSION_STREAM_EVENTS,
	.state_changed = on_state_changed,
	.param_changed = on_param_changed,
	.remove_buffer = on_remove_buffer,
	.process       = on_process,
};

struct bc_capture *bc_capture_start(int pw_fd, uint32_t node_id,
                                    struct bc_gpu *gpu)
{
	struct bc_capture *c = calloc(1, sizeof *c);
	if (!c)
		return NULL;
	c->gpu = gpu;

	pw_init(NULL, NULL);

	c->loop = pw_loop_new(NULL);
	if (!c->loop) {
		ERR("pw_loop_new failed");
		goto fail;
	}

	c->context = pw_context_new(c->loop, NULL, 0);
	if (!c->context) {
		ERR("pw_context_new failed");
		goto fail;
	}

	/* Takes ownership of pw_fd. */
	c->core = pw_context_connect_fd(c->context, pw_fd, NULL, 0);
	if (!c->core) {
		ERR("pw_context_connect_fd failed");
		goto fail;
	}

	c->stream = pw_stream_new(c->core, "betacensor-capture",
		pw_properties_new(
			PW_KEY_MEDIA_TYPE,     "Video",
			PW_KEY_MEDIA_CATEGORY, "Capture",
			PW_KEY_MEDIA_ROLE,     "Screen",
			NULL));
	if (!c->stream) {
		ERR("pw_stream_new failed");
		goto fail;
	}

	pw_stream_add_listener(c->stream, &c->stream_listener, &stream_events, c);

	/* Offer the formats we can import, each with the driver's modifier list. */
	static const uint32_t want[] = {
		SPA_VIDEO_FORMAT_BGRx, SPA_VIDEO_FORMAT_BGRA,
		SPA_VIDEO_FORMAT_RGBx, SPA_VIDEO_FORMAT_RGBA,
	};

	uint8_t pod_buf[8192];
	struct spa_pod_builder b = SPA_POD_BUILDER_INIT(pod_buf, sizeof pod_buf);
	const struct spa_pod *params[8];
	unsigned n_params = 0;

	for (unsigned i = 0; i < sizeof want / sizeof want[0]; i++) {
		uint32_t fourcc = spa_to_fourcc(want[i]);
		uint64_t mods[MAX_MODIFIERS];
		int n = bc_gpu_query_modifiers(gpu, fourcc, mods, MAX_MODIFIERS);

		/* Always allow the implicit-modifier path as a last resort. */
		if (n < MAX_MODIFIERS)
			mods[n++] = DRM_FORMAT_MOD_INVALID;

		params[n_params++] = build_format(&b, want[i], mods, n);
	}

	int rc = pw_stream_connect(c->stream, PW_DIRECTION_INPUT, node_id,
	                           PW_STREAM_FLAG_AUTOCONNECT, params, n_params);
	if (rc < 0) {
		ERR("pw_stream_connect: %s", spa_strerror(rc));
		goto fail;
	}

	pw_loop_enter(c->loop);
	c->entered = 1;

	LOG("capture stream connected to node %u", node_id);
	return c;

fail:
	bc_capture_stop(c);
	return NULL;
}

int bc_capture_poll(struct bc_capture *c)
{
	if (!c || c->failed)
		return -1;

	/* timeout 0: never block here; the overlay's vsync paces the loop. */
	int rc = pw_loop_iterate(c->loop, 0);
	if (rc < 0) {
		ERR("pw_loop_iterate: %s", spa_strerror(rc));
		return -1;
	}
	if (c->failed)
		return -1;

	if (c->new_frame) {
		c->new_frame = 0;
		return 1;
	}
	return 0;
}

uint64_t bc_capture_frames(const struct bc_capture *c)
{
	return c ? c->frames : 0;
}

void bc_capture_stop(struct bc_capture *c)
{
	if (!c)
		return;
	if (c->held && c->stream) {
		pw_stream_queue_buffer(c->stream, c->held);
		c->held = NULL;
	}
	if (c->entered)  pw_loop_leave(c->loop);
	if (c->stream)   pw_stream_destroy(c->stream);
	if (c->core)     pw_core_disconnect(c->core);
	if (c->context)  pw_context_destroy(c->context);
	if (c->loop)     pw_loop_destroy(c->loop);
	free(c);
	pw_deinit();
}

int bc_capture_one(int pw_fd, uint32_t node_id, struct bc_gpu *gpu,
                   const char *out_path)
{
	struct bc_capture *c = bc_capture_start(pw_fd, node_id, gpu);
	if (!c)
		return -1;

	int rc = -1;
	for (;;) {
		/* Blocking iterate; main.c's SIGALRM bounds the total wait. */
		if (pw_loop_iterate(c->loop, 200) < 0 || c->failed)
			break;
		if (c->new_frame) {
			rc = bc_gpu_dump_ppm(gpu, out_path);
			if (rc == 0)
				LOG("wrote %s", out_path);
			break;
		}
	}

	bc_capture_stop(c);
	return rc;
}

int64_t bc_capture_frame_pts(const struct bc_capture *c)
{
	return c ? c->frame_pts : 0;
}
