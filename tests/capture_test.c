/*
 * The DMA-BUF modifier negotiation, driven without a compositor.
 *
 * This exists because of a bug that could not be reproduced on the machine
 * that wrote it. The middle step of the protocol -- server hands the whole
 * modifier set back, client picks one and re-announces -- was skipped, and
 * KWin tolerates that by settling on the default itself. A wlroots compositor
 * on Mesa answers the same shortcut with "error alloc buffers: Invalid
 * argument", several steps from the actual mistake.
 *
 * A test that needed a compositor to fail would not have caught it either,
 * since the only compositor available here is the one that tolerates it.
 *
 * So the pods a producer would send are built here by hand and fed through the
 * same decision the stream callback makes. No PipeWire connection, no GPU, no
 * compositor: the thing under test is the protocol logic, and that is pure.
 */
#include <spa/pod/builder.h>
#include <spa/pod/parser.h>
#include <spa/param/video/format-utils.h>
#include <drm_fourcc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* From capture.c. Declared here rather than in a header so the test does not
 * drag the capture interface -- and its PipeWire types -- into every caller. */
uint64_t bc_capture_choose_modifier(const uint64_t *mods, uint32_t n);
struct spa_pod *bc_capture_build_format_fixed(struct spa_pod_builder *b,
                                              const struct spa_video_info_raw *info,
                                              uint64_t modifier);

/*
 * capture.c is linked whole, so the symbols it calls have to exist. None of
 * these run: the test never opens a stream. Stubs rather than linking gpu.c,
 * which would pull in EGL and a display connection.
 */
struct bc_gpu;
struct bc_dmabuf;
int  bc_gpu_import(struct bc_gpu *g, const void *key, const struct bc_dmabuf *b)
{ (void)g; (void)key; (void)b; return -1; }
void bc_gpu_forget(struct bc_gpu *g, const void *key) { (void)g; (void)key; }
int  bc_gpu_query_modifiers(struct bc_gpu *g, uint32_t fourcc,
                            uint64_t *out, int max)
{ (void)g; (void)fourcc; (void)out; (void)max; return 0; }
int  bc_gpu_dump_ppm(struct bc_gpu *g, const char *path)
{ (void)g; (void)path; return -1; }

static int failures;

static void check(int cond, const char *what)
{
	printf("  %-58s %s\n", what, cond ? "ok" : "FAIL");
	if (!cond)
		failures++;
}

/* Some real modifiers, so the values in the test are the values in the wild. */
#define MOD_LINEAR   ((uint64_t)0)                     /* DRM_FORMAT_MOD_LINEAR */
#define MOD_AMD_DCC  ((uint64_t)0x020000001046bb04ull) /* from the AMD report */
#define MOD_NV_TILED ((uint64_t)0x0300000000fe0014ull) /* an NVIDIA layout */

/*
 * A Format as a producer sends it *before* the modifier is settled: the
 * modifier property is a choice, flagged DONT_FIXATE. Mirrors what capture.c
 * offers, because the server echoes the shape back.
 */
static struct spa_pod *build_unfixated(struct spa_pod_builder *b,
                                       const uint64_t *mods, int n)
{
	struct spa_pod_frame f[2];

	spa_pod_builder_push_object(b, &f[0], SPA_TYPE_OBJECT_Format,
	                            SPA_PARAM_Format);
	spa_pod_builder_add(b,
		SPA_FORMAT_mediaType,    SPA_POD_Id(SPA_MEDIA_TYPE_video),
		SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
		SPA_FORMAT_VIDEO_format, SPA_POD_Id(SPA_VIDEO_FORMAT_BGRx),
		0);

	spa_pod_builder_prop(b, SPA_FORMAT_VIDEO_modifier,
	                     SPA_POD_PROP_FLAG_MANDATORY |
	                     SPA_POD_PROP_FLAG_DONT_FIXATE);
	spa_pod_builder_push_choice(b, &f[1], SPA_CHOICE_Enum, 0);
	spa_pod_builder_long(b, (int64_t)mods[0]);          /* default */
	for (int i = 0; i < n; i++)
		spa_pod_builder_long(b, (int64_t)mods[i]);
	spa_pod_builder_pop(b, &f[1]);

	spa_pod_builder_add(b,
		SPA_FORMAT_VIDEO_size,      SPA_POD_Rectangle(&SPA_RECTANGLE(2560, 1600)),
		SPA_FORMAT_VIDEO_framerate, SPA_POD_Fraction(&SPA_FRACTION(60, 1)),
		0);

	return spa_pod_builder_pop(b, &f[0]);
}

int main(void)
{
	printf("capture: DMA-BUF modifier negotiation\n\n");

	/* ---------------------------------------------------------------- */
	printf("choosing a modifier from the server's set\n");
	{
		unsetenv("BC_MODIFIER");
		unsetenv("BC_NO_MODIFIERS");

		const uint64_t only_invalid[] = { DRM_FORMAT_MOD_INVALID };
		check(bc_capture_choose_modifier(only_invalid, 1) == DRM_FORMAT_MOD_INVALID,
		      "nothing but the implicit path -> implicit");

		/* The reported case: our appended INVALID sits in the list, and
		 * taking it would throw away the tiling the import needs. */
		const uint64_t amd[] = { MOD_AMD_DCC, MOD_LINEAR, DRM_FORMAT_MOD_INVALID };
		check(bc_capture_choose_modifier(amd, 3) == MOD_AMD_DCC,
		      "a real modifier wins over the implicit fallback");

		const uint64_t invalid_first[] = { DRM_FORMAT_MOD_INVALID, MOD_AMD_DCC };
		check(bc_capture_choose_modifier(invalid_first, 2) == MOD_AMD_DCC,
		      "...even when the implicit entry is listed first");

		/* LINEAR is 0, which is a modifier and not an absence of one --
		 * a test worth having, because it is the value most likely to
		 * be mistaken for "unset" by a future edit. */
		const uint64_t linear_first[] = { MOD_LINEAR, MOD_AMD_DCC };
		check(bc_capture_choose_modifier(linear_first, 2) == MOD_LINEAR,
		      "LINEAR (0) is a real modifier, not an empty one");

		check(bc_capture_choose_modifier(NULL, 0) == DRM_FORMAT_MOD_INVALID,
		      "an empty set -> implicit");
	}

	printf("\noverrides, for a machine we cannot test from here\n");
	{
		const uint64_t mods[] = { MOD_AMD_DCC, MOD_LINEAR, MOD_NV_TILED };

		setenv("BC_MODIFIER", "0x0", 1);
		check(bc_capture_choose_modifier(mods, 3) == MOD_LINEAR,
		      "BC_MODIFIER picks one out of the offered set");

		setenv("BC_MODIFIER", "0xdeadbeef", 1);
		check(bc_capture_choose_modifier(mods, 3) == MOD_AMD_DCC,
		      "BC_MODIFIER naming one not on offer falls back, not fails");
		unsetenv("BC_MODIFIER");

		setenv("BC_NO_MODIFIERS", "1", 1);
		check(bc_capture_choose_modifier(mods, 3) == DRM_FORMAT_MOD_INVALID,
		      "BC_NO_MODIFIERS forces the implicit path");
		unsetenv("BC_NO_MODIFIERS");
	}

	/* ---------------------------------------------------------------- */
	printf("\nreading an unfixated Format, as a wlroots producer sends it\n");
	{
		uint8_t buf[4096];
		struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buf, sizeof buf);
		const uint64_t mods[] = { MOD_AMD_DCC, MOD_LINEAR };
		struct spa_pod *pod = build_unfixated(&b, mods, 2);

		const struct spa_pod_prop *mp =
			spa_pod_find_prop(pod, NULL, SPA_FORMAT_VIDEO_modifier);
		check(mp != NULL, "the modifier property is present");
		check(mp && (mp->flags & SPA_POD_PROP_FLAG_DONT_FIXATE),
		      "DONT_FIXATE is set -- the set is ours to narrow");

		uint32_t n_vals = 0, choice = 0;
		const struct spa_pod *vals =
			spa_pod_get_values(&mp->value, &n_vals, &choice);
		const uint64_t *got = (const uint64_t *)SPA_POD_BODY_CONST(vals);

		/* The default is repeated as the first alternative, so a
		 * two-modifier offer arrives as three values. Reading the count
		 * wrong is how a list gets silently truncated to its default. */
		check(n_vals == 3, "values = default + each alternative");
		check(got[0] == MOD_AMD_DCC && got[1] == MOD_AMD_DCC &&
		      got[2] == MOD_LINEAR, "the values read back in order");
		check(bc_capture_choose_modifier(got, n_vals) == MOD_AMD_DCC,
		      "and the choice made from them is the tiled one");
	}

	/* ---------------------------------------------------------------- */
	printf("\nre-announcing with the modifier fixated\n");
	{
		uint8_t buf[4096];
		struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buf, sizeof buf);
		struct spa_video_info_raw info = {
			.format    = SPA_VIDEO_FORMAT_BGRx,
			.size      = SPA_RECTANGLE(2560, 1600),
			.framerate = SPA_FRACTION(60, 1),
		};

		struct spa_pod *pod =
			bc_capture_build_format_fixed(&b, &info, MOD_AMD_DCC);

		const struct spa_pod_prop *mp =
			spa_pod_find_prop(pod, NULL, SPA_FORMAT_VIDEO_modifier);
		check(mp != NULL, "the modifier is named");
		check(mp && !(mp->flags & SPA_POD_PROP_FLAG_DONT_FIXATE),
		      "DONT_FIXATE is cleared -- this is an answer, not an offer");
		check(mp && mp->value.type == SPA_TYPE_Long,
		      "a plain value, not a choice");

		struct spa_video_info parsed = { 0 };
		check(spa_format_parse(pod, &parsed.media_type,
		                       &parsed.media_subtype) >= 0 &&
		      spa_format_video_raw_parse(pod, &parsed.info.raw) >= 0,
		      "the pod parses back as a video format");
		check(parsed.info.raw.modifier == MOD_AMD_DCC,
		      "carrying the modifier we chose");
		check(parsed.info.raw.size.width == 2560 &&
		      parsed.info.raw.size.height == 1600,
		      "and the size the server already settled on");

		/* The implicit path must not name a modifier at all -- naming
		 * INVALID would be asking for a layout that does not exist. */
		struct spa_pod_builder b2 = SPA_POD_BUILDER_INIT(buf, sizeof buf);
		struct spa_pod *implicit =
			bc_capture_build_format_fixed(&b2, &info,
			                              DRM_FORMAT_MOD_INVALID);
		check(spa_pod_find_prop(implicit, NULL,
		                        SPA_FORMAT_VIDEO_modifier) == NULL,
		      "the implicit path omits the property entirely");
	}

	printf("\n%s\n", failures ? "FAILURES" : "All checks passed.");
	return failures ? 1 : 0;
}
