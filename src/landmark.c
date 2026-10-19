#include "landmark.h"
#include "util.h"

#include <onnxruntime_c_api.h>

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/*
 * A separate ORT session rather than a second bc_detector.
 *
 * bc_detect owns a worker thread and a YOLO decoder, and this model is neither
 * -- it emits raw heatmaps and is wanted synchronously, for the faces in the
 * frame being built. Bolting a second output convention onto that decoder
 * would put two unrelated shapes behind one interface; the session setup it
 * would have shared is thirty lines.
 */
struct bc_landmark {
	const OrtApi       *api;
	OrtEnv             *env;
	OrtSessionOptions  *opts;
	OrtSession         *sess;
	OrtMemoryInfo      *mem;

	char  *in_name, *out_name;
	int    side;          /* input is side x side */
	int    hm_w, hm_h;    /* heatmap resolution */
	int    points;
	int    on_gpu;
	double last_ms;
};

static double now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int lm_fail(struct bc_landmark *l, OrtStatus *st, const char *what)
{
	if (!st)
		return 0;
	ERR("landmark: %s: %s", what, l->api->GetErrorMessage(st));
	l->api->ReleaseStatus(st);
	return 1;
}

void bc_landmark_destroy(struct bc_landmark *l)
{
	if (!l)
		return;
	if (l->api) {
		OrtAllocator *alloc = NULL;
		OrtStatus *st = l->api->GetAllocatorWithDefaultOptions(&alloc);
		if (st)
			l->api->ReleaseStatus(st);
		if (alloc) {
			if (l->in_name)  alloc->Free(alloc, l->in_name);
			if (l->out_name) alloc->Free(alloc, l->out_name);
		}
		if (l->mem)  l->api->ReleaseMemoryInfo(l->mem);
		if (l->sess) l->api->ReleaseSession(l->sess);
		if (l->opts) l->api->ReleaseSessionOptions(l->opts);
		if (l->env)  l->api->ReleaseEnv(l->env);
	}
	free(l);
}

struct bc_landmark *bc_landmark_create(const char *model_path, int threads)
{
	struct bc_landmark *l = calloc(1, sizeof *l);
	if (!l)
		return NULL;
	l->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

	if (lm_fail(l, l->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bc-landmark",
	                                 &l->env), "CreateEnv"))
		goto fail;
	if (lm_fail(l, l->api->CreateSessionOptions(&l->opts), "CreateSessionOptions"))
		goto fail;
	if (lm_fail(l, l->api->SetIntraOpNumThreads(l->opts, threads),
	            "SetIntraOpNumThreads"))
		goto fail;
	if (lm_fail(l, l->api->SetSessionGraphOptimizationLevel(l->opts,
	                                                        ORT_ENABLE_ALL),
	            "SetGraphOptimizationLevel"))
		goto fail;

	/* Same best-effort CUDA as the detector: a missing provider means slower,
	 * never "the censor does not start". */
	if (!getenv("BC_NO_CUDA")) {
		OrtCUDAProviderOptions cuda = { 0 };
		cuda.device_id = 0;
		OrtStatus *st = l->api->SessionOptionsAppendExecutionProvider_CUDA(
			l->opts, &cuda);
		if (st) {
			WARN("landmark: CUDA EP unavailable (%s) -- CPU",
			     l->api->GetErrorMessage(st));
			l->api->ReleaseStatus(st);
		} else {
			l->on_gpu = 1;
		}
	}

	if (lm_fail(l, l->api->CreateSession(l->env, model_path, l->opts, &l->sess),
	            "CreateSession"))
		goto fail;
	if (lm_fail(l, l->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
	                                           &l->mem), "CreateCpuMemoryInfo"))
		goto fail;

	OrtAllocator *alloc = NULL;
	if (lm_fail(l, l->api->GetAllocatorWithDefaultOptions(&alloc),
	            "GetAllocator"))
		goto fail;
	if (lm_fail(l, l->api->SessionGetInputName(l->sess, 0, alloc, &l->in_name),
	            "SessionGetInputName"))
		goto fail;
	if (lm_fail(l, l->api->SessionGetOutputName(l->sess, 0, alloc, &l->out_name),
	            "SessionGetOutputName"))
		goto fail;

	/*
	 * Read the geometry from the model rather than assuming it. A different
	 * model loaded here would not fail -- it would decode whatever came out
	 * into confident-looking eye positions, and the bar would sit somewhere
	 * arbitrary with nothing in the logs to say why.
	 */
	OrtTypeInfo *ti = NULL;
	const OrtTensorTypeAndShapeInfo *si = NULL;
	int64_t shape[8];
	size_t dims = 0;

	if (lm_fail(l, l->api->SessionGetInputTypeInfo(l->sess, 0, &ti),
	            "SessionGetInputTypeInfo"))
		goto fail;
	if (!lm_fail(l, l->api->CastTypeInfoToTensorInfo(ti, &si), "CastTypeInfo") &&
	    !lm_fail(l, l->api->GetDimensionsCount(si, &dims), "GetDimensionsCount") &&
	    dims == 4 &&
	    !lm_fail(l, l->api->GetDimensions(si, shape, 4), "GetDimensions"))
		l->side = (int)shape[3];
	l->api->ReleaseTypeInfo(ti);

	ti = NULL; si = NULL; dims = 0;
	if (lm_fail(l, l->api->SessionGetOutputTypeInfo(l->sess, 0, &ti),
	            "SessionGetOutputTypeInfo"))
		goto fail;
	if (!lm_fail(l, l->api->CastTypeInfoToTensorInfo(ti, &si), "CastTypeInfo") &&
	    !lm_fail(l, l->api->GetDimensionsCount(si, &dims), "GetDimensionsCount") &&
	    dims == 4 &&
	    !lm_fail(l, l->api->GetDimensions(si, shape, 4), "GetDimensions")) {
		l->points = (int)shape[1];
		l->hm_h   = (int)shape[2];
		l->hm_w   = (int)shape[3];
	}
	l->api->ReleaseTypeInfo(ti);

	if (l->side <= 0 || l->points != BC_LM_POINTS ||
	    l->hm_w <= 0 || l->hm_h <= 0) {
		ERR("landmark: '%s' is not a %d-point heatmap model "
		    "(input %d, output %dx%dx%d) -- refusing it rather than "
		    "decoding noise into eye positions",
		    model_path, BC_LM_POINTS, l->side, l->points, l->hm_h, l->hm_w);
		goto fail;
	}

	LOG("landmark: %s  %dx%d -> %d points on %dx%d heatmaps  [%s]",
	    model_path, l->side, l->side, l->points, l->hm_w, l->hm_h,
	    l->on_gpu ? "CUDA" : "CPU");
	return l;

fail:
	bc_landmark_destroy(l);
	return NULL;
}

int bc_landmark_input(const struct bc_landmark *l) { return l ? l->side : 0; }
double bc_landmark_last_ms(const struct bc_landmark *l) { return l ? l->last_ms : 0.0; }
int bc_landmark_on_gpu(const struct bc_landmark *l) { return l ? l->on_gpu : 0; }

/*
 * Argmax per channel, nudged a quarter cell toward the brighter neighbour.
 *
 * mmpose decodes these with DARK, which fits a Gaussian to a blurred heatmap
 * for sub-pixel accuracy. The quarter-step is the cheap standard alternative
 * and is enough here: the output is an eye *centre* averaged over six points
 * and an angle taken across the face, so a fraction of a heatmap cell -- about
 * a quarter of one input pixel in four -- moves neither. Verified against the
 * Python decode when the model was exported.
 */
static void decode(const struct bc_landmark *l, const float *hm,
                   struct bc_lm_face *f)
{
	const int n = l->hm_w * l->hm_h;

	for (int k = 0; k < l->points; k++) {
		const float *m = hm + (size_t)k * n;
		int best = 0;
		for (int i = 1; i < n; i++)
			if (m[i] > m[best])
				best = i;
		const int bx = best % l->hm_w, by = best / l->hm_w;
		float px = (float)bx, py = (float)by;

		if (bx > 0 && bx < l->hm_w - 1) {
			const float d = m[best + 1] - m[best - 1];
			px += d > 0.0f ? 0.25f : (d < 0.0f ? -0.25f : 0.0f);
		}
		if (by > 0 && by < l->hm_h - 1) {
			const float d = m[best + l->hm_w] - m[best - l->hm_w];
			py += d > 0.0f ? 0.25f : (d < 0.0f ? -0.25f : 0.0f);
		}

		/* Heatmap cell -> crop pixel. */
		f->x[k]     = (px + 0.5f) / (float)l->hm_w * (float)l->side;
		f->y[k]     = (py + 0.5f) / (float)l->hm_h * (float)l->side;
		f->score[k] = m[best];
	}

	float sx = 0, sy = 0, ss = 0;
	for (int k = BC_LM_LEYE_FIRST; k <= BC_LM_LEYE_LAST; k++) {
		sx += f->x[k]; sy += f->y[k]; ss += f->score[k];
	}
	const float ln = (float)(BC_LM_LEYE_LAST - BC_LM_LEYE_FIRST + 1);
	f->lx = sx / ln; f->ly = sy / ln;

	float rx = 0, ry = 0, rs = 0;
	for (int k = BC_LM_REYE_FIRST; k <= BC_LM_REYE_LAST; k++) {
		rx += f->x[k]; ry += f->y[k]; rs += f->score[k];
	}
	const float rn = (float)(BC_LM_REYE_LAST - BC_LM_REYE_FIRST + 1);
	f->rx = rx / rn; f->ry = ry / rn;

	f->eye_score = (ss + rs) / (ln + rn);

	/*
	 * How far the eye points actually reach, in the eyes' own frame. The
	 * bar is built from these rather than from the centre separation, so it
	 * covers the eyes that are there instead of the eyes a photographic
	 * face would have had.
	 */
	const float mx = (f->lx + f->rx) * 0.5f;
	const float my = (f->ly + f->ry) * 0.5f;
	float ax = f->rx - f->lx, ay = f->ry - f->ly;
	const float alen = sqrtf(ax * ax + ay * ay);
	if (alen > 1e-3f) {
		ax /= alen; ay /= alen;
	} else {
		ax = 1.0f; ay = 0.0f;   /* degenerate: treat as horizontal */
	}
	f->ext_along = f->ext_across = 0.0f;
	for (int k = BC_LM_LEYE_FIRST; k <= BC_LM_REYE_LAST; k++) {
		const float dx = f->x[k] - mx, dy = f->y[k] - my;
		const float pa = dx * ax + dy * ay;        /* along  */
		const float pc = -dx * ay + dy * ax;       /* across */
		const float aa = pa < 0.0f ? -pa : pa;
		const float ac = pc < 0.0f ? -pc : pc;
		if (aa > f->ext_along)  f->ext_along  = aa;
		if (ac > f->ext_across) f->ext_across = ac;
	}
}

int bc_landmark_run(struct bc_landmark *l, const float *tensor, int n,
                    struct bc_lm_face *out, int max)
{
	if (!l || !tensor || n <= 0)
		return -1;
	if (n > max)
		n = max;

	const double t0 = now_ms();
	const size_t floats = (size_t)3 * l->side * l->side;
	const size_t hm_n   = (size_t)l->points * l->hm_w * l->hm_h;
	int done = 0;

	/*
	 * One face per Run rather than a batch. The exported graph has a fixed
	 * batch of 1, and faces on screen are few -- batching would mean
	 * re-exporting with a dynamic axis to save a handful of launches.
	 */
	for (int i = 0; i < n; i++) {
		const int64_t shape[4] = { 1, 3, l->side, l->side };
		OrtValue *in = NULL, *outv = NULL;

		if (lm_fail(l, l->api->CreateTensorWithDataAsOrtValue(
		                l->mem, (void *)(tensor + (size_t)i * floats),
		                floats * sizeof(float), shape, 4,
		                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in),
		            "CreateTensorWithDataAsOrtValue"))
			break;

		const char *ins[]  = { l->in_name };
		const char *outs[] = { l->out_name };
		OrtStatus *st = l->api->Run(l->sess, NULL, ins,
		                            (const OrtValue *const *)&in, 1,
		                            outs, 1, &outv);
		l->api->ReleaseValue(in);
		if (lm_fail(l, st, "Run"))
			break;

		float *hm = NULL;
		if (lm_fail(l, l->api->GetTensorMutableData(outv, (void **)&hm),
		            "GetTensorMutableData") || !hm) {
			l->api->ReleaseValue(outv);
			break;
		}
		(void)hm_n;
		decode(l, hm, &out[done++]);
		l->api->ReleaseValue(outv);
	}

	l->last_ms = now_ms() - t0;
	return done;
}
