#include <math.h>
#include "detect.h"
#include "util.h"

#include <onnxruntime_c_api.h>

#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#define CONF_DEFAULT   0.35f
#define IOU_THRESHOLD  0.45f
#define MAX_CANDIDATES 512

/*
 * Instances whose masks are accumulated into one silhouette per frame. Each
 * costs a crop-sized pass over the prototypes, so this bounds the work; the
 * union means anything dropped was almost certainly already covered.
 */
#define BC_SEG_MAX_INSTANCES 32

/*
 * Confidence floor, overridable with BC_CONF.
 *
 * Measured on this machine: at 0.20 the model fires on ordinary desktop
 * content (a terminal produced sporadic BUTTOCKS_EXPOSED at 0.20-0.24), so
 * that is at the noise floor rather than below it. 0.35 clears the observed
 * false positives while staying well under the usual 0.5, which respects §2's
 * bias that a false negative is the failure that actually matters.
 */
static float conf_threshold(void)
{
	const char *e = getenv("BC_CONF");
	if (e) {
		float v = strtof(e, NULL);
		if (v > 0.0f && v < 1.0f)
			return v;
	}
	return CONF_DEFAULT;
}

/*
 * Read out of the model's own metadata rather than trusting a remembered
 * taxonomy. Getting this order wrong would censor faces and leave genitals
 * uncovered, which is the exact failure the program exists to prevent.
 */
static char *CLASS_NAMES[BC_MAX_CLASSES] = {
	"FEMALE_GENITALIA_COVERED", "FACE_FEMALE",     "BUTTOCKS_EXPOSED",
	"FEMALE_BREAST_EXPOSED",    "FEMALE_GENITALIA_EXPOSED",
	"MALE_BREAST_EXPOSED",      "ANUS_EXPOSED",    "FEET_EXPOSED",
	"BELLY_COVERED",            "FEET_COVERED",    "ARMPITS_COVERED",
	"ARMPITS_EXPOSED",          "FACE_MALE",       "BELLY_EXPOSED",
	"MALE_GENITALIA_EXPOSED",   "ANUS_COVERED",    "FEMALE_BREAST_COVERED",
	"BUTTOCKS_COVERED",
};
static int  N_CLASSES = 18;
static int  CLASS_OWNED;   /* names were strdup'd from metadata, so free them */

int bc_class_count(void) { return N_CLASSES; }

/*
 * Is the table above the loaded model's own, or still the built-in guess?
 *
 * Only matters to whoever has to decide whether a class name is unknown or
 * merely not-yet-known: the config is read once before any detector exists, and
 * at that point every name outside the 18 below looks foreign when it may be
 * the real model's.
 */
int bc_class_from_model(void) { return CLASS_OWNED; }

/*
 * Ultralytics stores the taxonomy in an ONNX metadata entry called "names",
 * formatted as a Python dict literal: {0: 'FEMALE_FACE', 1: 'MALE_FACE', ...}.
 * Parse it positionally by index rather than by order of appearance -- the
 * index is the channel the model actually emits, and trusting position would
 * reintroduce exactly the misalignment this exists to prevent.
 */
static int parse_names(const char *meta)
{
	char *found[BC_MAX_CLASSES] = { 0 };
	int   max_idx = -1;
	const char *p = meta;

	while ((p = strchr(p, ':'))) {
		/* Walk back over the index digits that precede the colon. */
		const char *d = p;
		while (d > meta && (d[-1] == ' ' || d[-1] == '\t'))
			d--;
		const char *end = d;
		while (d > meta && d[-1] >= '0' && d[-1] <= '9')
			d--;
		if (d == end) { p++; continue; }

		int idx = atoi(d);
		const char *q = strchr(p, '\'');
		if (!q) break;
		const char *r = strchr(++q, '\'');
		if (!r) break;

		if (idx >= 0 && idx < BC_MAX_CLASSES && !found[idx]) {
			size_t n = (size_t)(r - q);
			char  *v = malloc(n + 1);
			if (!v)
				goto fail;  /* everything parsed so far is ours */
			memcpy(v, q, n);
			v[n] = 0;
			/* Normalise case: every other model names its classes in
			 * upper case, and the config keys, the FACE/EYE tests and
			 * the similar-class pairs all assume it. A model that says
			 * 'face' must not read as a different class. */
			for (size_t k = 0; k < n; k++)
				v[k] = (char)toupper((unsigned char)v[k]);
			found[idx] = v;
			if (idx > max_idx)
				max_idx = idx;
		}
		p = r + 1;
	}

	/* Every index from 0..max must be present, or the table has a hole and
	 * some channel would map to the wrong name. Reject rather than guess. */
	for (int i = 0; i <= max_idx; i++)
		if (!found[i])
			goto fail;
	if (max_idx < 0)
		return 0;

	if (CLASS_OWNED)
		for (int i = 0; i < N_CLASSES; i++)
			free(CLASS_NAMES[i]);
	for (int i = 0; i <= max_idx; i++)
		CLASS_NAMES[i] = found[i];
	N_CLASSES   = max_idx + 1;
	CLASS_OWNED = 1;
	return N_CLASSES;

fail:
	/* Nothing was published, so every name parsed so far is still ours. */
	for (int i = 0; i < BC_MAX_CLASSES; i++)
		free(found[i]);
	return 0;
}

/*
 * The index a metadata "names" dict gives a class, without touching the global
 * table. For a model whose taxonomy is deliberately not adopted -- the COCO
 * segmentation model, whose `person` is nothing to do with the censor's own
 * classes -- this is the only way to ask what channel a name sits on.
 * Returns -1 if the name is absent.
 */
static int meta_index_of(const char *meta, const char *want)
{
	if (!meta || !want)
		return -1;

	const char *p = meta;
	while ((p = strchr(p, ':'))) {
		const char *d = p;
		while (d > meta && (d[-1] == ' ' || d[-1] == '\t'))
			d--;
		const char *end = d;
		while (d > meta && d[-1] >= '0' && d[-1] <= '9')
			d--;
		if (d == end) { p++; continue; }

		const int idx = atoi(d);
		const char *q = strchr(p, '\'');
		if (!q) break;
		const char *r = strchr(++q, '\'');
		if (!r) break;

		const size_t n = (size_t)(r - q);
		if (strlen(want) == n && !strncasecmp(q, want, n))
			return idx;
		p = r + 1;
	}
	return -1;
}

const char *bc_class_name(int cls)
{
	return (cls >= 0 && cls < N_CLASSES) ? CLASS_NAMES[cls] : "?";
}

/*
 * Split a class name on '_' into up to MAX_TOK uppercase tokens.
 * Returns the token count, or -1 if it does not fit.
 */
#define MAX_TOK 6
static int tokenize(const char *s, char tok[MAX_TOK][32])
{
	int n = 0, j = 0;
	for (;; s++) {
		if (*s == '_' || *s == 0) {
			if (j) {
				if (n >= MAX_TOK)
					return -1;
				tok[n++][j] = 0;
				j = 0;
			}
			if (*s == 0)
				return n;
		} else {
			if (n >= MAX_TOK || j >= 31)
				return -1;
			tok[n][j++] = toupper((unsigned char)*s);
		}
	}
}

/*
 * Do two class names denote the same thing under a different word order?
 *
 * The models disagree on this: the 320n/640m taxonomy writes FACE_FEMALE while
 * every y11-derived model (Hotscreen's included) writes FEMALE_FACE. Compared
 * with strcasecmp those are simply different classes, so a config written for
 * one model silently failed to apply to the other -- and since a class absent
 * from the config keeps its default of *off*, swapping models quietly stopped
 * censoring faces. Nothing warned: the key looked unknown, which is exactly
 * what a typo looks like.
 *
 * Comparing the token *set* rather than the string makes word order irrelevant
 * without a hand-maintained alias table, so a future model that reorders some
 * other name is handled too. It is deliberately only a reordering: differing
 * tokens (COVERED vs EXPOSED) stay different classes, which is the distinction
 * the whole taxonomy turns on.
 */
static int names_alias(const char *a, const char *b)
{
	char ta[MAX_TOK][32], tb[MAX_TOK][32];
	int na = tokenize(a, ta), nb = tokenize(b, tb);
	if (na <= 0 || na != nb)
		return 0;

	int used[MAX_TOK] = { 0 };
	for (int i = 0; i < na; i++) {
		int hit = 0;
		for (int j = 0; j < nb; j++) {
			if (!used[j] && !strcmp(ta[i], tb[j])) {
				used[j] = 1;
				hit = 1;
				break;
			}
		}
		if (!hit)
			return 0;
	}
	return 1;
}

/*
 * Compare ignoring '_' and case. Some taxonomies write the same class without
 * separators -- puryfi's YOLO26s says BELLYEXPOSED where NudeNet says
 * BELLY_EXPOSED -- and to a token comparison those are one token versus two, so
 * names_alias() cannot see they are the same. Squashing the separator makes the
 * existing config keys and the built-in defaults apply to such a model with no
 * per-model list, which is the difference between it censoring correctly and it
 * censoring nothing.
 */
static int names_squash_equal(const char *a, const char *b)
{
	while (*a && *b) {
		if (*a == '_') { a++; continue; }
		if (*b == '_') { b++; continue; }
		if (toupper((unsigned char)*a) != toupper((unsigned char)*b))
			return 0;
		a++; b++;
	}
	while (*a == '_') a++;
	while (*b == '_') b++;
	return !*a && !*b;
}

int bc_class_index(const char *name)
{
	if (!name)
		return -1;
	for (int i = 0; i < N_CLASSES; i++)
		if (!strcasecmp(name, CLASS_NAMES[i]))
			return i;
	/* Exact first, so an exact match always wins over a reordering. */
	for (int i = 0; i < N_CLASSES; i++)
		if (names_alias(name, CLASS_NAMES[i]))
			return i;
	for (int i = 0; i < N_CLASSES; i++)
		if (names_squash_equal(name, CLASS_NAMES[i]))
			return i;
	return -1;
}

struct bc_detector {
	const OrtApi   *api;
	OrtEnv         *env;
	OrtSessionOptions *opts;
	int    on_gpu;      /* CUDA EP accepted; CPU fallback if not */
	OrtSession     *session;
	OrtMemoryInfo  *mem;
	/*
	 * The same, describing device memory. Non-NULL only on a CUDA session:
	 * it is what lets an input tensor be built over a pointer ORT must not
	 * try to read with the CPU.
	 */
	OrtMemoryInfo  *dev_mem;
	char           *in_name;
	char           *out_name;
	/*
	 * A segmentation head's second output, the mask prototypes. NULL on every
	 * other model, and that is how "is this a seg model" is decided -- by what
	 * the session actually exposes rather than by the file's name.
	 */
	char           *proto_name;

	int    net_w, net_h;
	int    n_classes;   /* this model's own count; not always the global table */
	int    n_kpt;       /* keypoints per box; 0 unless the model has a Pose head */
	int    end2end;     /* model does its own NMS and emits final boxes */
	size_t floats;
	float  conf;        /* guarded by lock; may change while running */
	float  work_conf;   /* snapshot taken with the tensor, worker-private */

	/* Worker plumbing. */
	pthread_t       thread;
	pthread_mutex_t lock;
	pthread_cond_t  cv;
	pthread_cond_t  idle_cv;    /* signalled when a run finishes */
	int             running;
	int             pending;    /* a tensor is waiting to be run */
	int             busy;       /* worker is mid-inference */
	int             ready;      /* an outcome is waiting to be collected */
	int             ready_failed; /* ...and it is BC_DET_FAILED */

	/*
	 * Submitted tensors and the copy the worker owns while running. Each
	 * holds `tiles` consecutive tensors of `floats` each.
	 */
	int             tiles;
	float          *in_buf;
	float          *work_buf;

	/*
	 * Or, when the tensor never left VRAM, the device pointer standing in
	 * for them. Non-NULL means "read from here instead"; it is swapped
	 * alongside the host buffers so the worker's view cannot change under
	 * it mid-run.
	 */
	const void     *in_dev;
	const void     *work_dev;

	struct bc_det   results[BC_MAX_DET];
	int             n_results;

	/*
	 * Segmentation, when the model has that head.
	 *
	 * What is published is not one mask per detection but a single silhouette
	 * covering the whole frame: every detection's mask thresholded, cropped to
	 * its own box and OR-ed together. That is what the caller actually wants --
	 * a censor asks "is this pixel on a subject", not "which subject" -- and it
	 * keeps masks out of the box path entirely, where a tracked box is eased,
	 * predicted and merged and a mask tied to one inference's geometry would
	 * have no honest way to follow it.
	 *
	 * In network (letterboxed) coordinates at prototype resolution, one byte
	 * per pixel, 0 or 255. Double-buffered like the detections: `work_mask` is
	 * the worker's, `mask` is what a taker sees.
	 */
	int             n_mask;         /* mask coefficients per detection; 0 = not seg */
	int             mask_w, mask_h; /* prototype resolution, e.g. 160x160 */
	unsigned char  *mask;           /* published, guarded by lock */
	unsigned char  *work_mask;      /* worker-private */
	int             mask_built;     /* this run produced one */
	int             mask_ready;     /* a published mask is waiting to be taken */
	/*
	 * Which of the seg model's own classes to take silhouettes from, or -1
	 * for all of them. Resolved from that model's metadata at create time --
	 * its taxonomy is COCO and deliberately not adopted into the global
	 * table, so this cannot go through bc_class_index().
	 */
	int             mask_cls;

	double          last_ms;    /* of the outcome waiting in `ready` */
	unsigned long   count;
	unsigned long   errors;     /* runs that produced no usable output */
	double          fault_err_at; /* BC_FAULT_DETERR: fail runs from here */
};

static double now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int ort_fail(struct bc_detector *d, OrtStatus *st, const char *what)
{
	if (!st)
		return 0;
	ERR("%s: %s", what, d->api->GetErrorMessage(st));
	d->api->ReleaseStatus(st);
	return 1;
}

/* --- decode ---------------------------------------------------------------- */

static float iou(const struct bc_det *a, const struct bc_det *b)
{
	float x1 = a->x > b->x ? a->x : b->x;
	float y1 = a->y > b->y ? a->y : b->y;
	float x2 = (a->x + a->w) < (b->x + b->w) ? (a->x + a->w) : (b->x + b->w);
	float y2 = (a->y + a->h) < (b->y + b->h) ? (a->y + a->h) : (b->y + b->h);

	float iw = x2 - x1, ih = y2 - y1;
	if (iw <= 0.0f || ih <= 0.0f)
		return 0.0f;

	float inter = iw * ih;
	float uni   = a->w * a->h + b->w * b->h - inter;
	return uni > 0.0f ? inter / uni : 0.0f;
}

static int det_cmp(const void *pa, const void *pb)
{
	const struct bc_det *a = pa, *b = pb;
	return (a->score < b->score) - (a->score > b->score);  /* descending */
}

static int no_wbf(void) { return getenv("BC_NO_WBF") != NULL; }

/*
 * Greedy NMS, per class: overlapping boxes of different classes are both
 * legitimate (a breast inside a torso, say). Kept behind BC_NO_WBF as the
 * comparison arm for the fusion below, which replaced it.
 */
static int nms_greedy(const struct bc_det *cand, int n,
                      struct bc_det *dst, int max)
{
	int  kept = 0;
	char suppressed[MAX_CANDIDATES] = { 0 };

	for (int i = 0; i < n && kept < max; i++) {
		if (suppressed[i])
			continue;
		dst[kept++] = cand[i];
		for (int j = i + 1; j < n; j++) {
			if (suppressed[j] || cand[j].cls != cand[i].cls)
				continue;
			if (iou(&cand[i], &cand[j]) > IOU_THRESHOLD)
				suppressed[j] = 1;
		}
	}
	return kept;
}

/*
 * Weighted Box Fusion (Solovyev, Wang & Gabruseva 2021), in place of NMS.
 *
 * NMS answers "which of these boxes is right?" by keeping the best-scoring one
 * and deleting the rest. That is a hard choice between near-identical
 * candidates, and the model re-makes it every inference: when two boxes for one
 * object sit close in score, the winner alternates, and the kept box jumps
 * between their two geometries. Downstream that reads as a box that grows and
 * shrinks on content which never moved -- the edges are eased inward but jump
 * outward instantly, so an alternating winner pumps the box outward on every
 * swap and lets it creep back between them.
 *
 * WBF answers "where is this object?" instead: the cluster is *averaged*,
 * weighted by score, so a candidate winning by a hair moves the result by a
 * hair. The output is a box no single candidate proposed, which is the point --
 * it is the consensus of all of them, and consensus does not flicker when the
 * ranking does.
 *
 * Two deliberate departures from the paper, both because this is a censor:
 *
 * [1] The fused score is the cluster *maximum*, not the mean. The paper averages
 *     because it is scoring a detection benchmark, where a confident box
 *     agreeing with a weak one genuinely is less certain. Here the score decides
 *     whether content gets covered: averaging would let a strong detection be
 *     dragged below the threshold by a weak neighbour and uncover something that
 *     was correctly found. A cluster is at least as certain as its best member.
 *
 * [2] Keypoints are taken from the highest-scoring member rather than averaged.
 *     An average of two keypoint sets that disagree is a point on neither face.
 *
 * Clusters are matched against their current fused box, per the paper, and only
 * within a class -- same rule as the NMS it replaces.
 */
static int wbf(const struct bc_det *cand, int n, struct bc_det *dst, int max)
{
	/* Score-weighted coordinate sums, in corner form, per cluster. */
	struct acc {
		float sx1, sy1, sx2, sy2;   /* sum(score * coord) */
		float wsum;                 /* sum(score), the weight total */
	};
	struct acc acc[BC_MAX_DET];
	int nc = 0;

	for (int i = 0; i < n; i++) {
		const float w = cand[i].score > 0.0f ? cand[i].score : 1e-6f;
		int hit = -1;

		for (int c = 0; c < nc; c++) {
			if (dst[c].cls != cand[i].cls)
				continue;
			if (iou(&dst[c], &cand[i]) > IOU_THRESHOLD) {
				hit = c;
				break;
			}
		}

		if (hit < 0) {
			if (nc >= max)
				continue;      /* full: drop the weakest tail */
			hit          = nc++;
			dst[hit]     = cand[i];
			acc[hit]     = (struct acc){ 0, 0, 0, 0, 0 };
		} else if (cand[i].score > dst[hit].score) {
			/* Cannot happen -- candidates arrive sorted -- but if the
			 * ordering ever changes, keep [1] and [2] true. */
			dst[hit].score = cand[i].score;
			dst[hit].n_kpt = cand[i].n_kpt;
			for (int k = 0; k < cand[i].n_kpt; k++) {
				dst[hit].kx[k] = cand[i].kx[k];
				dst[hit].ky[k] = cand[i].ky[k];
				dst[hit].ks[k] = cand[i].ks[k];
			}
		}

		acc[hit].sx1  += w * cand[i].x;
		acc[hit].sy1  += w * cand[i].y;
		acc[hit].sx2  += w * (cand[i].x + cand[i].w);
		acc[hit].sy2  += w * (cand[i].y + cand[i].h);
		acc[hit].wsum += w;

		const float iw = 1.0f / acc[hit].wsum;
		const float x1 = acc[hit].sx1 * iw, y1 = acc[hit].sy1 * iw;
		const float x2 = acc[hit].sx2 * iw, y2 = acc[hit].sy2 * iw;
		dst[hit].x = x1;
		dst[hit].y = y1;
		dst[hit].w = x2 - x1;
		dst[hit].h = y2 - y1;
	}
	return nc;
}

/*
 * End-to-end head: output is [1, max_det, 6], row-major, each row
 * (x1, y1, x2, y2, score, class) in network pixels. The model has already run
 * its own NMS, so there is nothing to suppress here and no sigmoid to apply --
 * doing either would be a porting bug in the other direction.
 *
 * Rows are emitted in descending score and padded with zero-score entries, so
 * the first row below the threshold ends the list.
 */
static int decode_end2end(const float *out, int rows, int stride,
                          struct bc_det *dst, int max, float conf,
                          int n_classes)
{
	/*
	 * -1, not 0. A layout we cannot read says nothing about the screen, and
	 * 0 would be published as "nothing here" -- counted as a successful run,
	 * so the inference deadline never fires and every bar comes off.
	 */
	if (stride != 6) {
		static int warned;
		if (!warned++)
			ERR("unexpected end2end row width %d (want 6)", stride);
		return -1;
	}
	int n = 0;
	for (int i = 0; i < rows && n < max; i++) {
		const float *r = out + (size_t)i * 6;
		float score = r[4];
		if (score < conf)
			break;
		/* Round, do not truncate. The class arrives as a float, and a
		 * value that came back as 4.9999 rather than 5.0 would read as
		 * class 4 -- a silent mislabel, which under a per-class policy
		 * means covering the wrong thing or nothing at all. */
		int cls = (int)lrintf(r[5]);
		if (cls < 0 || cls >= n_classes)
			continue;
		dst[n++] = (struct bc_det){
			.x = r[0], .y = r[1],
			.w = r[2] - r[0], .h = r[3] - r[1],
			.score = score, .cls = cls
		};
	}
	return n;
}

/*
 * YOLOv8 head: output is [1, 4 + n_classes, n_anchors], channel-major, boxes
 * as centre/size in network pixels and class scores already through sigmoid.
 * There is no objectness channel -- multiplying by one is a common porting bug.
 */
static int decode(const float *out, int channels, int anchors,
                  struct bc_det *dst, int max, float conf, int n_classes,
                  int n_kpt, int n_mask)
{
	/* n_mask trailing channels are the segmentation coefficients. They sit
	 * past the class scores and are read by build_mask(), not here, but they
	 * still have to be accounted for or the shape check rejects the model.
	 *
	 * A mismatch is a failed run (-1), never an empty one: see
	 * decode_end2end(). It is how a model with more than BC_MAX_CLASSES
	 * classes, or with no readable 'names' metadata (so the built-in table's
	 * count is used), presents -- and returning 0 censored nothing while
	 * looking healthy. Logged once; the error count carries the rest. */
	if (channels != 4 + n_classes + 3 * n_kpt + n_mask) {
		static int warned;
		if (!warned++)
			ERR("unexpected output channels %d (want %d for a %d-class, "
			    "%d-keypoint, %d-mask model) -- every run will fail",
			    channels, 4 + n_classes + 3 * n_kpt + n_mask,
			    n_classes, n_kpt, n_mask);
		return -1;
	}

	struct bc_det cand[MAX_CANDIDATES];
	int n = 0;

	for (int a = 0; a < anchors && n < MAX_CANDIDATES; a++) {
		int   best = -1;
		float bs   = conf;

		for (int c = 0; c < n_classes; c++) {
			float s = out[(size_t)(4 + c) * anchors + a];
			if (s > bs) {
				bs   = s;
				best = c;
			}
		}
		if (best < 0)
			continue;

		float cx = out[(size_t)0 * anchors + a];
		float cy = out[(size_t)1 * anchors + a];
		float w  = out[(size_t)2 * anchors + a];
		float h  = out[(size_t)3 * anchors + a];

		struct bc_det *c = &cand[n++];
		*c = (struct bc_det){
			.x = cx - w * 0.5f, .y = cy - h * 0.5f,
			.w = w, .h = h, .score = bs, .cls = best
		};

		/*
		 * Keypoints follow the class scores, three channels per point
		 * (x, y, confidence) in the same channel-major layout. Already
		 * in network pixels, like the box.
		 */
		c->n_kpt = n_kpt;
		for (int k = 0; k < n_kpt && k < BC_MAX_KPT; k++) {
			size_t base = (size_t)(4 + n_classes + 3 * k) * anchors;
			c->kx[k] = out[base + a];
			c->ky[k] = out[base + anchors + a];
			c->ks[k] = out[base + 2 * (size_t)anchors + a];
		}
	}

	qsort(cand, (size_t)n, sizeof *cand, det_cmp);
	return no_wbf() ? nms_greedy(cand, n, dst, max)
	                : wbf(cand, n, dst, max);
}

/*
 * Build one silhouette from a segmentation head.
 *
 * The frame's masks are OR-ed into a single bitmap rather than kept apart. A
 * censor asks "is this pixel on a subject", never "which subject", and keeping
 * them separate would mean carrying a mask alongside a box through easing,
 * prediction and fusion -- where it would be geometry from one inference riding
 * on a box that has since moved, with no honest way to follow it.
 *
 * Anchors are walked directly instead of going through decode(): the mask
 * coefficients live in the raw output next to the class scores, and threading
 * 32 more floats through struct bc_det would widen the 512-entry candidate
 * array every other model pays for.
 *
 * `want_cls` is the class to take masks from, or -1 for all of them.
 */
static void build_mask(struct bc_detector *d, const float *out, int channels,
                       int anchors, const float *proto, float conf,
                       int n_classes, int want_cls)
{
	const int pw = d->mask_w, ph = d->mask_h, nm = d->n_mask;
	const size_t ppx = (size_t)pw * ph;

	memset(d->work_mask, 0, ppx);

	/* Network pixels per prototype pixel: the prototypes cover the whole
	 * letterboxed input, so this is just the ratio of the two grids. */
	const float sx = (float)d->net_w / (float)pw;
	const float sy = (float)d->net_h / (float)ph;

	/*
	 * One pass per detection, and every detection above the floor is taken:
	 * duplicates that greedy NMS would have removed are harmless here,
	 * because OR-ing a mask with a near-copy of itself changes nothing. That
	 * is only true because the output is a union -- it would be wrong if
	 * these were separate masks.
	 *
	 * Bounded so a pathological frame cannot turn into thousands of 160x160
	 * accumulations; the strongest are not first, so this is a cap on work,
	 * not a ranking. Coverage lost to it can only be a subject already
	 * covered by another detection of the same union.
	 */
	int used = 0;
	for (int a = 0; a < anchors && used < BC_SEG_MAX_INSTANCES; a++) {
		int   best = -1;
		float bs   = conf;
		for (int c = 0; c < n_classes; c++) {
			const float s = out[(size_t)(4 + c) * anchors + a];
			if (s > bs) { bs = s; best = c; }
		}
		if (best < 0 || (want_cls >= 0 && best != want_cls))
			continue;
		used++;

		/* Box in network pixels -> prototype pixels, clamped. The mask is
		 * cropped to it, which is what stops one subject's coefficients
		 * painting over the whole frame. */
		const float cx = out[(size_t)0 * anchors + a];
		const float cy = out[(size_t)1 * anchors + a];
		const float bw = out[(size_t)2 * anchors + a];
		const float bh = out[(size_t)3 * anchors + a];

		int x0 = (int)floorf((cx - bw * 0.5f) / sx);
		int y0 = (int)floorf((cy - bh * 0.5f) / sy);
		int x1 = (int)ceilf((cx + bw * 0.5f) / sx);
		int y1 = (int)ceilf((cy + bh * 0.5f) / sy);
		if (x0 < 0) x0 = 0;
		if (y0 < 0) y0 = 0;
		if (x1 > pw) x1 = pw;
		if (y1 > ph) y1 = ph;
		if (x1 <= x0 || y1 <= y0)
			continue;

		const float *coef = &out[(size_t)(channels - nm) * anchors];

		for (int y = y0; y < y1; y++) {
			for (int x = x0; x < x1; x++) {
				const size_t at = (size_t)y * pw + x;
				if (d->work_mask[at])
					continue;      /* already covered */
				float v = 0.0f;
				for (int i = 0; i < nm; i++)
					v += coef[(size_t)i * anchors + a] *
					     proto[(size_t)i * ppx + at];
				/* sigmoid(v) > 0.5 is exactly v > 0, and skipping
				 * the exp is worth it 160x160 times per instance. */
				if (v > 0.0f)
					d->work_mask[at] = 255;
			}
		}
	}
}

/* --- worker ---------------------------------------------------------------- */

/*
 * A run that produced nothing usable. Published as its own outcome, never as
 * an empty result: see enum bc_det_status. It replaces any uncollected one,
 * as a success would -- including a success's boxes, which describe an older
 * frame than the one that just failed.
 */
static void run_failed(struct bc_detector *d, double t0)
{
	pthread_mutex_lock(&d->lock);
	d->errors++;
	d->n_results    = 0;
	d->ready        = 1;
	d->ready_failed = 1;
	d->last_ms      = now_ms() - t0;
	pthread_mutex_unlock(&d->lock);
}

/* A run that worked: its detections, possibly none, as a BC_DET_OK outcome. */
static void publish(struct bc_detector *d, const struct bc_det *acc,
                    int total, double t0)
{
	pthread_mutex_lock(&d->lock);
	memcpy(d->results, acc, (size_t)total * sizeof *acc);
	d->n_results    = total;
	d->ready        = 1;
	d->ready_failed = 0;
	/*
	 * Only when this submission actually built one. A run that failed to
	 * read its prototypes must leave the previous silhouette standing rather
	 * than publish an empty one -- an empty mask reads as "no subject here",
	 * which would uncover everything it gates.
	 */
	if (d->n_mask > 0 && d->mask_built) {
		memcpy(d->mask, d->work_mask, (size_t)d->mask_w * d->mask_h);
		d->mask_ready = 1;
		d->mask_built = 0;
	}
	d->last_ms      = now_ms() - t0;   /* the whole submission, all tiles */
	d->count++;
	pthread_mutex_unlock(&d->lock);
}

/*
 * One inference over one tensor. Returns the number of detections written to
 * `out`, or -1 if the run failed -- which the caller must not confuse with 0.
 */
static int run_tile(struct bc_detector *d, const void *tensor, int on_device,
                    struct bc_det *dst, int max)
{
	const OrtApi *api = d->api;

	int64_t shape[4] = { 1, 3, d->net_h, d->net_w };
	OrtValue *in = NULL, *out = NULL;

	/*
	 * Which memory this describes is the whole of the zero-copy change. ORT
	 * looks at the OrtMemoryInfo to decide whether the input needs copying
	 * to the device before the graph can run; told the data is already
	 * there, it skips the copy rather than doing it differently.
	 *
	 * Outputs are deliberately left alone. They come back to the host
	 * either way -- the boxes have to be read by the CPU -- so binding them
	 * would move the copy without removing it.
	 */
	OrtMemoryInfo *mem = on_device ? d->dev_mem : d->mem;

	if (ort_fail(d, api->CreateTensorWithDataAsOrtValue(
			mem, (void *)tensor, d->floats * sizeof(float),
			shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in),
		"CreateTensor"))
		return -1;

	const char *in_names[]  = { d->in_name };
	const char *out_names[] = { d->out_name, d->proto_name };
	const size_t n_want     = d->proto_name ? 2 : 1;
	OrtValue    *outs[2]    = { NULL, NULL };

	if (ort_fail(d, api->Run(d->session, NULL, in_names,
	                         (const OrtValue *const *)&in, 1,
	                         out_names, n_want, outs), "Run")) {
		api->ReleaseValue(in);
		return -1;
	}
	out = outs[0];

	/* Shape is dynamic in the model, so read it back rather than assuming. */
	OrtTensorTypeAndShapeInfo *info = NULL;
	int64_t dims[4] = { 0 };
	size_t  n_dims  = 0;
	float  *data    = NULL;
	int     n       = 0;

	if (!ort_fail(d, api->GetTensorTypeAndShape(out, &info), "GetShape")) {
		if (ort_fail(d, api->GetDimensionsCount(info, &n_dims), "GetDimCount"))
			n_dims = 0;
		if (n_dims == 3 &&
		    ort_fail(d, api->GetDimensions(info, dims, 3), "GetDims"))
			n_dims = 0;
		api->ReleaseTensorTypeAndShapeInfo(info);
	}

	if (n_dims == 3 &&
	    !ort_fail(d, api->GetTensorMutableData(out, (void **)&data), "GetData"))
		n = d->end2end
		  ? decode_end2end(data, (int)dims[1], (int)dims[2], dst,
		                   max, d->work_conf,
		                   d->n_classes ? d->n_classes : BC_MAX_CLASSES)
		  : decode(data, (int)dims[1], (int)dims[2], dst,
		           max, d->work_conf,
		           d->n_classes ? d->n_classes
		                        : (int)dims[1] - 4 - 3 * d->n_kpt
		                          - d->n_mask,
		           d->n_kpt, d->n_mask);

	/*
	 * The silhouette, from the same output the boxes came from plus the
	 * prototypes. Built here rather than by the caller so the raw tensors
	 * never have to outlive the run that produced them.
	 */
	if (d->n_mask > 0 && n_dims == 3 && data && n >= 0 && outs[1]) {
		float *proto = NULL;
		if (!ort_fail(d, api->GetTensorMutableData(outs[1], (void **)&proto),
		              "GetProtoData") && proto) {
			build_mask(d, data, (int)dims[1], (int)dims[2], proto,
			           d->work_conf,
			           d->n_classes ? d->n_classes
			                        : (int)dims[1] - 4 - d->n_mask,
			           d->mask_cls);
			d->mask_built = 1;
		}
	}

	if (outs[1])
		api->ReleaseValue(outs[1]);
	api->ReleaseValue(out);
	api->ReleaseValue(in);

	/*
	 * Publishing n=0 here when the output could not be read would be
	 * fail-open: the caller cannot distinguish it from a clean screen and
	 * would drop every censor bar. Count it as an error instead and publish
	 * nothing, so the caller's deadline notices results have stopped.
	 */
	if (n_dims != 3 || !data)
		return -1;

	return n;
}

/*
 * One submission: every tile, then publish once.
 *
 * All-or-nothing on failure. A partial publish would show the tiles that
 * worked and silently omit the one that did not, and the caller has no way to
 * tell that region apart from one where nothing was found -- so a broken tile
 * would read as clean screen. Counting the whole submission as an error
 * instead lets the existing deadline and failsafe do their job.
 */
static void run_once(struct bc_detector *d)
{
	const double t0 = now_ms();
	struct bc_det acc[BC_MAX_DET];
	int total = 0;

	if (d->fault_err_at > 0.0 && t0 >= d->fault_err_at) {
		run_failed(d, t0);
		return;
	}

	/*
	 * Tiles are consecutive in either case, so the only difference is which
	 * base the offset is taken from. The device pointer is arithmetic-only
	 * here -- it is never dereferenced on this side of the bus.
	 */
	const int on_device = d->work_dev != NULL;
	const char *base = on_device ? (const char *)d->work_dev
	                             : (const char *)d->work_buf;
	const size_t stride = d->floats * sizeof(float);

	for (int t = 0; t < d->tiles; t++) {
		struct bc_det tmp[BC_MAX_DET];
		const int n = run_tile(d, base + (size_t)t * stride, on_device,
		                       tmp, BC_MAX_DET);
		if (n < 0) {
			run_failed(d, t0);
			return;
		}
		/*
		 * The budget is for the whole submission, not per tile, so a
		 * grid can overrun it. Filling it in tile order and stopping
		 * would drop whichever tiles run last -- always the same ones,
		 * so always the same part of the screen, which is uncensored
		 * every frame rather than occasionally. At 3x2 that is the
		 * bottom-right third going uncovered while the top-left works
		 * perfectly, which is exactly how it presented.
		 *
		 * Once full, the weakest detection anywhere in the accumulator
		 * gives way to a stronger one. Truncation is then by confidence
		 * and the loss is spread over the frame instead of carving a
		 * blind region out of it.
		 */
		for (int i = 0; i < n; i++) {
			tmp[i].tile = t;
			if (total < BC_MAX_DET) {
				acc[total++] = tmp[i];
				continue;
			}
			int lo = 0;
			for (int k = 1; k < total; k++)
				if (acc[k].score < acc[lo].score)
					lo = k;
			if (tmp[i].score > acc[lo].score)
				acc[lo] = tmp[i];

			static int warned;
			if (!warned) {
				warned = 1;
				ERR("detection budget (%d) full with %d tiles -- "
				    "keeping the strongest; raise BC_MAX_DET if "
				    "this is common", BC_MAX_DET, d->tiles);
			}
		}
	}

	publish(d, acc, total, t0);
}

static void *worker(void *arg)
{
	struct bc_detector *d = arg;

	for (;;) {
		pthread_mutex_lock(&d->lock);
		while (d->running && !d->pending)
			pthread_cond_wait(&d->cv, &d->lock);
		if (!d->running) {
			pthread_mutex_unlock(&d->lock);
			return NULL;
		}
		/* Take ownership of the submitted tensor. */
		float *tmp = d->work_buf;
		d->work_buf = d->in_buf;
		d->in_buf   = tmp;
		d->work_dev = d->in_dev;
		d->in_dev   = NULL;
		d->pending   = 0;
		d->busy      = 1;
		d->work_conf = d->conf;   /* snapshot: conf can change under us */
		pthread_mutex_unlock(&d->lock);

		run_once(d);

		pthread_mutex_lock(&d->lock);
		d->busy = 0;
		pthread_cond_broadcast(&d->idle_cv);
		pthread_mutex_unlock(&d->lock);
	}
}

/* --- public ---------------------------------------------------------------- */

struct bc_detector *bc_detect_create(const char *model_path,
                                     int net_w, int net_h, int threads)
{
	return bc_detect_create_ex(model_path, net_w, net_h, threads, 1);
}

struct bc_detector *bc_detect_create_ex(const char *model_path,
                                        int net_w, int net_h, int threads,
                                        int adopt_taxonomy)
{
	struct bc_detector *d = calloc(1, sizeof *d);
	if (!d)
		return NULL;

	d->api    = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	d->net_w  = net_w;
	d->net_h  = net_h;
	d->floats = (size_t)3 * net_w * net_h;
	d->conf   = conf_threshold();
	/* Not 0: calloc's zero is a valid class index, and defaulting the
	 * silhouette to "whatever class 0 happens to be" is a silent wrong
	 * answer rather than a missing one. */
	d->mask_cls = -1;

	d->tiles    = 1;
	d->in_buf   = calloc(d->floats, sizeof(float));
	d->work_buf = calloc(d->floats, sizeof(float));
	if (!d->in_buf || !d->work_buf)
		goto fail;

	if (ort_fail(d, d->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "betacensor",
	                                  &d->env), "CreateEnv"))
		goto fail;
	if (ort_fail(d, d->api->CreateSessionOptions(&d->opts), "CreateSessionOptions"))
		goto fail;

	if (ort_fail(d, d->api->SetIntraOpNumThreads(d->opts, threads),
	             "SetIntraOpNumThreads"))
		goto fail;
	if (ort_fail(d, d->api->SetSessionGraphOptimizationLevel(d->opts,
	                                                         ORT_ENABLE_ALL),
	             "SetGraphOptimizationLevel"))
		goto fail;

	/*
	 * CUDA execution provider, best-effort (DESIGN.md §10).
	 *
	 * Deliberately not fatal. The provider library dlopen()s cuDNN, so on a
	 * machine without it this call fails and we run on CPU -- which works,
	 * just slower. Making it fatal would turn a missing optional dependency
	 * into "the censor does not start", and a censor that does not start is
	 * the worst outcome available.
	 *
	 * Verified for this machine before wiring it up: the shipped
	 * libonnxruntime_providers_cuda.so contains native sm_61 SASS (alongside
	 * sm_60/70/75/80), so a GTX 1080 Ti runs precompiled kernels rather than
	 * JIT-ing from PTX. cuDNN must be <= 9.11.1: NVIDIA dropped Pascal in
	 * 9.12.0, so the distro's current cuDNN is not usable here.
	 */
	if (!getenv("BC_NO_CUDA")) {
		OrtCUDAProviderOptions cuda = { 0 };
		cuda.device_id = 0;
		OrtStatus *st = d->api->SessionOptionsAppendExecutionProvider_CUDA(
			d->opts, &cuda);
		if (st) {
			WARN("CUDA EP unavailable (%s) -- falling back to CPU",
			     d->api->GetErrorMessage(st));
			d->api->ReleaseStatus(st);
		} else {
			LOG("detector: CUDA execution provider enabled (device 0)");
			d->on_gpu = 1;
		}
	}

	/*
	 * Accepting the CUDA EP above only means the provider library loaded --
	 * it does not mean a CUDA context can actually be created. That happens
	 * here, and it can fail on its own (a driver mismatch reports itself as
	 * "out of memory" from cudaSetDevice even with the card nearly empty).
	 *
	 * So the CPU fallback has to cover this call too, not just the append.
	 * An earlier version failed the whole detector here, which meant a GPU
	 * problem stopped the censor from starting at all -- strictly worse than
	 * running slowly on the CPU, and the exact outcome the append-side
	 * fallback was written to avoid.
	 */
	OrtStatus *sess = d->api->CreateSession(d->env, model_path, d->opts,
	                                        &d->session);
	if (sess && d->on_gpu) {
		WARN("CUDA session failed (%s) -- retrying on CPU",
		     d->api->GetErrorMessage(sess));
		d->api->ReleaseStatus(sess);
		d->on_gpu = 0;

		/* Options are not un-appendable; build clean ones. */
		d->api->ReleaseSessionOptions(d->opts);
		d->opts = NULL;
		if (ort_fail(d, d->api->CreateSessionOptions(&d->opts),
		             "CreateSessionOptions"))
			goto fail;
		if (ort_fail(d, d->api->SetIntraOpNumThreads(d->opts, threads),
		             "SetIntraOpNumThreads") ||
		    ort_fail(d, d->api->SetSessionGraphOptimizationLevel(
		                    d->opts, ORT_ENABLE_ALL),
		             "SetGraphOptimizationLevel"))
			goto fail;

		sess = d->api->CreateSession(d->env, model_path, d->opts,
		                             &d->session);
	}
	if (ort_fail(d, sess, "CreateSession")) {
		ERR("could not load model '%s'", model_path);
		goto fail;
	}
	if (ort_fail(d, d->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
	                                            &d->mem), "CreateCpuMemoryInfo"))
		goto fail;

	/*
	 * A second one describing device memory, so a tensor the preprocessor
	 * left in VRAM can be handed over where it lies. Only meaningful on a
	 * CUDA session -- and only best-effort even then: "Cuda" is the name the
	 * EP registers its allocator under, and if this build of ORT does not
	 * know it, the failure is that zero-copy is unavailable, which the
	 * readback path already covers. So it is not fatal.
	 */
	if (d->on_gpu) {
		OrtStatus *ms = d->api->CreateMemoryInfo("Cuda", OrtDeviceAllocator, 0,
		                                         OrtMemTypeDefault, &d->dev_mem);
		if (ms) {
			WARN("no CUDA memory descriptor (%s) -- the tensor will go "
			     "via host memory", d->api->GetErrorMessage(ms));
			d->api->ReleaseStatus(ms);
			d->dev_mem = NULL;
		}
	}

	OrtAllocator *alloc = NULL;
	if (ort_fail(d, d->api->GetAllocatorWithDefaultOptions(&alloc), "GetAllocator"))
		goto fail;

	char *in_name = NULL, *out_name = NULL;
	if (ort_fail(d, d->api->SessionGetInputName(d->session, 0, alloc, &in_name),
	             "GetInputName"))
		goto fail;
	if (ort_fail(d, d->api->SessionGetOutputName(d->session, 0, alloc, &out_name),
	             "GetOutputName"))
		goto fail;

	d->in_name  = strdup(in_name);
	d->out_name = strdup(out_name);
	(void)ort_fail(d, d->api->AllocatorFree(alloc, in_name),  "AllocatorFree");
	(void)ort_fail(d, d->api->AllocatorFree(alloc, out_name), "AllocatorFree");
	if (!d->in_name || !d->out_name)
		goto fail;

	/*
	 * A second output means a segmentation head: output1 is the mask
	 * prototypes, [1, n_mask, ph, pw]. Decided by what the session exposes
	 * rather than by the filename, for the same reason the taxonomy is read
	 * from metadata -- the file can be called anything.
	 */
	size_t n_outputs = 0;
	if (!ort_fail(d, d->api->SessionGetOutputCount(d->session, &n_outputs),
	              "GetOutputCount") && n_outputs >= 2) {
		char *pn = NULL;
		if (!ort_fail(d, d->api->SessionGetOutputName(d->session, 1, alloc, &pn),
		              "GetOutputName(1)")) {
			d->proto_name = strdup(pn);
			(void)ort_fail(d, d->api->AllocatorFree(alloc, pn),
			               "AllocatorFree");
		}

		OrtTypeInfo *pti = NULL;
		if (d->proto_name &&
		    !ort_fail(d, d->api->SessionGetOutputTypeInfo(d->session, 1, &pti),
		              "GetOutputTypeInfo(1)")) {
			const OrtTensorTypeAndShapeInfo *psi = NULL;
			int64_t pd[4] = { 0 };
			size_t  pnd = 0;
			if (!ort_fail(d, d->api->CastTypeInfoToTensorInfo(pti, &psi),
			              "CastTypeInfo") && psi &&
			    !ort_fail(d, d->api->GetDimensionsCount(psi, &pnd),
			              "GetDimCount") && pnd == 4 &&
			    !ort_fail(d, d->api->GetDimensions(psi, pd, 4), "GetDims") &&
			    pd[1] > 0 && pd[2] > 0 && pd[3] > 0) {
				d->n_mask = (int)pd[1];
				d->mask_h = (int)pd[2];
				d->mask_w = (int)pd[3];
			}
			d->api->ReleaseTypeInfo(pti);
		}

		if (d->n_mask > 0) {
			const size_t px = (size_t)d->mask_w * d->mask_h;
			d->mask      = calloc(px, 1);
			d->work_mask = calloc(px, 1);
			if (!d->mask || !d->work_mask)
				goto fail;
			LOG("detector: segmentation head -- %d mask coefficients, "
			    "%dx%d prototypes", d->n_mask, d->mask_w, d->mask_h);
		} else {
			/* Two outputs but not a shape we understand: ignore the
			 * second rather than guess at it. */
			free(d->proto_name);
			d->proto_name = NULL;
		}
	}

	/*
	 * Take the input geometry from the model, not from the caller.
	 *
	 * The caller can only guess -- it was guessing from the filename -- and
	 * a wrong guess is not a soft failure: a model with a fixed input size
	 * rejects every Run() with "invalid dimensions", so the detector produces
	 * nothing at all while looking alive. Models with a dynamic input keep
	 * whatever the caller asked for, which is what lets the NudeNet models
	 * run at 320x192 instead of a square letterbox.
	 */
	OrtTypeInfo *ti = NULL;
	if (!ort_fail(d, d->api->SessionGetInputTypeInfo(d->session, 0, &ti),
	              "GetInputTypeInfo")) {
		const OrtTensorTypeAndShapeInfo *si = NULL;
		int64_t dims[4] = { 0 };
		size_t  nd = 0;
		if (!ort_fail(d, d->api->CastTypeInfoToTensorInfo(ti, &si),
		              "CastTypeInfo") && si &&
		    !ort_fail(d, d->api->GetDimensionsCount(si, &nd), "GetDimCount") &&
		    nd == 4 &&
		    !ort_fail(d, d->api->GetDimensions(si, dims, 4), "GetDims")) {
			/* NCHW; a negative or zero extent means dynamic. */
			if (dims[3] > 0 && dims[2] > 0 &&
			    (dims[3] != d->net_w || dims[2] != d->net_h)) {
				LOG("detector: model requires a fixed %lldx%lld "
				    "input (asked for %dx%d) -- using the "
				    "model's",
				    (long long)dims[3], (long long)dims[2],
				    d->net_w, d->net_h);
				d->net_w  = (int)dims[3];
				d->net_h  = (int)dims[2];
				d->floats = (size_t)3 * d->net_w * d->net_h;
				free(d->in_buf);
				free(d->work_buf);
				d->in_buf   = calloc(d->floats, sizeof(float));
				d->work_buf = calloc(d->floats, sizeof(float));
				if (!d->in_buf || !d->work_buf) {
					d->api->ReleaseTypeInfo(ti);
					goto fail;
				}
			}
		}
		d->api->ReleaseTypeInfo(ti);
	}

	/*
	 * Adopt the model's own taxonomy. The hardcoded table is only a default
	 * for models that carry no metadata; when the model states its classes,
	 * they win. Two models here disagree on both the count and the order
	 * (NudeNet v3: 18, starting FEMALE_GENITALIA_COVERED/FACE_FEMALE;
	 * Hotscreen's YOLOv11: 16, starting FEMALE_FACE/MALE_FACE), so trusting
	 * a remembered order would censor faces and miss genitals.
	 */
	/*
	 * A model that does not publish its taxonomy still has to be decoded
	 * against its own head, not the global table: the eye model emits 5
	 * channels (4 box + 1 class) where the primary emits 20, and validating
	 * one against the other rejected every inference it ever ran.
	 */
	d->n_classes = 0;   /* 0 = derive from the output shape */

	/*
	 * BC_FAULT_DETERR=ms  from that long after load, every run fails, as a
	 * CUDA error or an unreadable output would. The §6 fault injectors
	 * stop the capture or stop submitting; neither reaches the case of a
	 * detector that is fed and answers with an error, which is the one the
	 * change gate used to hide on a still screen.
	 */
	const char *fault = getenv("BC_FAULT_DETERR");
	if (fault && atof(fault) > 0.0) {
		d->fault_err_at = now_ms() + atof(fault);
		WARN("fault injection active: detector errors @%.0fms",
		     atof(fault));
	}

	OrtModelMetadata *md = NULL;
	if (!ort_fail(d, d->api->SessionGetModelMetadata(d->session, &md),
	              "GetModelMetadata")) {
		/*
		 * A Pose head appends 3 channels per keypoint after the class
		 * scores. The count has to come from the model -- 4 + nc + 3k
		 * has two unknowns, so the output shape alone cannot separate a
		 * keypoint model from one with more classes.
		 */
		char *kpt = NULL;
		OrtStatus *ks = d->api->ModelMetadataLookupCustomMetadataMap(
			md, alloc, "kpt_shape", &kpt);
		if (ks) {
			d->api->ReleaseStatus(ks);
		} else if (kpt) {
			int n = atoi(kpt[0] == '[' ? kpt + 1 : kpt);
			if (n > 0 && n <= BC_MAX_KPT) {
				d->n_kpt = n;
				LOG("detector: pose head, %d keypoints per box", n);
			} else if (n > BC_MAX_KPT) {
				WARN("detector: model has %d keypoints, over the "
				     "%d supported -- ignoring them", n, BC_MAX_KPT);
			}
			alloc->Free(alloc, kpt);
		}

		/*
		 * An end-to-end model emits final boxes, so the whole decode
		 * path differs. Read it from metadata rather than guessing at
		 * the output shape: [1, 300, 6] is ambiguous with a 2-class
		 * model's [1, 6, anchors] on shape alone.
		 */
		char *e2e = NULL;
		OrtStatus *es = d->api->ModelMetadataLookupCustomMetadataMap(
			md, alloc, "end2end", &e2e);
		if (es) {
			d->api->ReleaseStatus(es);
		} else if (e2e) {
			if (e2e[0] == 'T' || e2e[0] == 't' || e2e[0] == '1') {
				d->end2end = 1;
				LOG("detector: end-to-end head (model does its "
				    "own NMS)");
			}
			alloc->Free(alloc, e2e);
		}

		/*
		 * A seg model's own taxonomy, read but not adopted. It is COCO,
		 * so its `person` has nothing to do with the censor's classes and
		 * must never reach the global table -- but the silhouette has to
		 * come from people rather than from every sofa in the frame, so
		 * the channel still has to be looked up rather than assumed to be
		 * 0. Absent, -1 takes every class: more coverage, not less, which
		 * is the safe direction to be wrong in.
		 */
		if (d->n_mask > 0) {
			char *sn = NULL;
			OrtStatus *ss = d->api->ModelMetadataLookupCustomMetadataMap(
				md, alloc, "names", &sn);
			if (ss) {
				d->api->ReleaseStatus(ss);
			} else if (sn) {
				d->mask_cls = meta_index_of(sn, "person");
				if (d->mask_cls >= 0)
					LOG("detector: silhouettes from class %d "
					    "('person')", d->mask_cls);
				else
					WARN("detector: no 'person' class in the "
					     "segmentation model -- taking every "
					     "class's mask");
				alloc->Free(alloc, sn);
			}
		}

		char *names = NULL;
		OrtStatus *st = adopt_taxonomy
			? d->api->ModelMetadataLookupCustomMetadataMap(
				md, alloc, "names", &names)
			: NULL;
		if (st) {
			d->api->ReleaseStatus(st);
		} else if (names) {
			int n = parse_names(names);
			if (n)
				LOG("detector: taxonomy from model metadata "
				    "(%d classes, 0=%s)", n, bc_class_name(0));
			else
				WARN("detector: model 'names' metadata unparseable"
				     " -- keeping built-in %d-class table",
				     bc_class_count());
			alloc->Free(alloc, names);
		}
		d->api->ReleaseModelMetadata(md);
	}
	if (adopt_taxonomy)
		d->n_classes = bc_class_count();

	LOG("detector: %s  in='%s' out='%s'  %dx%d  %d thread(s)  conf=%.2f  [%s]",
	    model_path, d->in_name, d->out_name, d->net_w, d->net_h, threads, d->conf,
	    d->on_gpu ? "CUDA" : "CPU");

	pthread_mutex_init(&d->lock, NULL);
	pthread_cond_init(&d->cv, NULL);
	pthread_cond_init(&d->idle_cv, NULL);
	d->running = 1;
	if (pthread_create(&d->thread, NULL, worker, d) != 0) {
		ERR("could not start detector thread");
		d->running = 0;
		goto fail;
	}
	return d;

fail:
	bc_detect_destroy(d);
	return NULL;
}

void bc_detect_destroy(struct bc_detector *d)
{
	if (!d)
		return;

	if (d->running) {
		pthread_mutex_lock(&d->lock);
		d->running = 0;
		pthread_cond_signal(&d->cv);
		pthread_mutex_unlock(&d->lock);
		pthread_join(d->thread, NULL);
		pthread_mutex_destroy(&d->lock);
		pthread_cond_destroy(&d->cv);
		pthread_cond_destroy(&d->idle_cv);
	}

	if (d->api) {
		if (d->dev_mem) d->api->ReleaseMemoryInfo(d->dev_mem);
		if (d->mem)     d->api->ReleaseMemoryInfo(d->mem);
		if (d->session) d->api->ReleaseSession(d->session);
		if (d->opts)    d->api->ReleaseSessionOptions(d->opts);
		if (d->env)     d->api->ReleaseEnv(d->env);
	}
	free(d->in_name);
	free(d->out_name);
	free(d->proto_name);
	free(d->mask);
	free(d->work_mask);
	free(d->in_buf);
	free(d->work_buf);
	free(d);
}

int bc_detect_submit(struct bc_detector *d, const float *tensor)
{
	if (!d)
		return 0;

	pthread_mutex_lock(&d->lock);
	if (d->busy || d->pending) {
		pthread_mutex_unlock(&d->lock);
		return 0;               /* still working; skip this frame */
	}
	memcpy(d->in_buf, tensor,
	       d->floats * (size_t)d->tiles * sizeof(float));
	d->in_dev  = NULL;   /* a host tensor: read it from in_buf */
	d->pending = 1;
	pthread_cond_signal(&d->cv);
	pthread_mutex_unlock(&d->lock);
	return 1;
}

int bc_detect_submit_device(struct bc_detector *d, const void *dev)
{
	if (!d || !dev || !d->dev_mem)
		return 0;

	pthread_mutex_lock(&d->lock);
	if (d->busy || d->pending) {
		pthread_mutex_unlock(&d->lock);
		return 0;               /* still working; skip this frame */
	}
	/*
	 * No copy, which is the point and also the hazard: the caller now owns
	 * a promise not to touch that memory until bc_detect_busy() goes false.
	 */
	d->in_dev  = dev;
	d->pending = 1;
	pthread_cond_signal(&d->cv);
	pthread_mutex_unlock(&d->lock);
	return 1;
}

int bc_detect_zero_copy(const struct bc_detector *d)
{
	return d && d->on_gpu && d->dev_mem;
}

void bc_detect_wait_idle(struct bc_detector *d)
{
	if (!d)
		return;
	pthread_mutex_lock(&d->lock);
	/*
	 * `pending` as well as `busy`: a submitted-but-not-yet-started run owns
	 * the tensor just as much as a running one does. Bounded by one
	 * inference, and only ever reached on a config change.
	 */
	while (d->running && (d->busy || d->pending))
		pthread_cond_wait(&d->idle_cv, &d->lock);
	pthread_mutex_unlock(&d->lock);
}

int bc_detect_set_tiles(struct bc_detector *d, int n)
{
	if (!d || n < 1)
		return -1;
	if (n == d->tiles)
		return 0;

	const size_t want = d->floats * (size_t)n;
	float *a = calloc(want, sizeof(float));
	float *b = calloc(want, sizeof(float));
	if (!a || !b) {
		free(a);
		free(b);
		ERR("cannot size the detector for %d tiles", n);
		return -1;
	}

	pthread_mutex_lock(&d->lock);
	free(d->in_buf);
	free(d->work_buf);
	d->in_buf   = a;
	d->work_buf = b;
	d->tiles    = n;
	/* Anything queued was sized for the old grid. */
	d->pending  = 0;
	d->ready    = 0;
	pthread_mutex_unlock(&d->lock);
	return 0;
}

int bc_detect_has_mask(const struct bc_detector *d)
{
	return d && d->n_mask > 0;
}

int bc_detect_mask_w(const struct bc_detector *d) { return d ? d->mask_w : 0; }
int bc_detect_mask_h(const struct bc_detector *d) { return d ? d->mask_h : 0; }

int bc_detect_take_mask(struct bc_detector *d, unsigned char *out, size_t max)
{
	if (!d || !out || d->n_mask <= 0)
		return -1;

	const size_t px = (size_t)d->mask_w * d->mask_h;
	if (max < px)
		return -1;

	pthread_mutex_lock(&d->lock);
	const int have = d->mask_ready;
	if (have) {
		memcpy(out, d->mask, px);
		d->mask_ready = 0;
	}
	pthread_mutex_unlock(&d->lock);
	return have ? (int)px : -1;
}

int bc_detect_busy(const struct bc_detector *d)
{
	if (!d)
		return 0;
	struct bc_detector *m = (struct bc_detector *)d;
	pthread_mutex_lock(&m->lock);
	const int busy = m->busy || m->pending;
	pthread_mutex_unlock(&m->lock);
	return busy;
}

struct bc_det_result bc_detect_collect(struct bc_detector *d,
                                       struct bc_det *out, int max)
{
	struct bc_det_result r = { BC_DET_PENDING, 0, 0.0 };
	if (!d)
		return r;

	pthread_mutex_lock(&d->lock);
	if (d->ready) {
		r.ms = d->last_ms;
		if (d->ready_failed) {
			r.status = BC_DET_FAILED;
		} else {
			r.status = BC_DET_OK;
			r.n = d->n_results < max ? d->n_results : max;
			memcpy(out, d->results, (size_t)r.n * sizeof *out);
		}
		d->ready = 0;
	}
	pthread_mutex_unlock(&d->lock);
	return r;
}

void bc_detect_set_conf(struct bc_detector *d, float conf)
{
	if (!d || !(conf > 0.0f) || !(conf < 1.0f))
		return;
	pthread_mutex_lock(&d->lock);
	d->conf = conf;
	pthread_mutex_unlock(&d->lock);
}

int bc_detect_on_gpu(const struct bc_detector *d) { return d ? d->on_gpu : 0; }

int bc_detect_n_kpt(const struct bc_detector *d) { return d ? d->n_kpt : 0; }

int bc_detect_net_w(const struct bc_detector *d) { return d ? d->net_w : 0; }
int bc_detect_net_h(const struct bc_detector *d) { return d ? d->net_h : 0; }

unsigned long bc_detect_count(const struct bc_detector *d)
{
	return d ? d->count : 0;
}

unsigned long bc_detect_errors(const struct bc_detector *d)
{
	return d ? d->errors : 0;
}

