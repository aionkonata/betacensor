/*
 * bc-image -- censor still images offline: one, several, or whole folders.
 *
 * Pury.fi's batch converter, done the way bc-video does video: the detector,
 * the class policy, the config and the effect shaders are the live censor's
 * own (see offline.h), and only the two ends differ. Each image comes in
 * through an ffmpeg decoder pipe and goes back out through an encoder pipe.
 *
 * What a still changes, compared with a video:
 *
 * [1] No tracker. There is one frame, so there is nothing to carry a box
 *     across and no flicker to suppress. A detection either clears its
 *     class's floor or it does not.
 *
 * [2] Sectioning is honoured. The live censor can split the frame into a
 *     grid of tiles to find small subjects; a photo is often far larger than
 *     a screen, so it needs that more, not less, and there is no frame
 *     budget to stop it paying for every tile.
 *
 * [3] Every image is its own size, so the render target is resized per
 *     image rather than once.
 *
 * §6 holds per image: an image the detector did not answer for is never
 * written, rather than written uncensored. A batch with any such image ends
 * FAILED, naming each one.
 *
 * Outputs are written to a hidden temporary name and renamed into place once
 * the encoder has succeeded, so a half-written file never exists under the
 * real name. That is also what makes resuming safe: an output that exists is
 * a finished one, and by default it is skipped.
 */

#include "config.h"
#include "detect.h"
#include "preprocess.h"
#include "join.h"
#include "seam.h"
#include "offline.h"
#include "util.h"

#include <dirent.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_BOX 64

/* The live path's seam threshold (bc_main.c), so a tiled photo rejoins a
 * subject cut by a tile edge exactly as the screen would. */
#define SEAM_THRESH 0.4f

static volatile sig_atomic_t stop_requested;

static void on_stop(int sig)
{
	(void)sig;
	stop_requested = 1;
}

/* ------------------------------------------------------------------ */
/* Formats                                                             */

struct fmt {
	const char *ext;     /* lower case, no dot */
	const char *muxer;   /* ffmpeg -f */
	const char *codec;   /* ffmpeg codec flags */
	int         alpha;   /* can the output carry transparency? */
};

/*
 * What can be read. Decoding is ffmpeg's business and it reads all of these;
 * the list exists so a folder walk picks up images and not the text file
 * beside them.
 */
static const char *const READ_EXT[] = {
	"png", "jpg", "jpeg", "jfif", "webp", "bmp", "gif", "avif",
	"tif", "tiff", NULL
};

static const struct fmt FORMATS[] = {
	{ "png",  "image2", "-c:v png",                       1 },
	{ "jpg",  "image2", "-c:v mjpeg -q:v 2",              0 },
	{ "jpeg", "image2", "-c:v mjpeg -q:v 2",              0 },
	{ "jfif", "image2", "-c:v mjpeg -q:v 2",              0 },
	{ "webp", "webp",   "-c:v libwebp -quality 92",       1 },
	{ "bmp",  "image2", "-c:v bmp",                       0 },
	{ "tif",  "image2", "-c:v tiff",                      1 },
	{ "tiff", "image2", "-c:v tiff",                      1 },
	{ "avif", "avif",   "-c:v libaom-av1 -still-picture 1 -crf 18", 0 },
	{ NULL, NULL, NULL, 0 }
};

static const char *ext_of(const char *path)
{
	const char *slash = strrchr(path, '/');
	const char *dot   = strrchr(path, '.');
	if (!dot || (slash && dot < slash) || dot == path ||
	    (slash && dot == slash + 1))
		return "";
	return dot + 1;
}

static int readable(const char *path)
{
	const char *e = ext_of(path);
	for (int i = 0; READ_EXT[i]; i++)
		if (!strcasecmp(e, READ_EXT[i]))
			return 1;
	return 0;
}

static const struct fmt *fmt_for(const char *ext)
{
	for (int i = 0; FORMATS[i].ext; i++)
		if (!strcasecmp(ext, FORMATS[i].ext))
			return &FORMATS[i];
	return NULL;
}

/*
 * The output extension for an input. "same" keeps the input's own, except a
 * GIF, which becomes a PNG: only its first frame is censored, and a GIF
 * written back from a 24-bit render would also be re-quantised to 256
 * colours for no reason. Pury.fi makes the same substitution.
 */
static const char *out_ext(const char *in, const char *format)
{
	const char *e = ext_of(in);
	if (format && strcmp(format, "same"))
		return format;
	if (!strcasecmp(e, "gif"))
		return "png";
	return fmt_for(e) ? e : "png";
}

/* ------------------------------------------------------------------ */
/* Collecting the work                                                 */

struct job {
	char *in;
	char *out;
};

struct jobs {
	struct job *v;
	int         n, cap;
};

static void push(struct jobs *j, const char *in, const char *out)
{
	if (j->n == j->cap) {
		j->cap = j->cap ? j->cap * 2 : 64;
		j->v = realloc(j->v, (size_t)j->cap * sizeof *j->v);
		if (!j->v)
			DIE("out of memory listing images");
	}
	j->v[j->n].in  = strdup(in);
	j->v[j->n].out = strdup(out);
	j->n++;
}

/* "dir/sub/name.jpg" -> "OUT/sub/name.<ext>" for a relative "sub/name.jpg". */
static int out_path(char *dst, size_t cap, const char *out_dir,
                    const char *rel, const char *in, const char *format)
{
	const char *e = ext_of(rel);
	size_t stem = *e ? (size_t)(e - 1 - rel) : strlen(rel);
	int n = snprintf(dst, cap, "%s/%.*s.%s", out_dir, (int)stem, rel,
	                 out_ext(in, format));
	return n > 0 && (size_t)n < cap ? 0 : -1;
}

static int skip_name(const struct dirent *d)
{
	return d->d_name[0] != '.';
}

/*
 * Dropped into every output folder. The walk skips any folder holding one, so
 * an earlier run's output that sits inside the input tree is not censored a
 * second time into this run's output -- which is what happened the first
 * time a test put two output folders under one input folder. Comparing
 * against this run's output path alone cannot catch that; the marker can.
 */
#define OUT_MARKER ".betacensor-output"

static int is_output_dir(const char *dir)
{
	char m[PATH_MAX];
	if (snprintf(m, sizeof m, "%s/%s", dir, OUT_MARKER) >= (int)sizeof m)
		return 0;
	return access(m, F_OK) == 0;
}

/*
 * Walk a folder, depth first, in natural order ("img2" before "img10") so
 * progress reads the way a file manager lists the folder. Hidden entries are
 * skipped, and so is any output folder inside the input -- this run's or an
 * earlier one's -- or every run would censor the previous run's output too.
 */
static void walk(struct jobs *j, const char *root, const char *rel,
                 const char *out_dir, const char *out_real,
                 const char *format)
{
	char dir[PATH_MAX];
	if (snprintf(dir, sizeof dir, "%s%s%s", root, *rel ? "/" : "", rel)
	    >= (int)sizeof dir)
		return;

	char real[PATH_MAX];
	/* The marker is only checked below the top: a folder named
	 * explicitly on the command line is taken at its word. */
	if ((realpath(dir, real) && !strcmp(real, out_real)) ||
	    (*rel && is_output_dir(dir)))
		return;

	struct dirent **ents = NULL;
	int n = scandir(dir, &ents, skip_name, versionsort);
	if (n < 0) {
		WARN("cannot read folder %s: %s", dir, strerror(errno));
		return;
	}
	for (int i = 0; i < n; i++) {
		char child_rel[PATH_MAX], child[PATH_MAX], out[PATH_MAX];
		int ok = snprintf(child_rel, sizeof child_rel, "%s%s%s", rel,
		                  *rel ? "/" : "", ents[i]->d_name)
		             < (int)sizeof child_rel &&
		         snprintf(child, sizeof child, "%s/%s", root, child_rel)
		             < (int)sizeof child;
		struct stat st, lst;
		if (ok && lstat(child, &lst) == 0 && stat(child, &st) == 0) {
			/* Symlinked folders are not followed: a link back up
			 * the tree would walk forever. */
			if (S_ISDIR(st.st_mode) && !S_ISLNK(lst.st_mode))
				walk(j, root, child_rel, out_dir, out_real,
				     format);
			else if (S_ISREG(st.st_mode) && readable(child) &&
			         out_path(out, sizeof out, out_dir, child_rel,
			                  child, format) == 0)
				push(j, child, out);
		}
		free(ents[i]);
	}
	free(ents);
}

/*
 * Two inputs can map to one output: "x.webp" and "x.avif" both become
 * "x.jpg" under --format jpg, "a.png" and "a.gif" both become "a.png" under
 * the default, and two loose files from different folders can share a name.
 * Left alone, the second is then skipped as "already done" and silently never
 * censored -- the first test of --format did exactly that.
 *
 * So every clash is renamed apart, and said so: the input whose extension the
 * output already has keeps the plain name, the others get their own
 * extension kept in the name ("x.avif.jpg"), and exact duplicates are
 * numbered. Anything still clashing after that is refused, not overwritten.
 */
static const struct jobs *sort_jobs;

static int by_out(const void *a, const void *b)
{
	const int ia = *(const int *)a, ib = *(const int *)b;
	const char *oa = sort_jobs->v[ia].out, *ob = sort_jobs->v[ib].out;
	int c = strcmp(oa ? oa : "", ob ? ob : "");
	return c ? c : ia - ib;
}

static int *sorted_by_out(const struct jobs *j)
{
	int *ix = malloc((size_t)j->n * sizeof *ix);
	if (!ix)
		DIE("out of memory");
	for (int i = 0; i < j->n; i++)
		ix[i] = i;
	sort_jobs = j;
	qsort(ix, (size_t)j->n, sizeof *ix, by_out);
	return ix;
}

static void retag(struct job *jb, const char *tag)
{
	const char *oe = ext_of(jb->out);
	char buf[PATH_MAX];
	int stem = (int)(oe - 1 - jb->out);
	if (snprintf(buf, sizeof buf, "%.*s%s.%s", stem, jb->out, tag, oe)
	    >= (int)sizeof buf)
		return;
	WARN("%s and another input share an output name; writing %s", jb->in,
	     buf);
	free(jb->out);
	jb->out = strdup(buf);
}

/*
 * One file named twice -- on its own and inside a folder that was also given,
 * say -- is one image, not two. Compared by real path, so a relative and an
 * absolute spelling of it count as the same file. The first mention wins,
 * which keeps its place in the order.
 */
static void drop_repeats(struct jobs *j)
{
	struct jobs real = { 0 };
	for (int i = 0; i < j->n; i++) {
		char r[PATH_MAX];
		push(&real, realpath(j->v[i].in, r) ? r : j->v[i].in, "");
		free(real.v[i].out);
		real.v[i].out = real.v[i].in;   /* sort key */
	}
	int *ix = sorted_by_out(&real);
	char *drop = calloc((size_t)j->n, 1);
	for (int k = 1; k < j->n; k++)
		if (!strcmp(real.v[ix[k]].in, real.v[ix[k - 1]].in))
			drop[ix[k]] = 1;   /* ties sort by index: k is later */
	int n = 0;
	for (int i = 0; i < j->n; i++) {
		if (drop[i]) {
			free(j->v[i].in);
			free(j->v[i].out);
			continue;
		}
		j->v[n++] = j->v[i];
	}
	j->n = n;
	for (int i = 0; i < real.n; i++)
		free(real.v[i].in);
	free(real.v);
	free(drop);
	free(ix);
}

static int separate_clashes(struct jobs *j)
{
	drop_repeats(j);
	int *ix = sorted_by_out(j);
	for (int s = 0; s < j->n;) {
		int e = s + 1;
		while (e < j->n && !strcmp(j->v[ix[s]].out, j->v[ix[e]].out))
			e++;
		if (e - s > 1) {
			int kept = 0, dup = 1;
			for (int k = s; k < e; k++) {
				struct job *jb = &j->v[ix[k]];
				const char *ie = ext_of(jb->in);
				const char *oe = ext_of(jb->out);
				char tag[64];
				if (!strcasecmp(ie, oe) && !kept) {
					kept = 1;
					continue;
				}
				if (strcasecmp(ie, oe))
					snprintf(tag, sizeof tag, ".%s", ie);
				else
					snprintf(tag, sizeof tag, " (%d)", ++dup);
				retag(jb, tag);
			}
		}
		s = e;
	}
	free(ix);

	/* Whatever still clashes is refused. */
	int refused = 0;
	ix = sorted_by_out(j);
	for (int k = 1; k < j->n; k++)
		if (!strcmp(j->v[ix[k]].out, j->v[ix[k - 1]].out)) {
			ERR("%s: output name %s is taken by %s too; not "
			    "processing it", j->v[ix[k]].in, j->v[ix[k]].out,
			    j->v[ix[k - 1]].in);
			refused++;
		}
	free(ix);
	return refused;
}

static int mkdirs(const char *path)
{
	char p[PATH_MAX];
	if (snprintf(p, sizeof p, "%s", path) >= (int)sizeof p)
		return -1;
	for (char *s = p + 1; *s; s++) {
		if (*s != '/')
			continue;
		*s = 0;
		if (mkdir(p, 0755) != 0 && errno != EEXIST)
			return -1;
		*s = '/';
	}
	if (mkdir(p, 0755) != 0 && errno != EEXIST)
		return -1;
	return 0;
}

static int mkparent(const char *file)
{
	char p[PATH_MAX];
	if (snprintf(p, sizeof p, "%s", file) >= (int)sizeof p)
		return -1;
	char *slash = strrchr(p, '/');
	if (!slash || slash == p)
		return 0;
	*slash = 0;
	return mkdirs(p);
}

/* ------------------------------------------------------------------ */
/* Decode and encode                                                   */

struct image {
	int            w, h;
	unsigned char *rgba;     /* as decoded */
	unsigned char *rgb;      /* flattened over white: what is detected */
	int            has_alpha;
	int            animated; /* more than one frame; only the first is used */
	uint64_t       hash;     /* of the pixels; keys the hardening pattern */
};

static void image_free(struct image *im)
{
	free(im->rgba);
	free(im->rgb);
	memset(im, 0, sizeof *im);
}

/*
 * PAM, not raw RGB, because the header carries the size: ffmpeg applies a
 * JPEG's EXIF orientation while decoding, so a portrait phone photo comes out
 * with width and height swapped relative to what a probe of the file says.
 * Reading the size from the decoded frame itself cannot disagree with it.
 */
static int read_pam_header(FILE *f, int *w, int *h, int *depth)
{
	char line[128];
	*w = *h = *depth = 0;
	int maxval = 0;
	if (!fgets(line, sizeof line, f) || strncmp(line, "P7", 2))
		return -1;
	while (fgets(line, sizeof line, f)) {
		if (!strncmp(line, "ENDHDR", 6))
			return *w > 0 && *h > 0 && *depth == 4 &&
			       maxval == 255 ? 0 : -1;
		sscanf(line, "WIDTH %d", w);
		sscanf(line, "HEIGHT %d", h);
		sscanf(line, "DEPTH %d", depth);
		sscanf(line, "MAXVAL %d", &maxval);
	}
	return -1;
}

static int decode(const char *path, int max_side, struct image *im)
{
	memset(im, 0, sizeof *im);
	char q[PATH_MAX + 64], cmd[PATH_MAX + 256];
	if (bc_off_shq(q, sizeof q, path) != 0)
		return -1;
	/*
	 * The file is the decoder's stdin rather than its -i argument, so its
	 * name is never parsed by ffmpeg at all: a '%' would otherwise be read
	 * as an image-sequence pattern and a leading '-' as an option. Two
	 * frames are asked for so that an animation can be told apart from a
	 * still -- a still simply ends after one.
	 */
	snprintf(cmd, sizeof cmd,
	         "ffmpeg -v error -i - -frames:v 2 -f image2pipe -c:v pam "
	         "-pix_fmt rgba - < %s", q);
	FILE *f = popen(cmd, "r");
	if (!f)
		return -1;

	int w, h, depth;
	if (read_pam_header(f, &w, &h, &depth) != 0) {
		pclose(f);
		return -1;
	}
	if (w > max_side || h > max_side) {
		ERR("%dx%d is larger than this GPU's %d px limit", w, h,
		    max_side);
		pclose(f);
		return -1;
	}
	const size_t px = (size_t)w * h;
	im->w = w;
	im->h = h;
	im->rgba = malloc(px * 4);
	im->rgb  = malloc(px * 3);
	if (!im->rgba || !im->rgb || fread(im->rgba, 1, px * 4, f) != px * 4) {
		pclose(f);
		image_free(im);
		return -1;
	}
	/* Drain the rest; whatever is there is the second frame. */
	char sink[65536];
	size_t extra = 0, got;
	while ((got = fread(sink, 1, sizeof sink, f)) > 0)
		extra += got;
	im->animated = extra > 0;

	/*
	 * FNV-1a over the decoded pixels. Mixed into the run's seed, it gives
	 * every image its own hardening pattern -- so one original turning up
	 * beside its censored copy reveals that image's pattern and no other's
	 * -- while the preview, decoding the same pixels with the same run
	 * seed, draws exactly the pattern the batch will.
	 */
	uint64_t hv = 1469598103934665603ull;
	for (size_t i = 0; i < px * 4; i++)
		hv = (hv ^ im->rgba[i]) * 1099511628211ull;
	im->hash = hv;
	pclose(f);

	for (size_t i = 0; i < px; i++) {
		const unsigned a = im->rgba[i * 4 + 3];
		if (a != 255)
			im->has_alpha = 1;
		for (int c = 0; c < 3; c++)
			im->rgb[i * 3 + c] = (unsigned char)
			    ((im->rgba[i * 4 + c] * a + 255u * (255u - a) + 127u)
			     / 255u);
	}
	return 0;
}

/*
 * Write `pix` (RGB, or RGBA when `alpha`) to `path` via a hidden temporary
 * beside it, renamed into place only once the encoder reports success.
 */
static int encode(const char *path, const struct fmt *fm,
                  const unsigned char *pix, int w, int h, int alpha)
{
	char tmp[PATH_MAX];
	const char *slash = strrchr(path, '/');
	int dlen = slash ? (int)(slash - path + 1) : 0;
	if (snprintf(tmp, sizeof tmp, "%.*s.%s.bc-part.%s", dlen, path,
	             path + dlen, fm->ext) >= (int)sizeof tmp)
		return -1;

	char q[PATH_MAX + 64], cmd[PATH_MAX + 512];
	if (bc_off_shq(q, sizeof q, tmp) != 0)
		return -1;
	/*
	 * file: keeps the name literal on the way out, as stdin does on the
	 * way in, and -update 1 stops the image2 muxer reading a '%' in it
	 * as a sequence number.
	 */
	snprintf(cmd, sizeof cmd,
	         "ffmpeg -v error -y -f rawvideo -pix_fmt %s -s %dx%d -i - "
	         "-frames:v 1 %s -f %s %s file:%s",
	         alpha ? "rgba" : "rgb24", w, h, fm->codec, fm->muxer,
	         strcmp(fm->muxer, "image2") ? "" : "-update 1", q);
	FILE *f = popen(cmd, "w");
	if (!f)
		return -1;
	const size_t sz = (size_t)w * h * (alpha ? 4 : 3);
	const int wrote = fwrite(pix, 1, sz, f) == sz;
	const int rc = pclose(f);
	if (!wrote || rc != 0) {
		unlink(tmp);
		return -1;
	}
	if (rename(tmp, path) != 0) {
		ERR("could not move %s into place: %s", path, strerror(errno));
		unlink(tmp);
		return -1;
	}
	return 0;
}

/* ------------------------------------------------------------------ */
/* The per-image pipeline                                              */

/*
 * Everything that turns one decoded image into its censored pixels. A batch
 * and the GUI's preview (--serve) both go through it, so a preview is not a
 * likeness of the output: it is the output, short of the encoder.
 */
struct engine {
	const char         *cfg_path;
	char               *model;
	struct bc_config    cfg;
	struct stat         cfg_st;     /* to notice the file being saved */
	struct bc_detector *det;
	struct bc_preproc  *pp;
	struct bc_off_gl    g;
	float              *tensor;
	int                 tiles;
	int                 grid_cols, grid_rows, grid_full;
	float               grid_overlap;
	int                 test_box[4];
	unsigned char      *render;
	size_t              render_cap;
	int                 harden, badge;   /* both on unless asked not to */
	uint64_t            seed;            /* this run's hardening seed */
	struct bc_off_badge bdg;
};

/* The model and the config. Enough for --list-classes; no GL yet. */
static int engine_load(struct engine *e, const char *cfg_path,
                       const char *model)
{
	memset(e, 0, sizeof *e);
	e->cfg_path = cfg_path;
	bc_config_load(&e->cfg, cfg_path);
	if (!model && e->cfg.model[0])
		model = e->cfg.model;
	if (!model) {
		ERR("no model: pass --model or set one in the config");
		return -1;
	}
	e->model = strdup(model);
	/* 640x640 is a request; see the same call in video.c. */
	e->det = bc_detect_create(e->model, 640, 640, 0);
	if (!e->det) {
		ERR("could not load model '%s'", e->model);
		return -1;
	}
	/* Class names resolve against the model's taxonomy only from now. */
	bc_config_load(&e->cfg, cfg_path);
	stat(cfg_path, &e->cfg_st);
	return 0;
}

/* (Re)apply the config's section grid. A no-op when it has not changed. */
static int engine_grid(struct engine *e)
{
	const struct bc_config *c = &e->cfg;
	if (e->tensor && e->grid_cols == c->section_cols &&
	    e->grid_rows == c->section_rows &&
	    e->grid_full == c->section_full_frame &&
	    e->grid_overlap == c->section_overlap)
		return 0;
	bc_detect_wait_idle(e->det);
	if (bc_preproc_set_grid(e->pp, c->section_cols, c->section_rows,
	                        c->section_overlap,
	                        c->section_full_frame) != 0) {
		WARN("could not set a %dx%d grid; using the whole image",
		     c->section_cols, c->section_rows);
		bc_preproc_set_grid(e->pp, 1, 1, c->section_overlap, 0);
	}
	e->tiles = bc_preproc_tiles(e->pp);
	if (bc_detect_set_tiles(e->det, e->tiles) != 0) {
		ERR("detector could not take %d tiles", e->tiles);
		return -1;
	}
	free(e->tensor);
	e->tensor = malloc(bc_preproc_total_floats(e->pp) * sizeof *e->tensor);
	if (!e->tensor) {
		ERR("out of memory for the input tensor");
		return -1;
	}
	e->grid_cols    = c->section_cols;
	e->grid_rows    = c->section_rows;
	e->grid_full    = c->section_full_frame;
	e->grid_overlap = c->section_overlap;
	return 0;
}

/* GL, the preprocessor and the grid: what it takes to run an image. */
static int engine_start(struct engine *e)
{
	if (bc_off_gl_init(&e->g, 64, 64) != 0)
		return -1;
	e->pp = bc_preproc_create(bc_detect_net_w(e->det),
	                          bc_detect_net_h(e->det));
	if (!e->pp) {
		ERR("preprocess init failed");
		return -1;
	}
	if (engine_grid(e) != 0)
		return -1;
	if (getenv("BC_IMAGE_TEST_BOX"))
		sscanf(getenv("BC_IMAGE_TEST_BOX"), "%d,%d,%d,%d",
		       &e->test_box[0], &e->test_box[1], &e->test_box[2],
		       &e->test_box[3]);
	return 0;
}

/* The lowest detection floor among the classes in `cover` (all if NULL). */
static float floor_of(const struct engine *e, const int *cover)
{
	float m = 1.0f;
	for (int c = 0; c < bc_class_count() && c < BC_MAX_CLASSES; c++)
		if (!cover || cover[c]) {
			float f = bc_cfg_class_conf(&e->cfg, c);
			if (f < m)
				m = f;
		}
	return m;
}

/*
 * Run the detector over `im` and hand back every detection that clears its
 * class's floor, in image pixels -- for every class, covered or not, so the
 * preview can show what was found but left uncovered. The detector reports
 * down to `conf`.
 *
 * 0 on success; -1 when the detector gave no result (§6: the image must not
 * be written); -2 when it gave no answer at all (stop everything).
 * *saturated is set when the detection budget was full, in which case a
 * lower `conf` may have crowded out detections a higher one would keep.
 */
static int engine_detect(struct engine *e, const struct image *im, float conf,
                         struct bc_det *out, int *n_out, int *saturated)
{
	*n_out = 0;
	*saturated = 0;
	if (bc_off_gl_resize(&e->g, im->w, im->h) != 0)
		return -1;
	bc_off_upload(&e->g, im->rgb);
	bc_detect_set_conf(e->det, conf);

	struct bc_det dets[BC_MAX_DET];
	int hung = 0;
	struct bc_det_result res = bc_off_detect(&e->g, e->pp, e->det,
	                                         e->tensor, dets, BC_MAX_DET,
	                                         &hung);
	if (hung)
		return -2;
	if (res.status != BC_DET_OK)
		return -1;
	*saturated = res.n >= BC_MAX_DET;

	/* Tile network space -> image pixels, then the floors. */
	int n = 0;
	for (int d = 0; d < res.n && d < BC_MAX_DET; d++) {
		const int c = dets[d].cls;
		if (c < 0 || c >= BC_MAX_CLASSES ||
		    dets[d].score < bc_cfg_class_conf(&e->cfg, c))
			continue;
		float x0, y0, x1, y1;
		bc_preproc_tile_to_screen(e->pp, dets[d].tile, dets[d].x,
		                          dets[d].y, &x0, &y0);
		bc_preproc_tile_to_screen(e->pp, dets[d].tile,
		                          dets[d].x + dets[d].w,
		                          dets[d].y + dets[d].h, &x1, &y1);
		out[n] = dets[d];
		out[n].x = x0;
		out[n].y = y0;
		out[n].w = x1 - x0;
		out[n].h = y1 - y0;
		n++;
	}
	*n_out = n;
	return 0;
}

/*
 * The covers for one set of classes: the covered detections, rejoined across
 * tile seams, grown by the dilation, clamped, then joined. The class filter
 * comes before the seam merge, as it always has -- merging a covered box
 * with an uncovered neighbour of a "similar" class could otherwise hand the
 * union the uncovered class.
 *
 * `kept` (optional) receives the merged covered detections, before dilation.
 * `found` receives the covered class names, comma-separated.
 */
static int engine_boxes(struct engine *e, int w, int h,
                        const struct bc_det *raw, int n_raw, const int *cover,
                        struct bc_box *boxes, struct bc_det *kept,
                        int *n_kept, char *found, size_t found_cap)
{
	struct bc_det d[BC_MAX_DET];
	int n = 0;
	for (int i = 0; i < n_raw; i++)
		if (cover[raw[i].cls])
			d[n++] = raw[i];
	if (e->tiles > 1 && n > 1) {
		float ox, oy;
		bc_preproc_overlap_px(e->pp, &ox, &oy);
		n = bc_seam_merge(d, n, e->cfg.section_cols,
		                  bc_preproc_full_tile(e->pp), ox, oy,
		                  SEAM_THRESH, bc_off_classes_similar);
	}

	int nb = 0;
	if (found_cap)
		found[0] = 0;
	for (int i = 0; i < n && nb < MAX_BOX; i++) {
		/* §2's dilation, then clamp: a scissor outside the image
		 * draws nothing. */
		float dx = d[i].w * e->cfg.dilate / 200.0f;
		float dy = d[i].h * e->cfg.dilate / 200.0f;
		float x0 = d[i].x - dx, y0 = d[i].y - dy;
		float x1 = d[i].x + d[i].w + dx;
		float y1 = d[i].y + d[i].h + dy;
		if (x0 < 0) x0 = 0;
		if (y0 < 0) y0 = 0;
		if (x1 > w) x1 = (float)w;
		if (y1 > h) y1 = (float)h;
		if (x1 - x0 < 1.0f || y1 - y0 < 1.0f)
			continue;
		boxes[nb++] = bc_off_box(&e->cfg, d[i].cls, (int)x0, (int)y0,
		                         (int)(x1 - x0 + 0.999f),
		                         (int)(y1 - y0 + 0.999f));
		const char *cn = bc_class_name(d[i].cls);
		if (cn && found_cap && !strstr(found, cn) &&
		    strlen(found) + strlen(cn) + 3 < found_cap)
			snprintf(found + strlen(found), found_cap - strlen(found),
			         "%s%s", *found ? ", " : "", cn);
	}
	if (e->test_box[2] > 0 && nb < MAX_BOX)
		boxes[nb++] = bc_off_box(&e->cfg, -1, e->test_box[0],
		                         e->test_box[1], e->test_box[2],
		                         e->test_box[3]);
	if (e->cfg.join && nb > 1)
		nb = bc_join_boxes(boxes, nb, 6, e->cfg.join_waste);

	if (kept) {
		memcpy(kept, d, (size_t)n * sizeof *d);
		*n_kept = n;
	}
	return nb;
}

/*
 * Draw `boxes` over the uploaded image, harden them, and stamp the badge;
 * RGB24, top-down, or NULL on OOM. `hash` is the image's (struct image).
 * *badged says whether the badge went on -- an image too small to hold it
 * legibly is left without one.
 */
static unsigned char *engine_render(struct engine *e,
                                    const struct bc_box *boxes, int nb,
                                    uint64_t hash, int *badged)
{
	const size_t sz = (size_t)e->g.w * e->g.h * 3;
	if (sz > e->render_cap) {
		free(e->render);
		e->render = malloc(sz);
		e->render_cap = e->render ? sz : 0;
		if (!e->render)
			return NULL;
	}
	bc_off_draw(&e->g, boxes, nb, 0.0);
	if (e->harden)
		bc_off_harden(&e->g, boxes, nb, bc_off_mix_seed(e->seed, hash));
	bc_off_read(&e->g, e->render);
	*badged = 0;
	if (e->badge && bc_off_badge_make(&e->bdg, e->g.w, e->g.h) == 0) {
		bc_off_badge_apply(&e->bdg, e->render);
		*badged = 1;
	}
	return e->render;
}

static void engine_stop(struct engine *e, int hung)
{
	free(e->render);
	free(e->tensor);
	if (e->pp)
		bc_preproc_destroy(e->pp);
	/* Destroying joins the worker, which is exactly what has hung. */
	if (e->det && !hung)
		bc_detect_destroy(e->det);
	free(e->model);
	bc_off_badge_free(&e->bdg);
}

/*
 * "A,B,C" -> cover[]. "-" means the config's own selection, "all" every
 * class. -1 on a name the model does not have, after saying so.
 */
static int parse_cover(const struct engine *e, const char *list, int *cover)
{
	if (!list || !strcmp(list, "-")) {
		memcpy(cover, e->cfg.censor, sizeof(int) * BC_MAX_CLASSES);
		return 0;
	}
	memset(cover, 0, sizeof(int) * BC_MAX_CLASSES);
	char *copy = strdup(list), *save = NULL;
	int rc = 0;
	for (char *tok = strtok_r(copy, ", ", &save); tok;
	     tok = strtok_r(NULL, ", ", &save)) {
		if (!strcasecmp(tok, "all")) {
			for (int c = 0; c < bc_class_count() &&
			                c < BC_MAX_CLASSES; c++)
				cover[c] = 1;
			continue;
		}
		int c = bc_class_index(tok);
		if (c < 0 || c >= BC_MAX_CLASSES) {
			ERR("'%s' is not a class of %s. It has:", tok, e->model);
			for (int k = 0; k < bc_class_count(); k++)
				fprintf(stderr, "  %s\n", bc_class_name(k));
			rc = -1;
			break;
		}
		cover[c] = 1;
	}
	free(copy);
	return rc;
}

/* ------------------------------------------------------------------ */
/* --serve: the GUI's preview                                          */

static void reply_err(const char *msg)
{
	printf("ERR %s\n", msg);
	fflush(stdout);
}

static int same_stat(const struct stat *a, const struct stat *b)
{
	return a->st_size == b->st_size &&
	       a->st_mtim.tv_sec == b->st_mtim.tv_sec &&
	       a->st_mtim.tv_nsec == b->st_mtim.tv_nsec &&
	       a->st_ino == b->st_ino;
}

/*
 * One request per line on stdin:  CLASSES <tab> FLAGS <tab> PATH
 * (CLASSES as for --classes, or "-" for the config's; FLAGS holds 'h' to
 * harden and 'b' for the badge, "-" for neither). The hardening seed is the
 * process's, from BC_HARDEN_SEED: the GUI hands the same one to its batch,
 * which is what keeps the preview's pattern the batch's pattern. One reply
 * per request on stdout, either "ERR <message>" or:
 *
 *   OK <w> <h> <boxes> <dets> <animated>
 *   B <x> <y> <w> <h>                          once per cover drawn
 *   D <x> <y> <w> <h> <covered> <score> <name> once per detection
 *   <w*h*3 bytes>  the image as detected (RGB, top-down)
 *   <w*h*3 bytes>  the image as it would be written
 *
 * The model stays loaded between requests, and the last image's detections
 * are kept, so ticking a class on and off redraws without inferring again.
 * The config is re-read whenever its file changes, so the preview follows
 * the main settings panel; a different model there ends this process with
 * status 3, for the GUI to start a fresh one.
 */
static int serve(struct engine *e)
{
	struct image im = { 0 };
	char cached[PATH_MAX] = "";
	struct stat cached_st = { 0 };
	struct bc_det raw[BC_MAX_DET];
	int n_raw = 0, saturated = 0;
	static char line[PATH_MAX + 4096];

	printf("READY %d\n", bc_class_count());
	fflush(stdout);

	while (fgets(line, sizeof line, stdin)) {
		line[strcspn(line, "\n")] = 0;
		char *tab = strchr(line, '\t');
		if (!tab) {
			reply_err("bad request");
			continue;
		}
		*tab = 0;
		char *tab2 = strchr(tab + 1, '\t');
		if (!tab2) {
			reply_err("bad request");
			continue;
		}
		*tab2 = 0;
		const char *classes = line, *flags = tab + 1, *path = tab2 + 1;
		e->harden = strchr(flags, 'h') != NULL;
		e->badge  = strchr(flags, 'b') != NULL;

		struct stat st;
		if (stat(e->cfg_path, &st) == 0 && !same_stat(&st, &e->cfg_st)) {
			bc_config_load(&e->cfg, e->cfg_path);
			e->cfg_st = st;
			if (e->cfg.model[0] && strcmp(e->cfg.model, e->model)) {
				reply_err("the model was changed in the settings");
				image_free(&im);
				return 3;
			}
			if (engine_grid(e) != 0) {
				reply_err("could not apply the section grid");
				image_free(&im);
				return 1;
			}
			cached[0] = 0;   /* floors or the grid may differ */
		}

		int cover[BC_MAX_CLASSES];
		if (parse_cover(e, classes, cover) != 0) {
			reply_err("unknown class name");
			continue;
		}

		struct stat ist;
		if (stat(path, &ist) != 0) {
			reply_err("cannot read the file");
			continue;
		}
		if (strcmp(path, cached) || !same_stat(&ist, &cached_st)) {
			image_free(&im);
			cached[0] = 0;
			if (decode(path, e->g.max_tex, &im) != 0) {
				reply_err("could not decode the image");
				continue;
			}
			int r = engine_detect(e, &im, floor_of(e, NULL), raw,
			                      &n_raw, &saturated);
			if (r == -2) {
				reply_err("the detector stopped answering");
				image_free(&im);
				return 1;
			}
			if (r != 0) {
				reply_err("the detector returned no result -- a "
				          "batch would not write this image");
				image_free(&im);
				continue;
			}
			snprintf(cached, sizeof cached, "%s", path);
			cached_st = ist;
		}

		/*
		 * The cached pass ran at the lowest floor of every class. If
		 * that filled the detection budget, a batch -- which runs at
		 * the lowest floor of the covered classes only -- could keep
		 * detections this pass lost, so do exactly what it would.
		 */
		const struct bc_det *use = raw;
		int n_use = n_raw;
		struct bc_det exact[BC_MAX_DET];
		if (saturated) {
			int sat2;
			if (engine_detect(e, &im, floor_of(e, cover), exact,
			                  &n_use, &sat2) != 0) {
				reply_err("the detector returned no result");
				continue;
			}
			use = exact;
		}

		struct bc_box boxes[MAX_BOX];
		struct bc_det kept[BC_MAX_DET];
		int n_kept = 0;
		const int nb = engine_boxes(e, im.w, im.h, use, n_use, cover,
		                            boxes, kept, &n_kept, NULL, 0);
		int badged;
		const unsigned char *pix = engine_render(e, boxes, nb, im.hash,
		                                         &badged);
		if (!pix) {
			reply_err("out of memory");
			continue;
		}

		int n_unc = 0;
		for (int i = 0; i < n_raw; i++)
			n_unc += !cover[raw[i].cls];
		printf("OK %d %d %d %d %d\n", im.w, im.h, nb, n_kept + n_unc,
		       im.animated);
		for (int i = 0; i < nb; i++)
			printf("B %d %d %d %d\n", boxes[i].x, boxes[i].y,
			       boxes[i].w, boxes[i].h);
		for (int i = 0; i < n_kept; i++)
			printf("D %d %d %d %d 1 %.3f %s\n", (int)kept[i].x,
			       (int)kept[i].y, (int)kept[i].w, (int)kept[i].h,
			       kept[i].score, bc_class_name(kept[i].cls));
		for (int i = 0; i < n_raw; i++)
			if (!cover[raw[i].cls])
				printf("D %d %d %d %d 0 %.3f %s\n", (int)raw[i].x,
				       (int)raw[i].y, (int)raw[i].w,
				       (int)raw[i].h, raw[i].score,
				       bc_class_name(raw[i].cls));
		const size_t sz = (size_t)im.w * im.h * 3;
		if (fwrite(im.rgb, 1, sz, stdout) != sz ||
		    fwrite(pix, 1, sz, stdout) != sz || fflush(stdout) != 0)
			break;   /* the GUI went away */
	}
	image_free(&im);
	return 0;
}

/* ------------------------------------------------------------------ */

static void usage(void)
{
	fprintf(stderr,
	    "usage: bc-image [options] -o OUTDIR <image|folder>...\n"
	    "\n"
	    "Censors each image and writes the result into OUTDIR. A folder is\n"
	    "searched through, subfolders included, and its layout is copied\n"
	    "into OUTDIR. Effect, confidence and sectioning come from\n"
	    "betacensor.conf, the same file the live censor and the GUI use.\n"
	    "\n"
	    "  -o, --out DIR        where the censored images go (required)\n"
	    "  -c, --classes LIST   which classes to cover, comma-separated\n"
	    "                       (\"all\" for every class); default: the\n"
	    "                       classes betacensor.conf covers\n"
	    "  -f, --format FMT     output format: same (default; a GIF becomes\n"
	    "                       PNG), png, jpg, webp, bmp, tiff or avif\n"
	    "  -m, --model PATH     detector model (default: from the config)\n"
	    "      --overwrite      replace outputs that already exist; without\n"
	    "                       it they are skipped, so a stopped batch\n"
	    "                       resumes where it left off\n"
	    "      --list-classes   print the model's classes and exit:\n"
	    "                       index, name, 1 if the config covers it\n"
	    "      --no-harden      leave covers as drawn; by default those that\n"
	    "                       sample the image (pixelate, blur, ...) get a\n"
	    "                       random per-block shift, grain and a faint\n"
	    "                       watermark, to make them harder to reverse\n"
	    "      --no-badge       no \"censored with BetaCensor\" badge in the\n"
	    "                       bottom-right corner\n"
	    "      --serve          preview server for the GUI: requests on\n"
	    "                       stdin, rendered images on stdout (see\n"
	    "                       serve() in image.c for the protocol)\n"
	    "\n"
	    "Reads png, jpg, webp, bmp, gif, avif and tiff. An animated image\n"
	    "has only its first frame censored -- use bc-video for the whole\n"
	    "animation. Metadata (EXIF, GPS) is not copied to the output.\n"
	    "\n"
	    "  BC_IMAGE_CONFIG    config file (default betacensor.conf)\n"
	    "  BC_IMAGE_TEST_BOX  x,y,w,h: also cover this box in every image\n"
	    "  BC_HARDEN_SEED     hex seed for the hardening pattern (default:\n"
	    "                     random per run -- keep it secret if set)\n");
}

int main(int argc, char **argv)
{
	const char *out_dir = NULL, *classes = NULL, *model = NULL;
	const char *format = "same";
	int overwrite = 0, list_classes = 0, serve_mode = 0;
	int no_badge = 0, no_harden = 0;

	static const struct option opts[] = {
		{ "out",          required_argument, 0, 'o' },
		{ "classes",      required_argument, 0, 'c' },
		{ "format",       required_argument, 0, 'f' },
		{ "model",        required_argument, 0, 'm' },
		{ "overwrite",    no_argument,       0, 'O' },
		{ "list-classes", no_argument,       0, 'L' },
		{ "serve",        no_argument,       0, 'S' },
		{ "no-badge",     no_argument,       0, 'B' },
		{ "no-harden",    no_argument,       0, 'H' },
		{ "help",        no_argument,       0, 'h' },
		{ 0, 0, 0, 0 }
	};
	int ch;
	while ((ch = getopt_long(argc, argv, "o:c:f:m:h", opts, NULL)) != -1) {
		switch (ch) {
		case 'o': out_dir = optarg; break;
		case 'c': classes = optarg; break;
		case 'f': format = optarg; break;
		case 'm': model = optarg; break;
		case 'O': overwrite = 1; break;
		case 'L': list_classes = 1; break;
		case 'S': serve_mode = 1; break;
		case 'B': no_badge = 1; break;
		case 'H': no_harden = 1; break;
		case 'h': usage(); return 0;
		default:  usage(); return 2;
		}
	}
	if (strcmp(format, "same") && !fmt_for(format)) {
		ERR("unknown output format '%s'", format);
		return 2;
	}
	if (!list_classes && !serve_mode && (!out_dir || optind >= argc)) {
		usage();
		return 2;
	}

	signal(SIGPIPE, SIG_IGN);
	/*
	 * Stop between images, not inside one: killed mid-encode, the
	 * temporary file would be left behind, and the GUI's Stop button sends
	 * exactly this.
	 */
	signal(SIGINT, on_stop);
	signal(SIGTERM, on_stop);

	const char *cfg_path = getenv("BC_IMAGE_CONFIG");
	if (!cfg_path)
		cfg_path = "betacensor.conf";

	struct engine e;
	if (engine_load(&e, cfg_path, model) != 0)
		return 1;
	e.harden = !no_harden;
	e.badge  = !no_badge;
	e.seed   = bc_off_run_seed();

	if (list_classes) {
		for (int c = 0; c < bc_class_count(); c++)
			printf("%d\t%s\t%d\n", c, bc_class_name(c),
			       e.cfg.censor[c] ? 1 : 0);
		engine_stop(&e, 0);
		return 0;
	}
	if (serve_mode) {
		/* The GUI owns the lifetime: stdin closing is the signal. */
		signal(SIGINT, SIG_DFL);
		signal(SIGTERM, SIG_DFL);
		if (engine_start(&e) != 0)
			return 1;
		const int rc = serve(&e);
		engine_stop(&e, rc == 1);
		return rc;
	}

	/* ---- what to cover ---- */
	int cover[BC_MAX_CLASSES];
	if (parse_cover(&e, classes, cover) != 0)
		return 2;
	int n_cover = 0;
	for (int c = 0; c < bc_class_count() && c < BC_MAX_CLASSES; c++)
		n_cover += cover[c] != 0;
	if (!n_cover) {
		ERR("no classes selected to cover -- nothing would be censored");
		return 2;
	}
	/* The detector reports down to the lowest floor in use; each class
	 * is then held to its own. */
	const float floor_min = floor_of(&e, cover);

	/* ---- the work list ---- */
	if (mkdirs(out_dir) != 0) {
		ERR("cannot create output folder %s: %s", out_dir,
		    strerror(errno));
		return 1;
	}
	char out_real[PATH_MAX];
	if (!realpath(out_dir, out_real)) {
		ERR("cannot resolve %s: %s", out_dir, strerror(errno));
		return 1;
	}
	{
		char m[PATH_MAX];
		snprintf(m, sizeof m, "%s/%s", out_real, OUT_MARKER);
		FILE *mf = fopen(m, "a");
		if (mf)
			fclose(mf);
	}

	struct jobs jobs = { 0 };
	for (int i = optind; i < argc; i++) {
		const char *in = argv[i];
		struct stat st;
		if (stat(in, &st) != 0) {
			ERR("%s: %s", in, strerror(errno));
			return 1;
		}
		if (S_ISDIR(st.st_mode)) {
			char real[PATH_MAX];
			if (realpath(in, real) && !strcmp(real, out_real)) {
				ERR("the output folder cannot be an input folder "
				    "(%s) -- the censored copies would replace "
				    "the originals", in);
				return 2;
			}
			char root[PATH_MAX];
			snprintf(root, sizeof root, "%s", in);
			size_t l = strlen(root);
			while (l > 1 && root[l - 1] == '/')
				root[--l] = 0;
			walk(&jobs, root, "", out_real, out_real, format);
		} else {
			const char *base = strrchr(in, '/');
			base = base ? base + 1 : in;
			char out[PATH_MAX];
			if (out_path(out, sizeof out, out_real, base, in,
			             format) != 0) {
				ERR("path too long: %s", in);
				return 1;
			}
			push(&jobs, in, out);
		}
	}
	if (!jobs.n) {
		ERR("no images found");
		return 1;
	}
	if (separate_clashes(&jobs)) {
		ERR("some inputs could not be given an output name of their own "
		    "-- nothing was processed");
		return 2;
	}

	/* ---- the pipeline ---- */
	LOG("model  %s: net %dx%d, %s", e.model, bc_detect_net_w(e.det),
	    bc_detect_net_h(e.det), bc_detect_on_gpu(e.det) ? "GPU" : "CPU");
	if (engine_start(&e) != 0)
		return 1;
	LOG("%d image%s -> %s  (%d class%s covered, %d tile%s per image)",
	    jobs.n, jobs.n == 1 ? "" : "s", out_real, n_cover,
	    n_cover == 1 ? "" : "es", e.tiles, e.tiles == 1 ? "" : "s");

	int done = 0, skipped = 0, failed = 0, with_boxes = 0, hung = 0;
	long boxes_total = 0;
	unsigned char *outbuf = NULL;
	size_t outbuf_cap = 0;
	char in_real[PATH_MAX], dst_real[PATH_MAX];
	const double t0 = bc_off_now_ms();

	for (int i = 0; i < jobs.n && !stop_requested; i++) {
		const char *in = jobs.v[i].in, *out = jobs.v[i].out;
		const int k = i + 1;

		if (realpath(in, in_real) && realpath(out, dst_real) &&
		    !strcmp(in_real, dst_real)) {
			ERR("[%d/%d] FAILED %s: the output would replace the "
			    "input", k, jobs.n, in);
			failed++;
			continue;
		}
		if (!overwrite && access(out, F_OK) == 0) {
			LOG("[%d/%d] skip %s (already done)", k, jobs.n, out);
			skipped++;
			continue;
		}

		/*
		 * Ctrl-C reaches the ffmpeg children too, so a decode or
		 * encode that fails after a stop was asked for is the stop,
		 * not a bad image: the image is left for the resume.
		 */
		struct image im;
		if (decode(in, e.g.max_tex, &im) != 0) {
			if (stop_requested)
				break;
			ERR("[%d/%d] FAILED %s: could not decode it", k, jobs.n,
			    in);
			failed++;
			continue;
		}

		struct bc_det dets[BC_MAX_DET];
		int n = 0, saturated;
		const int r = engine_detect(&e, &im, floor_min, dets, &n,
		                            &saturated);
		if (r == -2) {
			ERR("[%d/%d] the detector gave no answer in 10 s -- "
			    "stopping", k, jobs.n);
			image_free(&im);
			failed++;
			hung = 1;
			break;
		}
		/*
		 * §6: nobody looked at this image. Writing it with no boxes
		 * would hand back an uncensored copy under a censored name.
		 */
		if (r != 0) {
			ERR("[%d/%d] FAILED %s: the detector returned no result; "
			    "nothing written", k, jobs.n, in);
			image_free(&im);
			failed++;
			continue;
		}

		struct bc_box boxes[MAX_BOX];
		char found[160];
		const int nb = engine_boxes(&e, im.w, im.h, dets, n, cover,
		                            boxes, NULL, NULL, found,
		                            sizeof found);
		int badged;
		const unsigned char *render = engine_render(&e, boxes, nb,
		                                            im.hash, &badged);
		const size_t px = (size_t)im.w * im.h;
		if (render && px * 4 > outbuf_cap) {
			free(outbuf);
			outbuf = malloc(px * 4);
			outbuf_cap = outbuf ? px * 4 : 0;
		}
		if (!render || !outbuf) {
			ERR("out of memory for a %dx%d image", im.w, im.h);
			image_free(&im);
			failed++;
			break;
		}

		/*
		 * Transparency survives where the format can hold it. Every
		 * pixel an effect touched -- inside a box, or changed at all,
		 * which also catches tape's full-width bars and static's
		 * feathered edge -- is taken from the render and made opaque,
		 * so no original colour hides under a transparent censor.
		 * Everything else is the source pixel exactly.
		 */
		const struct fmt *fm = fmt_for(ext_of(out));
		const int keep_alpha = im.has_alpha && fm && fm->alpha;
		const unsigned char *pix = render;
		if (keep_alpha) {
			for (size_t p = 0; p < px; p++) {
				const int x = (int)(p % im.w), y = (int)(p / im.w);
				int touched = memcmp(render + p * 3,
				                     im.rgb + p * 3, 3) != 0;
				for (int b = 0; b < nb && !touched; b++)
					touched = x >= boxes[b].x &&
					          y >= boxes[b].y &&
					          x < boxes[b].x + boxes[b].w &&
					          y < boxes[b].y + boxes[b].h;
				if (touched) {
					memcpy(outbuf + p * 4, render + p * 3, 3);
					outbuf[p * 4 + 3] = 255;
				} else {
					memcpy(outbuf + p * 4, im.rgba + p * 4, 4);
				}
			}
			pix = outbuf;
		}

		if (!fm || mkparent(out) != 0 ||
		    encode(out, fm, pix, im.w, im.h, keep_alpha) != 0) {
			if (stop_requested) {
				image_free(&im);
				break;
			}
			ERR("[%d/%d] FAILED %s: could not write %s", k, jobs.n,
			    in, out);
			image_free(&im);
			failed++;
			continue;
		}

		done++;
		boxes_total += nb;
		if (nb)
			with_boxes++;
		LOG("[%d/%d] %s -> %s: %d covered%s%s%s%s%s", k, jobs.n, in, out,
		    nb, *found ? " (" : "", found, *found ? ")" : "",
		    im.animated ? "  [animated: first frame only]" : "",
		    e.badge && !badged ? "  [too small for the badge]" : "");
		image_free(&im);
	}

	const int not_reached = jobs.n - done - skipped - failed;
	const double secs = (bc_off_now_ms() - t0) / 1000.0;
	LOG("summary");
	LOG("  images written          : %d", done);
	LOG("  with something covered  : %d", with_boxes);
	LOG("  boxes drawn             : %ld", boxes_total);
	LOG("  skipped (already done)  : %d", skipped);
	LOG("  failed                  : %d", failed);
	if (not_reached)
		LOG("  not reached             : %d", not_reached);
	LOG("  detector errors         : %lu", bc_detect_errors(e.det));
	LOG("  elapsed                 : %.1fs (%.2f s/image)", secs,
	    done ? secs / done : 0.0);

	free(outbuf);
	engine_stop(&e, hung);

	if (failed) {
		ERR("FAILED -- %d image%s not written; see above", failed,
		    failed == 1 ? "" : "s");
		return 1;
	}
	if (stop_requested && not_reached) {
		ERR("STOPPED -- %d image%s not reached; run again to resume",
		    not_reached, not_reached == 1 ? "" : "s");
		return 1;
	}
	LOG("SUCCESS -- %d censored, %d already done.", done, skipped);
	return 0;
}

