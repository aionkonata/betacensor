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
#include "anim.h"
#include "eyebar.h"
#include "landmark.h"
#include "offline.h"
#include "stream.h"
#include "util.h"

#include <dirent.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#define MAX_BOX 64
#define BC_LM_FACES 8   /* most faces handed to the landmark model at once */

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
 * Will this input be written as an animation? Only an animated GIF, WebP or
 * PNG whose target format can animate: GIF, WebP and PNG (as an APNG). Asked
 * for JPEG, BMP, TIFF or AVIF it is written as a still of its first frame,
 * since those cannot hold the rest.
 */
static int writes_animation(const char *in, const char *format)
{
	const char *ie = ext_of(in);
	if (strcasecmp(ie, "gif") && strcasecmp(ie, "webp") && strcasecmp(ie, "png"))
		return 0;
	const char *target = (format && strcmp(format, "same")) ? format : ie;
	return bc_anim_fmt_for(target) && bc_anim_is_animated(in);
}

/*
 * The output extension for an input. "same" keeps the input's own, except a
 * GIF that is not an animation, which becomes a PNG: a GIF written back from
 * a 24-bit render would be re-quantised to 256 colours for no reason, and with
 * one frame there is nothing a GIF holds that a PNG does not. Pury.fi makes
 * the same substitution. An animated one stays what it is.
 */
static const char *out_ext(const char *in, const char *format)
{
	const char *e = ext_of(in);
	if (writes_animation(in, format))
		return (format && strcmp(format, "same")) ? format : e;
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
	         "ffmpeg -v error -i - -fps_mode passthrough -frames:v 2 -f image2pipe -c:v pam "
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
	/* The eye bar's landmark model and counters; see offline.h. */
	struct bc_off_eb    eb;
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

/* The heads among d[], for the eye bar, and a way back to them. */
static void engine_bars(struct engine *e, const struct bc_det *d, int n,
                        struct bc_eyebar *bars, int *kind)
{
	struct bc_off_face faces[BC_MAX_DET];
	struct bc_eyebar   hb[BC_MAX_DET];
	int                hk[BC_MAX_DET], idx[BC_MAX_DET], nh = 0;
	for (int i = 0; i < n; i++) {
		kind[i] = 0;
		if (bc_is_head_class(bc_class_name(d[i].cls))) {
			faces[nh] = (struct bc_off_face){ d[i].x, d[i].y, d[i].w, d[i].h };
			idx[nh++] = i;
		}
	}
	bc_off_eb_bars(&e->eb, &e->cfg, &e->g, faces, nh, hb, hk);
	for (int m = 0; m < nh; m++) {
		bars[idx[m]] = hb[m];
		kind[idx[m]] = hk[m];
	}
}

static void note_class(char *found, size_t cap, const char *cn)
{
	if (cn && cap && !strstr(found, cn) &&
	    strlen(found) + strlen(cn) + 3 < cap)
		snprintf(found + strlen(found), cap - strlen(found), "%s%s",
		         *found ? ", " : "", cn);
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

	/* After the seam merge, so an eye cut by a tile edge is judged whole. */
	n = bc_off_drop_eyes(&e->cfg, d, n, &e->eb.eyes_dropped);

	struct bc_eyebar bars[BC_MAX_DET];
	int kind[BC_MAX_DET] = { 0 };
	if (e->cfg.eye_bar)
		engine_bars(e, d, n, bars, kind);

	int nb = 0;
	if (found_cap)
		found[0] = 0;
	for (int i = 0; i < n && nb < MAX_BOX; i++) {
		/*
		 * With the face left alone (eye_bar_face off) the bar *is* the
		 * cover, and nothing is drawn over the rest of the face.
		 */
		if (kind[i] && !e->cfg.eye_bar_face) {
			note_class(found, found_cap, bc_class_name(d[i].cls));
			continue;
		}
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
		note_class(found, found_cap, bc_class_name(d[i].cls));
	}
	if (e->test_box[2] > 0 && nb < MAX_BOX)
		boxes[nb++] = bc_off_box(&e->cfg, -1, e->test_box[0],
		                         e->test_box[1], e->test_box[2],
		                         e->test_box[3]);
	if (e->cfg.join && nb > 1)
		nb = bc_join_boxes(boxes, nb, 6, e->cfg.join_waste);

	/*
	 * The bars, last: drawn on top of the face cover they belong to, and
	 * after the join, which would otherwise merge an upright band into the
	 * face under it and keep the wrong effect. (bc_join_boxes already
	 * leaves turned boxes alone, for the same reason.)
	 *
	 * Grown by the dilation the way the live censor grows them. With the
	 * face covered too, the bar is laid out as fractions of the face and so
	 * scales about the face's centre; on its own it grows about its own.
	 */
	const float grow = 1.0f + e->cfg.dilate / 100.0f;
	for (int i = 0; i < n && nb < MAX_BOX; i++) {
		if (!kind[i])
			continue;
		const struct bc_off_face face = { d[i].x, d[i].y, d[i].w, d[i].h };
		struct bc_box bx;
		if (bc_off_eb_box(&e->cfg, d[i].cls, &face, &bars[i], grow, w, h, &bx))
			boxes[nb++] = bx;
	}

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
	bc_off_eb_free(&e->eb);
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
/* Manual edits                                                        */

static int same_stat(const struct stat *a, const struct stat *b)
{
	return a->st_size == b->st_size &&
	       a->st_mtim.tv_sec == b->st_mtim.tv_sec &&
	       a->st_mtim.tv_nsec == b->st_mtim.tv_nsec &&
	       a->st_ino == b->st_ino;
}

/*
 * What the user did by hand in the panel's preview, per image, so the batch
 * writes what they saw. One edit per line of a TSV file:
 *
 *   <path> TAB <w>x<h> TAB add  TAB x TAB y TAB w TAB h
 *   <path> TAB <w>x<h> TAB drop TAB x TAB y
 *
 * An *add* is a cover drawn by hand, with the global effect -- the backstop
 * for what the detector missed, like a fixed region in the live censor. A
 * *drop* is a point: "no cover where I clicked". It is a point and not the
 * box that was there so that it keeps working when the classes or the effect
 * change, which moves boxes but not the thing the user was pointing at.
 *
 * The path and the image's pixel size both have to match. A different file
 * written over the same name has different pixels, and coordinates from the
 * old one would cover the wrong place -- silently, which is worse than not
 * applying them.
 *
 * These are the only way covers are ever removed. They are explicit, per
 * image, and shown in the preview; nothing here removes a cover on its own.
 */
struct edit {
	char *path;           /* realpath'd where the file still exists */
	int   iw, ih;
	int   kind;           /* 0 add, 1 drop */
	int   x, y, w, h;
};

struct edits {
	char        *file;
	struct edit *v;
	int          n, cap;
	struct stat  st;
	int          have_st;
};

static void edits_clear(struct edits *ed)
{
	for (int i = 0; i < ed->n; i++)
		free(ed->v[i].path);
	free(ed->v);
	ed->v = NULL;
	ed->n = ed->cap = 0;
}

static void edits_load(struct edits *ed)
{
	edits_clear(ed);
	ed->have_st = ed->file && stat(ed->file, &ed->st) == 0;
	FILE *f = ed->file ? fopen(ed->file, "r") : NULL;
	if (!f)
		return;
	static char line[PATH_MAX + 256];
	while (fgets(line, sizeof line, f)) {
		line[strcspn(line, "\r\n")] = 0;
		char *save = NULL;
		char *path = strtok_r(line, "\t", &save);
		char *dim  = strtok_r(NULL, "\t", &save);
		char *kind = strtok_r(NULL, "\t", &save);
		if (!path || !dim || !kind)
			continue;
		struct edit e = { 0 };
		if (sscanf(dim, "%dx%d", &e.iw, &e.ih) != 2)
			continue;
		const char *a = strtok_r(NULL, "\t", &save);
		const char *b = strtok_r(NULL, "\t", &save);
		const char *c = strtok_r(NULL, "\t", &save);
		const char *d = strtok_r(NULL, "\t", &save);
		if (!strcmp(kind, "add") && a && b && c && d) {
			e.kind = 0;
			e.x = atoi(a); e.y = atoi(b); e.w = atoi(c); e.h = atoi(d);
			if (e.w < 1 || e.h < 1)
				continue;
		} else if (!strcmp(kind, "drop") && a && b) {
			e.kind = 1;
			e.x = atoi(a); e.y = atoi(b);
		} else {
			continue;
		}
		char rp[PATH_MAX];
		e.path = strdup(realpath(path, rp) ? rp : path);
		if (ed->n == ed->cap) {
			ed->cap = ed->cap ? ed->cap * 2 : 32;
			ed->v = realloc(ed->v, (size_t)ed->cap * sizeof *ed->v);
			if (!ed->v)
				DIE("out of memory reading edits");
		}
		ed->v[ed->n++] = e;
	}
	fclose(f);
}

/* Re-read when the file changed: the panel rewrites it on every edit. */
static void edits_refresh(struct edits *ed)
{
	if (!ed->file)
		return;
	struct stat st;
	const int have = stat(ed->file, &st) == 0;
	if (have != ed->have_st || (have && !same_stat(&st, &ed->st)))
		edits_load(ed);
}

/*
 * Apply this image's edits to its covers: drops first (removing every cover
 * that contains a dropped point, order kept so bars stay on top), then adds
 * appended after. `removed` receives the boxes taken out, for the preview.
 * Returns the new count; *first_manual is where the adds begin.
 */
static int edits_apply(const struct edits *ed, const char *real, int iw, int ih,
                       const struct bc_config *cfg, struct bc_box *boxes,
                       int nb, struct bc_box *removed, int *n_removed,
                       int *first_manual)
{
	*n_removed = 0;
	for (int i = 0; i < ed->n; i++) {
		const struct edit *e = &ed->v[i];
		if (e->kind != 1 || e->iw != iw || e->ih != ih ||
		    strcmp(e->path, real))
			continue;
		for (int b = 0; b < nb;) {
			if (!bc_off_box_has(&boxes[b], e->x, e->y)) {
				b++;
				continue;
			}
			if (*n_removed < MAX_BOX)
				removed[(*n_removed)++] = boxes[b];
			memmove(&boxes[b], &boxes[b + 1],
			        (size_t)(nb - b - 1) * sizeof *boxes);
			nb--;
		}
	}
	*first_manual = nb;
	for (int i = 0; i < ed->n && nb < MAX_BOX; i++) {
		const struct edit *e = &ed->v[i];
		if (e->kind != 0 || e->iw != iw || e->ih != ih ||
		    strcmp(e->path, real))
			continue;
		int x0 = e->x < 0 ? 0 : e->x, y0 = e->y < 0 ? 0 : e->y;
		int x1 = e->x + e->w > iw ? iw : e->x + e->w;
		int y1 = e->y + e->h > ih ? ih : e->y + e->h;
		if (x1 - x0 < 1 || y1 - y0 < 1)
			continue;
		boxes[nb++] = bc_off_box(cfg, -1, x0, y0, x1 - x0, y1 - y0);
	}
	return nb;
}

/* Did an edit take this box out? Judged by overlap with what was removed: a
 * bar sits inside its face, so a removed bar must count for the face too. */
static int was_removed(const struct bc_box *removed, int n_removed,
                       float x, float y, float w, float h)
{
	for (int i = 0; i < n_removed; i++) {
		const float ox = fminf(x + w, removed[i].x + removed[i].w) -
		                 fmaxf(x, removed[i].x);
		const float oy = fminf(y + h, removed[i].y + removed[i].h) -
		                 fmaxf(y, removed[i].y);
		if (ox <= 0 || oy <= 0)
			continue;
		const float small = fminf(w * h, (float)removed[i].w * removed[i].h);
		if (small > 0 && ox * oy >= 0.5f * small)
			return 1;
	}
	return 0;
}

/* ------------------------------------------------------------------ */
/* --serve: the GUI's preview                                          */

static void reply_err(const char *msg)
{
	printf("ERR %s\n", msg);
	fflush(stdout);
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
 *   B <x> <y> <w> <h> <angle> <manual>         once per cover drawn; angle in
 *                                              radians, GL's (an eye bar is
 *                                              turned); manual is 1 for one
 *                                              the user drew
 *   D <x> <y> <w> <h> <state> <score> <name>   once per detection; state 1
 *                                              covered, 0 found but not
 *                                              ticked, 2 covered until the
 *                                              user removed it
 *   <w*h*3 bytes>  the image as detected (RGB, top-down)
 *   <w*h*3 bytes>  the image as it would be written
 *
 * The model stays loaded between requests, and the last image's detections
 * are kept, so ticking a class on and off redraws without inferring again.
 * The config is re-read whenever its file changes, so the preview follows
 * the main settings panel; a different model there ends this process with
 * status 3, for the GUI to start a fresh one.
 */
static int serve(struct engine *e, struct edits *ed)
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

		struct bc_box boxes[MAX_BOX], removed[MAX_BOX];
		struct bc_det kept[BC_MAX_DET];
		int n_kept = 0, n_removed = 0, first_manual;
		int nb = engine_boxes(e, im.w, im.h, use, n_use, cover,
		                      boxes, kept, &n_kept, NULL, 0);
		/* The user's own edits, from the file the panel keeps. */
		char real[PATH_MAX];
		if (!realpath(path, real))
			snprintf(real, sizeof real, "%s", path);
		edits_refresh(ed);
		nb = edits_apply(ed, real, im.w, im.h, &e->cfg, boxes, nb, removed,
		                 &n_removed, &first_manual);
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
			printf("B %d %d %d %d %.5f %d\n", boxes[i].x, boxes[i].y,
			       boxes[i].w, boxes[i].h, boxes[i].angle,
			       i >= first_manual);
		/* State: 1 covered, 2 covered until you removed it. */
		for (int i = 0; i < n_kept; i++)
			printf("D %d %d %d %d %d %.3f %s\n", (int)kept[i].x,
			       (int)kept[i].y, (int)kept[i].w, (int)kept[i].h,
			       was_removed(removed, n_removed, kept[i].x, kept[i].y,
			                   kept[i].w, kept[i].h) ? 2 : 1,
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
/* Animations                                                          */

/*
 * An animated GIF, WebP or APNG is a video with a duration per frame, so it is
 * censored as one: through the same stream as bc-video (stream.c), which keeps
 * a tracker and each eye bar's history from frame to frame. Censoring the
 * frames as independent stills would be simpler and worse -- a detection the
 * model misses on one frame would flash uncovered for that frame, where the
 * tracker carries the box through it.
 *
 * Differences from a video worth knowing: no tile grid (as in bc-video, and
 * animations are small); the user's hand edits apply to *every* frame, which is
 * the only reading of "cover here" / "no cover here" that makes sense for a
 * picture that moves; and time on the tracker's clock is the sum of the
 * frames' own durations, so a slow frame is a long gap to the tracker too.
 */

struct anim_edit_ctx {
	const struct edits     *ed;
	const struct bc_config *cfg;
	char                    real[PATH_MAX];
	int                     w, h;
};

static int anim_edit_adjust(void *user, struct bc_box *boxes, int nb, int max)
{
	const struct anim_edit_ctx *c = user;
	struct bc_box removed[MAX_BOX];
	int n_removed, first_manual;
	(void)max;
	return edits_apply(c->ed, c->real, c->w, c->h, c->cfg, boxes, nb, removed,
	                   &n_removed, &first_manual);
}

enum { ANIM_OK = 0, ANIM_FAILED = -1, ANIM_HUNG = -2, ANIM_STOPPED = -3 };

struct anim_result {
	long   frames, censored, reused;
	double seconds, play_ms;
};

static int censor_animation(struct engine *e, const int *cover,
                            const struct edits *ed, const char *in,
                            const char *out, struct anim_result *res)
{
	memset(res, 0, sizeof *res);
	const struct bc_anim_fmt *af = bc_anim_fmt_for(ext_of(out));
	int w, h;
	struct bc_anim_reader *rd = af ? bc_anim_open(in, &w, &h) : NULL;
	if (!rd) {
		ERR("%s: could not decode it as an animation", in);
		return ANIM_FAILED;
	}
	if (bc_off_gl_resize(&e->g, w, h) != 0) {
		bc_anim_close(rd);
		return ANIM_FAILED;
	}
	/*
	 * The frames are staged on disk until the encoder has seen them all, at
	 * 4 bytes a pixel. A long or large animation can want gigabytes, and
	 * running out of room halfway through is a worse way to find that out
	 * than being told first.
	 */
	{
		char dir[PATH_MAX];
		snprintf(dir, sizeof dir, "%s", out);
		char *slash = strrchr(dir, '/');
		if (slash)
			*slash = 0;
		else
			snprintf(dir, sizeof dir, ".");
		struct statvfs sv;
		const double need = (double)bc_anim_expected(rd) * w * h * 4.0;
		if (need > 0 && statvfs(dir, &sv) == 0 &&
		    need > 0.8 * (double)sv.f_bavail * (double)sv.f_frsize) {
			ERR("%s: %d frames of %dx%d need about %.1f GB of working space "
			    "beside the output, and %.1f GB is free", in,
			    bc_anim_expected(rd), w, h, need / 1e9,
			    (double)sv.f_bavail * (double)sv.f_frsize / 1e9);
			bc_anim_close(rd);
			return ANIM_FAILED;
		}
	}
	const double t0 = bc_off_now_ms();

	/*
	 * A stream wants one tile per frame and the video-style tracker
	 * confidence; the engine's grid is put back afterwards. The preprocessor
	 * here is the plain 1x1 one, so the engine's own is left untouched.
	 */
	bc_detect_wait_idle(e->det);
	struct bc_preproc *pp = bc_preproc_create(bc_detect_net_w(e->det),
	                                          bc_detect_net_h(e->det));
	float *tensor = pp ? malloc(bc_preproc_total_floats(pp) * sizeof *tensor)
	                   : NULL;
	const size_t px = (size_t)w * h;
	unsigned char *rgba = malloc(px * 4), *flat = malloc(px * 3);
	unsigned char *rend = malloc(px * 3), *outa = malloc(px * 4);
	if (!pp || !tensor || !rgba || !flat || !rend || !outa ||
	    bc_detect_set_tiles(e->det, 1) != 0) {
		ERR("%s: out of memory or no detector for an animation", in);
		bc_anim_close(rd);
		free(rgba); free(flat); free(rend); free(outa); free(tensor);
		if (pp)
			bc_preproc_destroy(pp);
		return ANIM_FAILED;
	}
	bc_detect_set_conf(e->det, e->cfg.confidence * 0.5f);

	struct bc_off_badge bdg = { 0 };
	const int badge_on = e->badge && bc_off_badge_make(&bdg, w, h) == 0;

	struct anim_edit_ctx ctx = { .ed = ed, .cfg = &e->cfg, .w = w, .h = h };
	if (ed && ed->file) {
		if (!realpath(in, ctx.real))
			snprintf(ctx.real, sizeof ctx.real, "%s", in);
	}
	struct bc_stream st = {
		.cfg = &e->cfg, .cover = cover, .det = e->det, .g = &e->g, .pp = pp,
		.tensor = tensor, .bdg = badge_on ? &bdg : NULL, .harden = e->harden,
		.test_box = { e->test_box[0], e->test_box[1], e->test_box[2],
		              e->test_box[3] },
	};
	if (ed && ed->file) {
		st.adjust = anim_edit_adjust;
		st.adjust_user = &ctx;
	}
	bc_stream_init(&st);

	struct bc_anim_writer *wr = bc_anim_begin(out, af, w, h);
	int rc = wr ? ANIM_OK : ANIM_FAILED;
	double t_ms = 0.0, dur;
	int r;
	while (rc == ANIM_OK && (r = bc_anim_next(rd, rgba, &dur)) == 1) {
		if (stop_requested) {
			rc = ANIM_STOPPED;
			break;
		}
		int has_alpha = 0;
		uint64_t hv = 1469598103934665603ull;
		for (size_t p = 0; p < px; p++) {
			const unsigned a = rgba[p * 4 + 3];
			has_alpha |= a != 255;
			for (int c = 0; c < 3; c++)
				flat[p * 3 + c] = (unsigned char)
				    ((rgba[p * 4 + c] * a + 255u * (255u - a) + 127u) / 255u);
		}
		/* One hardening pattern for the whole animation, keyed on its first
		 * frame: fixed to the picture and not to time, as in a video. */
		if (st.frames == 0) {
			for (size_t i = 0; i < px * 4; i++)
				hv = (hv ^ rgba[i]) * 1099511628211ull;
			st.harden_seed = bc_off_mix_seed(e->seed, hv);
		}
		struct bc_box boxes[BC_STREAM_MAX_BOX];
		int nb = 0;
		const int sr = bc_stream_frame(&st, flat, t_ms, rend, boxes, &nb);
		if (sr == BC_STREAM_HUNG) {
			rc = ANIM_HUNG;
			break;
		}
		if (sr == BC_STREAM_BLANKED) {
			/* §6: nobody looked at this frame, so the animation is not
			 * written at all -- a black frame in the middle of someone's GIF
			 * is not a result either. */
			rc = ANIM_FAILED;
			break;
		}
		if (af->alpha && has_alpha)
			bc_off_compose_alpha(rend, flat, rgba, w, h, boxes, nb, outa);
		else
			for (size_t p = 0; p < px; p++) {
				memcpy(outa + p * 4, rend + p * 3, 3);
				outa[p * 4 + 3] = 255;
			}
		if (bc_anim_add(wr, outa, dur, af->alpha && has_alpha) != 0) {
			ERR("%s: could not stage a frame", in);
			rc = ANIM_FAILED;
			break;
		}
		t_ms += dur;
	}
	if (rc == ANIM_OK && r < 0) {
		ERR("%s: the animation could not be read to its end", in);
		rc = ANIM_FAILED;
	}
	if (rc == ANIM_OK && st.frames == 0) {
		ERR("%s: no frames decoded", in);
		rc = ANIM_FAILED;
	}
	if (rc == ANIM_OK) {
		if (bc_anim_finish(wr) != 0) {
			ERR("%s: could not write %s", in, out);
			rc = ANIM_FAILED;
		}
	} else if (wr) {
		bc_anim_abort(wr);
	}
	if (st.blanked_frames)
		ERR("%s: %ld frame%s had no detection result; nothing written", in,
		    st.blanked_frames, st.blanked_frames == 1 ? "" : "s");

	res->frames = st.frames;
	res->censored = st.censored_frames;
	res->reused = (long)st.reused;
	res->play_ms = t_ms;
	res->seconds = (bc_off_now_ms() - t0) / 1000.0;

	bc_stream_free(&st);
	bc_anim_close(rd);
	bc_off_badge_free(&bdg);
	free(rgba); free(flat); free(rend); free(outa); free(tensor);
	bc_preproc_destroy(pp);
	/* Back to the grid the stills use. */
	bc_detect_wait_idle(e->det);
	bc_detect_set_tiles(e->det, e->tiles);
	return rc;
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
	    "      --edits FILE     covers drawn and removed by hand in the panel's\n"
	    "                       preview (see edits_load() in image.c for the\n"
	    "                       format); applied to the images they name\n"
	    "      --serve          preview server for the GUI: requests on\n"
	    "                       stdin, rendered images on stdout (see\n"
	    "                       serve() in image.c for the protocol)\n"
	    "\n"
	    "Reads png, jpg, webp, bmp, gif, avif and tiff. An animated GIF, WebP\n"
	    "or PNG is censored frame by frame, with tracking between frames, and\n"
	    "written back animated with every frame's own timing. Asked for jpg,\n"
	    "bmp, tiff or avif, it is written as a still of its first frame.\n"
	    "Metadata (EXIF, GPS) is not copied to the output.\n"
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
	const char *edits_file = NULL;

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
		{ "edits",        required_argument, 0, 'E' },
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
		case 'E': edits_file = optarg; break;
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

	struct edits edits = { .file = (char *)edits_file };
	if (edits_file)
		edits_load(&edits);
	long edits_applied_removed = 0, edits_applied_added = 0;

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
		const int rc = serve(&e, &edits);
		engine_stop(&e, rc == 1);
		edits_clear(&edits);
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
	int anims_done = 0;
	long anim_frames = 0;
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

		/* An animation is a sequence, and goes through the stream. */
		if (writes_animation(in, format)) {
			if (mkparent(out) != 0) {
				ERR("[%d/%d] FAILED %s: cannot create %s", k, jobs.n, in,
				    out);
				failed++;
				continue;
			}
			struct anim_result ar;
			const int arc = censor_animation(&e, cover, edits.file ? &edits : NULL,
			                                 in, out, &ar);
			if (arc == ANIM_STOPPED)
				break;
			if (arc == ANIM_HUNG) {
				ERR("[%d/%d] the detector gave no answer in 10 s -- "
				    "stopping", k, jobs.n);
				failed++;
				hung = 1;
				break;
			}
			if (arc != ANIM_OK) {
				ERR("[%d/%d] FAILED %s: animation not written", k, jobs.n, in);
				failed++;
				continue;
			}
			done++;
			anims_done++;
			anim_frames += ar.frames;
			if (ar.censored)
				with_boxes++;
			LOG("[%d/%d] %s -> %s: animation, %ld frame%s over %.2f s, %ld "
			    "with a cover%s%s", k, jobs.n, in, out, ar.frames,
			    ar.frames == 1 ? "" : "s", ar.play_ms / 1000.0, ar.censored,
			    ar.reused ? "  [repeated frames reused their detections]" : "",
			    "");
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

		struct bc_box boxes[MAX_BOX], removed[MAX_BOX];
		char found[160];
		int nb = engine_boxes(&e, im.w, im.h, dets, n, cover,
		                      boxes, NULL, NULL, found, sizeof found);
		/* What the user did by hand in the preview, if they gave us it. */
		int n_removed = 0, first_manual = nb;
		if (edits.file) {
			char rp[PATH_MAX];
			if (!realpath(in, rp))
				snprintf(rp, sizeof rp, "%s", in);
			nb = edits_apply(&edits, rp, im.w, im.h, &e.cfg, boxes, nb,
			                 removed, &n_removed, &first_manual);
		}
		const int n_added = nb - first_manual;
		edits_applied_removed += n_removed;
		edits_applied_added += n_added;
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
			bc_off_compose_alpha(render, im.rgb, im.rgba, im.w, im.h, boxes,
			                     nb, outbuf);
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
		char edited[96] = "";
		if (n_removed || n_added)
			snprintf(edited, sizeof edited, "  [your edits: +%d drawn, -%d removed]",
			         n_added, n_removed);
		LOG("[%d/%d] %s -> %s: %d covered%s%s%s%s%s%s", k, jobs.n, in, out,
		    nb, *found ? " (" : "", found, *found ? ")" : "",
		    edited,
		    im.animated ? "  [animated: first frame only]" : "",
		    e.badge && !badged ? "  [too small for the badge]" : "");
		image_free(&im);
	}

	const int not_reached = jobs.n - done - skipped - failed;
	const double secs = (bc_off_now_ms() - t0) / 1000.0;
	LOG("summary");
	LOG("  images written          : %d", done);
	LOG("  with something covered  : %d", with_boxes);
	if (anims_done)
		LOG("  of which animations     : %d (%ld frames)", anims_done,
		    anim_frames);
	LOG("  boxes drawn             : %ld", boxes_total);
	LOG("  skipped (already done)  : %d", skipped);
	LOG("  failed                  : %d", failed);
	if (not_reached)
		LOG("  not reached             : %d", not_reached);
	if (e.cfg.eye_bar && (e.eb.bars || e.eb.bands))
		LOG("  eye bars                : %lu from landmarks, %lu as the band "
		    "(%lu weak, %lu implausible), %lu EYE boxes left to them",
		    e.eb.bars, e.eb.bands, e.eb.weak, e.eb.rejected,
		    e.eb.eyes_dropped);
	if (edits.file)
		LOG("  your edits              : %ld drawn, %ld removed, across the "
		    "images they name", edits_applied_added, edits_applied_removed);
	LOG("  detector errors         : %lu", bc_detect_errors(e.det));
	LOG("  elapsed                 : %.1fs (%.2f s/image)", secs,
	    done ? secs / done : 0.0);

	edits_clear(&edits);
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

