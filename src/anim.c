/*
 * Animated image input and output. See anim.h.
 */

#include "anim.h"
#include "offline.h"
#include "util.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

static const struct bc_anim_fmt FORMATS[] = {
	{ "gif",  "gif",  1 },
	{ "webp", "webp", 1 },
	{ "png",  "apng", 1 },
	{ NULL, NULL, 0 }
};

const struct bc_anim_fmt *bc_anim_fmt_for(const char *ext)
{
	for (int i = 0; FORMATS[i].ext; i++)
		if (!strcasecmp(ext, FORMATS[i].ext))
			return &FORMATS[i];
	return NULL;
}

static const char *ext_of(const char *path)
{
	const char *slash = strrchr(path, '/');
	const char *dot   = strrchr(path, '.');
	if (!dot || (slash && dot < slash) || dot == path ||
	    (slash && dot == slash + 1))
		return "";
	return dot + 1;
}

/* ------------------------------------------------------------------ */
/* Is it an animation?                                                 */

/* PNG: an acTL chunk before the first IDAT makes it an APNG. */
static int png_is_animated(const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f)
		return 0;
	unsigned char sig[8];
	int anim = 0;
	if (fread(sig, 1, 8, f) == 8 && !memcmp(sig, "\x89PNG\r\n\x1a\n", 8)) {
		for (int chunks = 0; chunks < 64; chunks++) {
			unsigned char hdr[8];
			if (fread(hdr, 1, 8, f) != 8)
				break;
			const unsigned long len = ((unsigned long)hdr[0] << 24) |
			                          (hdr[1] << 16) | (hdr[2] << 8) | hdr[3];
			if (!memcmp(hdr + 4, "acTL", 4)) {
				anim = 1;
				break;
			}
			if (!memcmp(hdr + 4, "IDAT", 4))
				break;       /* the animation control comes before this */
			if (fseek(f, (long)len + 4, SEEK_CUR) != 0)   /* data + CRC */
				break;
		}
	}
	fclose(f);
	return anim;
}

int bc_anim_is_animated(const char *path)
{
	const char *e = ext_of(path);
	if (!strcasecmp(e, "png"))
		return png_is_animated(path);
	if (strcasecmp(e, "gif") && strcasecmp(e, "webp"))
		return 0;

	char q[4200], cmd[4600], line[64];
	if (bc_off_shq(q, sizeof q, path) != 0)
		return 0;
	snprintf(cmd, sizeof cmd,
	         "ffprobe -v error -count_packets -select_streams v:0 "
	         "-show_entries stream=nb_read_packets -of csv=p=0 -i - < %s "
	         "2>/dev/null", q);
	FILE *f = popen(cmd, "r");
	if (!f)
		return 0;
	int n = 0;
	if (fgets(line, sizeof line, f))
		n = atoi(line);
	pclose(f);
	return n > 1;
}

/* ------------------------------------------------------------------ */
/* Reading                                                             */

struct bc_anim_reader {
	FILE   *dec;
	int     w, h;
	size_t  px;
	double *dur;          /* the container's durations, ms, in frame order */
	int     n_dur, idx;
	int     have_first;   /* the header of frame 0 was read to learn the size */
};

/*
 * Per-frame durations from the container, in ms. Not every frame has one:
 * ffprobe says N/A for some, and a GIF's zero delay means "as fast as you can",
 * which every viewer renders as ~100 ms. Those are filled with the same
 * 100 ms, so a file plays here as it does where it came from.
 */
static void read_durations(struct bc_anim_reader *r, const char *path)
{
	char q[4200], cmd[4600], line[128];
	if (bc_off_shq(q, sizeof q, path) != 0)
		return;
	snprintf(cmd, sizeof cmd,
	         "ffprobe -v error -select_streams v:0 -show_entries "
	         "packet=duration_time -of csv=p=0 -i - < %s 2>/dev/null", q);
	FILE *f = popen(cmd, "r");
	if (!f)
		return;
	int cap = 0;
	while (fgets(line, sizeof line, f)) {
		if (r->n_dur == cap) {
			cap = cap ? cap * 2 : 64;
			r->dur = realloc(r->dur, (size_t)cap * sizeof *r->dur);
			if (!r->dur) {
				r->n_dur = 0;
				break;
			}
		}
		const double s = strtod(line, NULL);
		r->dur[r->n_dur++] = s > 0.0 ? s * 1000.0 : 0.0;
	}
	pclose(f);
}

static int read_pam_header(FILE *f, int *w, int *h)
{
	char line[128];
	int depth = 0, maxval = 0;
	*w = *h = 0;
	if (!fgets(line, sizeof line, f) || strncmp(line, "P7", 2))
		return -1;
	while (fgets(line, sizeof line, f)) {
		if (!strncmp(line, "ENDHDR", 6))
			return *w > 0 && *h > 0 && depth == 4 && maxval == 255 ? 0 : -1;
		sscanf(line, "WIDTH %d", w);
		sscanf(line, "HEIGHT %d", h);
		sscanf(line, "DEPTH %d", &depth);
		sscanf(line, "MAXVAL %d", &maxval);
	}
	return -1;
}

struct bc_anim_reader *bc_anim_open(const char *path, int *w, int *h)
{
	char q[4200], cmd[4600];
	if (bc_off_shq(q, sizeof q, path) != 0)
		return NULL;
	struct bc_anim_reader *r = calloc(1, sizeof *r);
	if (!r)
		return NULL;
	/* The file is stdin, so its name is never parsed by ffmpeg (a '%' would
	 * read as a sequence pattern). passthrough: each stored frame once, not
	 * resampled to a rate the format does not have. */
	snprintf(cmd, sizeof cmd,
	         "ffmpeg -v error -i - -fps_mode passthrough -an -f image2pipe "
	         "-c:v pam -pix_fmt rgba - < %s", q);
	r->dec = popen(cmd, "r");
	if (!r->dec) {
		free(r);
		return NULL;
	}
	if (read_pam_header(r->dec, &r->w, &r->h) != 0) {
		pclose(r->dec);
		free(r);
		return NULL;
	}
	r->have_first = 1;
	r->px = (size_t)r->w * r->h;
	read_durations(r, path);
	*w = r->w;
	*h = r->h;
	return r;
}

int bc_anim_expected(const struct bc_anim_reader *r)
{
	return r->n_dur;
}

int bc_anim_next(struct bc_anim_reader *r, unsigned char *rgba, double *dur_ms)
{
	if (!r->have_first) {
		int w, h;
		if (read_pam_header(r->dec, &w, &h) != 0)
			return 0;      /* no more frames */
		if (w != r->w || h != r->h)
			return -1;     /* ffmpeg composites to one canvas; this is not it */
	}
	r->have_first = 0;
	if (fread(rgba, 1, r->px * 4, r->dec) != r->px * 4)
		return -1;
	double d = r->idx < r->n_dur ? r->dur[r->idx] : 0.0;
	r->idx++;
	*dur_ms = d > 0.0 ? d : 100.0;
	return 1;
}

void bc_anim_close(struct bc_anim_reader *r)
{
	if (!r)
		return;
	if (r->dec)
		pclose(r->dec);
	free(r->dur);
	free(r);
}

/* ------------------------------------------------------------------ */
/* Writing                                                             */

struct bc_anim_writer {
	char   path[PATH_MAX];       /* the final name */
	char   dir[PATH_MAX];        /* staging directory */
	const struct bc_anim_fmt *fmt;
	int    w, h, n, any_alpha;
	double *dur;
	int    cap;
};

struct bc_anim_writer *bc_anim_begin(const char *path, const struct bc_anim_fmt *f,
                                     int w, int h)
{
	struct bc_anim_writer *wr = calloc(1, sizeof *wr);
	if (!wr)
		return NULL;
	snprintf(wr->path, sizeof wr->path, "%s", path);
	wr->fmt = f;
	wr->w = w;
	wr->h = h;
	/* Beside the output, hidden: a half-staged animation is not a result,
	 * and the same filesystem keeps the final rename atomic. */
	const char *slash = strrchr(path, '/');
	const int dlen = slash ? (int)(slash - path + 1) : 0;
	if (snprintf(wr->dir, sizeof wr->dir, "%.*s.bc-anim-XXXXXX", dlen, path)
	    >= (int)sizeof wr->dir || !mkdtemp(wr->dir)) {
		free(wr);
		return NULL;
	}
	return wr;
}

int bc_anim_frames(const struct bc_anim_writer *wr) { return wr->n; }

int bc_anim_add(struct bc_anim_writer *wr, const unsigned char *rgba,
                double dur_ms, int has_alpha)
{
	if (wr->n == wr->cap) {
		wr->cap = wr->cap ? wr->cap * 2 : 64;
		wr->dur = realloc(wr->dur, (size_t)wr->cap * sizeof *wr->dur);
		if (!wr->dur)
			return -1;
	}
	char name[PATH_MAX + 32];
	snprintf(name, sizeof name, "%s/f%06d.pam", wr->dir, wr->n);
	FILE *f = fopen(name, "wb");
	if (!f)
		return -1;
	fprintf(f, "P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\n"
	           "TUPLTYPE RGB_ALPHA\nENDHDR\n", wr->w, wr->h);
	const size_t sz = (size_t)wr->w * wr->h * 4;
	const int ok = fwrite(rgba, 1, sz, f) == sz;
	if (fclose(f) != 0 || !ok)
		return -1;
	wr->dur[wr->n++] = dur_ms;
	wr->any_alpha |= has_alpha;
	return 0;
}

static void remove_staging(struct bc_anim_writer *wr)
{
	char name[PATH_MAX + 32];
	for (int i = 0; i < wr->n; i++) {
		snprintf(name, sizeof name, "%s/f%06d.pam", wr->dir, i);
		unlink(name);
	}
	snprintf(name, sizeof name, "%s/list.ffconcat", wr->dir);
	unlink(name);
	rmdir(wr->dir);
}

void bc_anim_abort(struct bc_anim_writer *wr)
{
	if (!wr)
		return;
	remove_staging(wr);
	free(wr->dur);
	free(wr);
}

/*
 * The last frame's duration, set in the container.
 *
 * ffmpeg's encoders learn how long a frame lasts from where the next one
 * starts, so the last frame has no next and gets a guess: a copy of the one
 * before. That is wrong exactly where it matters -- a GIF that holds its last
 * frame for two seconds before looping is common and deliberate -- so it is
 * fixed here, after the fact, in the three containers' own terms. Every other
 * frame's timing comes through ffmpeg exactly.
 *
 * The new value is whatever makes the whole come to `total_ms`, rather than
 * the last frame's own duration, so it stays right if an encoder merged
 * identical neighbours (WebP's does) and the frame count is not what went in.
 */
static unsigned long crc_table[256];
static unsigned long crc32_of(const unsigned char *p, size_t n)
{
	if (!crc_table[1])
		for (unsigned long i = 0; i < 256; i++) {
			unsigned long c = i;
			for (int k = 0; k < 8; k++)
				c = c & 1 ? 0xedb88320ul ^ (c >> 1) : c >> 1;
			crc_table[i] = c;
		}
	unsigned long c = 0xfffffffful;
	for (size_t i = 0; i < n; i++)
		c = crc_table[(c ^ p[i]) & 0xff] ^ (c >> 8);
	return c ^ 0xfffffffful;
}

static unsigned long be32(const unsigned char *p)
{
	return ((unsigned long)p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3];
}

static int patch_gif(unsigned char *b, size_t n, double total_ms)
{
	if (n < 14 || memcmp(b, "GIF8", 4))
		return -1;
	size_t pos = 13;
	if (b[10] & 0x80)
		pos += 3 * ((size_t)1 << ((b[10] & 7) + 1));
	size_t last = 0;
	long sum = 0;
	int have = 0;
	long prev_cs = 0;
	while (pos < n && b[pos] != 0x3b) {
		if (b[pos] == 0x21 && pos + 1 < n) {
			if (b[pos + 1] == 0xf9 && pos + 7 < n && b[pos + 2] == 4) {
				if (have)
					sum += prev_cs;
				prev_cs = b[pos + 4] | (b[pos + 5] << 8);
				last = pos + 4;
				have = 1;
				pos += 8;
				continue;
			}
			pos += 2;
			while (pos < n && b[pos]) pos += b[pos] + 1;
			pos++;
		} else if (b[pos] == 0x2c && pos + 10 < n) {
			const unsigned char packed = b[pos + 9];
			pos += 10;
			if (packed & 0x80)
				pos += 3 * ((size_t)1 << ((packed & 7) + 1));
			pos++;
			while (pos < n && b[pos]) pos += b[pos] + 1;
			pos++;
		} else {
			return -1;
		}
	}
	if (!have)
		return -1;
	long cs = (long)(total_ms / 10.0 + 0.5) - sum;
	if (cs < 2)
		cs = 2;     /* browsers read 0 and 1 as "a default" of 10; 2 is the least that means itself */
	if (cs > 65535)
		cs = 65535;
	b[last] = (unsigned char)(cs & 0xff);
	b[last + 1] = (unsigned char)(cs >> 8);
	return 0;
}

static int patch_apng(unsigned char *b, size_t n, double total_ms)
{
	if (n < 8 || memcmp(b, "\x89PNG\r\n\x1a\n", 8))
		return -1;
	size_t pos = 8, last = 0;
	double sum = 0.0, prev = 0.0;
	int have = 0;
	while (pos + 12 <= n) {
		const unsigned long len = be32(b + pos);
		if (pos + 12 + len > n)
			return -1;
		if (!memcmp(b + pos + 4, "fcTL", 4) && len == 26) {
			if (have)
				sum += prev;
			const unsigned char *d = b + pos + 8;
			const unsigned num = (d[20] << 8) | d[21];
			const unsigned den = (d[22] << 8) | d[23];
			prev = 1000.0 * num / (den ? den : 100);
			last = pos;
			have = 1;
		}
		pos += 12 + len;
	}
	if (!have)
		return -1;
	double ms = total_ms - sum;
	if (ms < 1.0)
		ms = 1.0;
	unsigned num = (unsigned)(ms + 0.5), den = 1000;
	if (num > 65535) {          /* a long hold: coarser units */
		num = (unsigned)(ms / 10.0 + 0.5);
		den = 100;
		if (num > 65535)
			num = 65535;
	}
	unsigned char *d = b + last + 8;
	d[20] = (unsigned char)(num >> 8); d[21] = (unsigned char)num;
	d[22] = (unsigned char)(den >> 8); d[23] = (unsigned char)den;
	const unsigned long crc = crc32_of(b + last + 4, 4 + 26);   /* type + data */
	d[26] = (unsigned char)(crc >> 24); d[27] = (unsigned char)(crc >> 16);
	d[28] = (unsigned char)(crc >> 8);  d[29] = (unsigned char)crc;
	return 0;
}

static int patch_webp(unsigned char *b, size_t n, double total_ms)
{
	if (n < 12 || memcmp(b, "RIFF", 4) || memcmp(b + 8, "WEBP", 4))
		return -1;
	size_t pos = 12, last = 0;
	double sum = 0.0, prev = 0.0;
	int have = 0;
	while (pos + 8 <= n) {
		const unsigned long len = b[pos + 4] | (b[pos + 5] << 8) |
		                          (b[pos + 6] << 16) | ((unsigned long)b[pos + 7] << 24);
		if (!memcmp(b + pos, "ANMF", 4) && len >= 16 && pos + 8 + len <= n) {
			if (have)
				sum += prev;
			const unsigned char *d = b + pos + 8;
			prev = d[12] | (d[13] << 8) | (d[14] << 16);
			last = pos + 8 + 12;
			have = 1;
		}
		pos += 8 + len + (len & 1);
	}
	if (!have)
		return -1;
	double ms = total_ms - sum;
	if (ms < 0.0)
		ms = 0.0;
	if (ms > 16777215.0)
		ms = 16777215.0;
	const unsigned long v = (unsigned long)(ms + 0.5);
	b[last] = (unsigned char)v;
	b[last + 1] = (unsigned char)(v >> 8);
	b[last + 2] = (unsigned char)(v >> 16);
	return 0;
}

/* Patch the file at `path` in place. 0 on success. */
static int fix_last_duration(const char *path, const char *ext, double total_ms)
{
	FILE *f = fopen(path, "rb");
	if (!f)
		return -1;
	fseek(f, 0, SEEK_END);
	const long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	unsigned char *b = n > 0 ? malloc((size_t)n) : NULL;
	if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) {
		fclose(f);
		free(b);
		return -1;
	}
	fclose(f);
	int rc = -1;
	if (!strcasecmp(ext, "gif"))
		rc = patch_gif(b, (size_t)n, total_ms);
	else if (!strcasecmp(ext, "webp"))
		rc = patch_webp(b, (size_t)n, total_ms);
	else
		rc = patch_apng(b, (size_t)n, total_ms);
	if (rc == 0) {
		f = fopen(path, "wb");
		rc = f && fwrite(b, 1, (size_t)n, f) == (size_t)n ? 0 : -1;
		if (f && fclose(f) != 0)
			rc = -1;
	}
	free(b);
	return rc;
}

int bc_anim_finish(struct bc_anim_writer *wr)
{
	int rc = -1;
	if (wr->n < 1)
		goto out;

	/*
	 * The concat demuxer carries a duration per file. Its last entry has to
	 * be named twice for that last duration to count: ffmpeg only applies a
	 * duration when the next file arrives.
	 */
	char list[PATH_MAX + 32];
	snprintf(list, sizeof list, "%s/list.ffconcat", wr->dir);
	FILE *lf = fopen(list, "w");
	if (!lf)
		goto out;
	fprintf(lf, "ffconcat version 1.0\n");
	double total_ms = 0.0;
	for (int i = 0; i < wr->n; i++) {
		/*
		 * `framerate 1000` is not about speed. The image demuxer inside
		 * concat stamps its frames in a 1/25 s time base, which rounds every
		 * duration to a multiple of 40 ms: 70, 130 and 100 ms came out as 80,
		 * 120 and 120. A finer one keeps them.
		 */
		fprintf(lf, "file 'f%06d.pam'\noption framerate 1000\nduration %.6f\n",
		        i, wr->dur[i] / 1000.0);
		total_ms += wr->dur[i];
	}
	if (fclose(lf) != 0)
		goto out;

	/* Written under a hidden name and renamed once complete, as stills are. */
	char tmp[PATH_MAX];
	const char *slash = strrchr(wr->path, '/');
	const int dlen = slash ? (int)(slash - wr->path + 1) : 0;
	if (snprintf(tmp, sizeof tmp, "%.*s.%s.bc-part.%s", dlen, wr->path,
	             wr->path + dlen, wr->fmt->ext) >= (int)sizeof tmp)
		goto out;

	char qlist[PATH_MAX + 64], qtmp[PATH_MAX + 64], enc[512];
	if (bc_off_shq(qlist, sizeof qlist, list) != 0 ||
	    bc_off_shq(qtmp, sizeof qtmp, tmp) != 0)
		goto out;
	const char *ext = wr->fmt->ext;
	if (!strcasecmp(ext, "gif")) {
		/*
		 * One palette for the whole animation, chosen over every frame,
		 * with a slot kept for transparency only when something is
		 * transparent -- a reserved colour costs a little quality on every
		 * frame for nothing otherwise.
		 */
		snprintf(enc, sizeof enc,
		         "-vf \"split[a][b];[a]palettegen=stats_mode=full:"
		         "reserve_transparent=%d[p];[b][p]paletteuse=alpha_threshold=128\" "
		         "-loop 0",
		         wr->any_alpha ? 1 : 0);
	} else if (!strcasecmp(ext, "webp")) {
		snprintf(enc, sizeof enc, "-c:v libwebp_anim -lossless 0 -quality 92 -loop 0");
	} else {
		snprintf(enc, sizeof enc, "-c:v apng -plays 0");
	}
	/* vfr: keep each frame's own timing. Left to its default the encoder
	 * resamples to a constant rate, which is exactly what this exists to avoid. */
	char cmd[PATH_MAX * 2 + 1024];
	snprintf(cmd, sizeof cmd,
	         "ffmpeg -v error -y -f concat -safe 0 -i %s %s -fps_mode vfr "
	         "-f %s file:%s", qlist, enc, wr->fmt->muxer, qtmp);
	if (system(cmd) != 0) {
		unlink(tmp);
		goto out;
	}
	/* The last frame's duration, which the encoder can only guess. A failure
	 * here leaves ffmpeg's timing, which is right but for that frame. */
	if (fix_last_duration(tmp, wr->fmt->ext, total_ms) != 0)
		WARN("%s: could not set the last frame's duration; it keeps the "
		     "encoder's guess", wr->path);
	if (rename(tmp, wr->path) != 0) {
		ERR("could not move %s into place: %s", wr->path, strerror(errno));
		unlink(tmp);
		goto out;
	}
	rc = 0;
out:
	remove_staging(wr);
	free(wr->dur);
	free(wr);
	return rc;
}
