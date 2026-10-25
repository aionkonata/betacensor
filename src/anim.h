#ifndef BC_ANIM_H
#define BC_ANIM_H

/*
 * Animated GIF, WebP and PNG (APNG): reading their frames with their own
 * timing, and writing a result with the timing kept.
 *
 * Both ends go through ffmpeg, as everything else here does, but not the way
 * bc-video does. A video has a frame rate; an animated image has a *duration
 * per frame*, and the two are not the same thing. A GIF that holds one picture
 * for 130 ms and the next for 40 ms is not a 25 fps video: decoded at a fixed
 * rate it is resampled, and its pacing -- often the whole joke -- is lost. So
 * each frame is decoded once, as the format stores it, with its duration read
 * from the container, and written back the same way.
 */

#include <stdio.h>

/* Formats that can carry an animation, and what they need. */
struct bc_anim_fmt {
	const char *ext;      /* gif, webp, png */
	const char *muxer;    /* ffmpeg -f */
	int         alpha;    /* can the output carry transparency */
};
const struct bc_anim_fmt *bc_anim_fmt_for(const char *ext);

/*
 * Is `path` an animation -- more than one frame? Cheap for the formats that
 * cannot be (anything but gif, webp and png answers 0 without opening it); a
 * PNG is checked for an acTL chunk, the animation control chunk that makes it
 * an APNG, without running ffmpeg.
 */
int bc_anim_is_animated(const char *path);

struct bc_anim_reader;

/*
 * Start decoding. Reads the first frame's header, so w/h are valid on return;
 * NULL if ffmpeg cannot read it. Frames come back RGBA, composited by ffmpeg
 * into the full canvas (disposal and blending already applied), so each is a
 * complete picture and not a patch.
 */
struct bc_anim_reader *bc_anim_open(const char *path, int *w, int *h);

/* How many frames the container says there are (0 if it would not say). */
int  bc_anim_expected(const struct bc_anim_reader *r);

/* Next frame into `rgba` (w*h*4). 1 got one, 0 end of animation, -1 broken.
 * *dur_ms is how long this frame is shown. */
int  bc_anim_next(struct bc_anim_reader *r, unsigned char *rgba, double *dur_ms);
void bc_anim_close(struct bc_anim_reader *r);

struct bc_anim_writer;

/*
 * Begin an animation at `path`. Frames are staged on disk next to it and the
 * encoder is run once at the end, because the encoder has to see the whole
 * thing: a GIF's palette is chosen over every frame, and each frame carries
 * its own duration, which a pipe of raw frames cannot say.
 */
struct bc_anim_writer *bc_anim_begin(const char *path, const struct bc_anim_fmt *f,
                                     int w, int h);
/* One frame, `rgba` (w*h*4) shown for dur_ms. `has_alpha`: any pixel < 255. */
int  bc_anim_add(struct bc_anim_writer *wr, const unsigned char *rgba,
                 double dur_ms, int has_alpha);
/* Encode and move into place. 0 on success; the staged frames are removed
 * either way. */
int  bc_anim_finish(struct bc_anim_writer *wr);
/* Give up: nothing is written, the staged frames are removed. */
void bc_anim_abort(struct bc_anim_writer *wr);
int  bc_anim_frames(const struct bc_anim_writer *wr);

#endif /* BC_ANIM_H */
