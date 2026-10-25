#ifndef BC_STREAM_H
#define BC_STREAM_H

/*
 * One frame in, one censored frame out, with memory between them.
 *
 * Everything bc-video does to a frame, lifted out of its main loop so that
 * anything else that has a *sequence* of frames does the same: detect,
 * track, build the covers and the eye bars, draw, harden, badge. bc-video
 * feeds it a decoder pipe; bc-image feeds it the frames of an animated GIF or
 * WebP. The memory is what makes it a stream and not a pile of images -- the
 * tracker carries a box through a frame the detector missed, and each eye bar
 * is steadied against the last -- and it is why an animation is run through
 * here rather than censored frame by frame as stills.
 *
 * The caller owns the detector, the GL context, the preprocessor and the
 * tensor; this owns the tracker and the eye bars' history.
 */

#include "config.h"
#include "detect.h"
#include "offline.h"
#include "preprocess.h"
#include "track.h"

#include <stdint.h>

#define BC_STREAM_MAX_BOX 64

struct bc_stream_vbar {
	double born;                  /* the track this belongs to */
	int    valid, missed;
	float  dcx, dcy, fw, fh;      /* bar centre offset and size, as fractions of the face */
	float  angle;
};

struct bc_stream {
	/* Borrowed. */
	const struct bc_config *cfg;
	const int              *cover;     /* which classes are covered (BC_MAX_CLASSES) */
	struct bc_detector     *det;
	struct bc_off_gl       *g;
	struct bc_preproc      *pp;
	float                  *tensor;
	const struct bc_off_badge *bdg;    /* NULL for no badge */
	int                     harden;
	uint32_t                harden_seed;
	int                     test_box[4];  /* x,y,w,h pinned in every frame; w <= 0: none */
	/*
	 * Optional: last word on the covers before they are drawn, for edits the
	 * user made by hand. Returns the new count.
	 */
	int (*adjust)(void *user, struct bc_box *boxes, int nb, int max);
	void *adjust_user;

	/* Owned. */
	struct bc_tracker       tr;
	struct bc_off_eb        eb;
	struct bc_stream_vbar   vbar[BC_TRK_MAX];
	/*
	 * The last frame and what the detector said about it. A frame identical
	 * to its predecessor gets the same answer without asking again -- exact,
	 * not approximate, since the same pixels cannot score differently -- and
	 * it matters because an animation holds frames: a GIF delayed 100 ms is
	 * often one picture shown for several of the format's ticks.
	 */
	unsigned char          *prev;
	size_t                  prev_cap, prev_len;
	struct bc_det           prev_dets[BC_MAX_DET];
	int                     prev_n, prev_ok;

	/* What happened, for the summary. */
	long          frames, censored_frames, blanked_frames;
	unsigned long reused;
	double        covered_px;
};

enum {
	BC_STREAM_OK      = 0,
	BC_STREAM_BLANKED = 1,    /* no detector result: written black, never clean */
	BC_STREAM_HUNG    = -1,   /* the detector stopped answering: stop everything */
};

/* Reset the tracker and history; the borrowed fields are the caller's to fill
 * before the first frame. */
void bc_stream_init(struct bc_stream *s);
void bc_stream_free(struct bc_stream *s);

/*
 * Censor one frame: `rgb` in (top-down RGB24, the size the GL context was
 * made for), `out` the same size. `t_ms` is the frame's time on the stream's
 * clock -- from the frame index or the source's own timestamps, never the wall
 * clock, so a run is the same however loaded the machine is.
 *
 * `boxes`/`nb` (both optional) receive the covers drawn, for callers that need
 * to know which pixels are cover.
 */
int bc_stream_frame(struct bc_stream *s, const unsigned char *rgb, double t_ms,
                    unsigned char *out, struct bc_box *boxes, int *nb);

#endif /* BC_STREAM_H */
