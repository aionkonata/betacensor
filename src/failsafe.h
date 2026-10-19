#ifndef BC_FAILSAFE_H
#define BC_FAILSAFE_H

/*
 * What the overlay may show this frame: the one decision in the live loop that
 * everything in §6 rests on, kept free of GL, Wayland and the clock so it can
 * be tested by itself (tests/failsafe_test.c).
 *
 * The loop gathers the facts, this ranks them, and the loop carries the
 * verdict out -- blanking, clearing boxes, counting, logging. Nothing here has
 * side effects, so the same facts always give the same answer.
 */

/*
 * §6 fail-closed deadlines. The pipeline has to keep *proving* it is looking at
 * the current screen; if it stops, the output is blanked rather than left
 * showing content nothing has inspected.
 *
 * Measured on this machine: KWin free-runs the screencast at ~146 frames/s even
 * with the desktop completely idle -- it does not send only on damage -- so a
 * frame is due every ~7 ms and 500 ms is ~70 missed frames. That is a wide
 * margin against a hiccup and still a fifth of a second against a real stall.
 *
 * The inference deadline is much longer because inference legitimately runs far
 * below the frame rate and a single slow run must not blank the screen; it is
 * there to catch "results have stopped entirely", not "results are late".
 */
#define STALL_MS      500.0
#define INFER_DEAD_MS 2000.0

struct bc_fs_facts {
	double now;
	double last_frame;    /* a captured frame last arrived */
	double last_infer;    /* the pipeline last proved it saw the screen */
	int    capture_dead;  /* the stream is gone, latched */
	int    panicked;      /* the user asked for a blank screen */
	int    paused;        /* the user asked for the real screen */
	int    enabled;       /* the config's master switch */
	int    reverse;       /* the config asks for reverse mode */
};

enum bc_fs_show {
	BC_SHOW_CENSOR = 0,   /* the ordinary path: draw the covers */
	BC_SHOW_PANIC,        /* opaque black over everything, by request */
	BC_SHOW_FAILSAFE,     /* opaque black over everything, by §6 */
	BC_SHOW_RAW,          /* no covers at all, by request */
};

struct bc_fs_verdict {
	enum bc_fs_show show;
	const char     *why;      /* BC_SHOW_FAILSAFE: which deadline, for the log */
	int             reverse;  /* reverse mode may be on this frame */
};

struct bc_fs_verdict bc_failsafe_decide(const struct bc_fs_facts *f);

#endif
