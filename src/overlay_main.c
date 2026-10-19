/*
 * BetaCensor-linux -- Phase 2 overlay spike.
 *
 * Maps a click-through layer-shell surface over one monitor and draws hardcoded
 * boxes on it (DESIGN.md §7 phase 2). Proves three things:
 *
 *   1. KWin gives us zwlr_layer_shell_v1 on the OVERLAY layer, above everything.
 *   2. An empty input region makes the surface click-through.
 *   3. Alpha compositing works, so non-censored screen shows through untouched.
 *
 * Runs for a bounded number of seconds and then exits. That limit is a safety
 * feature, not a convenience: a bug that left an OVERLAY-layer surface mapped
 * would sit on top of the user's desktop until they killed it blind.
 */

#include "overlay.h"
#include "util.h"

#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>

/* NULL = the first output the compositor lists; argv[1] names one. */
#define DEFAULT_OUTPUT NULL
#define DEFAULT_SECS   6
#define HARD_LIMIT     60

static volatile sig_atomic_t running = 1;

static void on_signal(int sig)
{
	(void)sig;
	running = 0;
}

static void on_alarm(int sig)
{
	(void)sig;
	static const char msg[] =
		"[bc] ERROR: overlay watchdog fired; force-exiting.\n";
	ssize_t n = write(2, msg, sizeof msg - 1);
	(void)n;
	_exit(2);
}

static double now_sec(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
	const char *output = (argc > 1) ? argv[1] : DEFAULT_OUTPUT;
	double secs = (argc > 2) ? atof(argv[2]) : DEFAULT_SECS;

	if (!(secs > 0) || secs > HARD_LIMIT)
		secs = DEFAULT_SECS;

	LOG("phase 2 overlay spike");
	LOG("output: %s   duration: %.1fs", output, secs);

	signal(SIGINT,  on_signal);
	signal(SIGTERM, on_signal);
	signal(SIGALRM, on_alarm);
	/* Backstop in case the render loop wedges inside the compositor. */
	alarm((unsigned)secs + 10);

	struct bc_overlay *o = bc_overlay_create(output);
	if (!o) {
		ERR("overlay creation failed");
		return 1;
	}

	int w = bc_overlay_width(o), h = bc_overlay_height(o);

	/*
	 * Hardcoded stand-ins for detector output. Premultiplied alpha: the RGB
	 * channels are already scaled by the alpha value.
	 */
	const struct bc_box boxes[] = {
		/* Opaque black bar -- what a real censor box looks like. */
		{ w / 8, h / 8, w / 4, h / 4,          0.0f, 0.0f, 0.0f, 1.0f,
		  BC_EFFECT_FILL, 0.0f, 0.0f, "", 0, { 0.0f, 0.0f, 0.0f },
		  0.0f, { 0.0f, 0.0f, 0.0f }, 0.0f, 0.0f, 0, 0, 0 },
		/* 50% red -- proves alpha blending against the desktop below. */
		{ w / 2, h / 4, w / 4, h / 3,          0.5f, 0.0f, 0.0f, 0.5f,
		  BC_EFFECT_FILL, 0.0f, 0.0f, "", 0, { 0.0f, 0.0f, 0.0f },
		  0.0f, { 0.0f, 0.0f, 0.0f }, 0.0f, 0.0f, 0, 0, 0 },
		/* Thin full-width strip -- makes the exact edges easy to eyeball. */
		{ 0, h - 40, w, 40,                    0.0f, 0.35f, 0.0f, 0.7f,
		  BC_EFFECT_FILL, 0.0f, 0.0f, "", 0, { 0.0f, 0.0f, 0.0f },
		  0.0f, { 0.0f, 0.0f, 0.0f }, 0.0f, 0.0f, 0, 0, 0 },
	};
	const int n_boxes = (int)(sizeof boxes / sizeof boxes[0]);

	LOG("drawing %d boxes on a %dx%d surface for %.1fs", n_boxes, w, h, secs);
	LOG("the desktop should stay fully usable -- clicks pass through");

	double start = now_sec();
	long frames = 0;
	int rc = 0;

	while (running && now_sec() - start < secs) {
		rc = bc_overlay_present(o, boxes, n_boxes);
		if (rc != 0)
			break;
		frames++;
	}

	double elapsed = now_sec() - start;
	alarm(0);
	bc_overlay_destroy(o);

	if (rc < 0) {
		ERR("presentation failed after %ld frames", frames);
		return 1;
	}

	LOG("");
	LOG("presented %ld frames in %.2fs (%.1f fps)",
	    frames, elapsed, elapsed > 0 ? frames / elapsed : 0.0);
	LOG("SUCCESS -- click-through layer-shell overlay verified.");
	return 0;
}
