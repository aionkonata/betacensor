/*
 * BetaCensor-linux — Phase 1 capture spike.
 *
 * Proves the one thing the whole design rests on (DESIGN.md §0): that we can
 * get a screen frame from KWin into a GL texture without the pixels ever
 * touching the CPU. Negotiates portal consent, connects the PipeWire stream,
 * imports the DMA-BUF via EGLImage, and dumps a single frame to PPM as proof.
 *
 * The PPM dump is the *only* readback and exists purely to verify correctness.
 */

#include "portal.h"
#include "capture.h"
#include "gpu.h"
#include "util.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>

#define DEFAULT_OUT     "frame.ppm"
#define DEFAULT_RESTORE ".restore_token"
#define TIMEOUT_SECS    30

static void on_alarm(int sig)
{
	(void)sig;
	static const char msg[] =
		"[bc] ERROR: timed out waiting for a frame.\n"
		"[bc] If the consent dialog never appeared, check that "
		"xdg-desktop-portal-kde is running.\n";
	ssize_t n = write(2, msg, sizeof msg - 1);
	(void)n;
	_exit(2);
}

int main(int argc, char **argv)
{
	const char *out     = (argc > 1) ? argv[1] : DEFAULT_OUT;
	const char *restore = (argc > 2) ? argv[2] : DEFAULT_RESTORE;

	struct bc_portal portal;
	struct bc_gpu   *gpu = NULL;
	int rc;

	LOG("phase 1 capture spike");
	LOG("output: %s   restore-token: %s", out, restore);

	/*
	 * GL context first: we need the driver's modifier list before we can
	 * tell PipeWire which buffer layouts we are able to import.
	 */
	gpu = bc_gpu_create();
	if (!gpu) {
		ERR("GPU init failed — cannot continue");
		return 1;
	}

	rc = bc_portal_acquire(&portal, restore);
	if (rc < 0) {
		ERR("portal negotiation failed");
		bc_gpu_destroy(gpu);
		return 1;
	}

	signal(SIGALRM, on_alarm);
	alarm(TIMEOUT_SECS);

	rc = bc_capture_one(portal.pipewire_fd, portal.node_id, gpu, out);
	alarm(0);

	/* pw_context_connect_fd took the fd; don't double-close it. */
	portal.pipewire_fd = -1;
	bc_portal_release(&portal);
	bc_gpu_destroy(gpu);

	if (rc < 0) {
		ERR("capture failed");
		return 1;
	}

	LOG("");
	LOG("SUCCESS — zero-copy DMA-BUF import verified.");
	LOG("view with:  ffplay %s      (or: ffmpeg -i %s frame.png)", out, out);
	return 0;
}
