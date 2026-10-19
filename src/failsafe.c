#include "failsafe.h"

#include <stddef.h>

struct bc_fs_verdict bc_failsafe_decide(const struct bc_fs_facts *f)
{
	struct bc_fs_verdict v = { BC_SHOW_CENSOR, NULL, 0 };

	/*
	 * §6: has the pipeline proved recently that it is looking at the screen
	 * as it is now? Two deadlines, checked only while censoring is actually
	 * wanted -- see BC_SHOW_RAW below for why pause and the master switch
	 * suppress this rather than the other way round.
	 */
	const char *fail = NULL;
	if (!f->paused && f->enabled) {
		if (f->capture_dead)
			fail = "capture stream lost";
		else if (f->now - f->last_frame > STALL_MS)
			fail = "capture stalled";
		else if (f->now - f->last_infer > INFER_DEAD_MS)
			fail = "detector stopped producing results";
	}

	if (f->panicked) {
		/*
		 * Panic outranks everything -- the config's master switch, the
		 * pause, the detections themselves. It covers the whole output
		 * with an opaque black rectangle: fail-closed, and it must work
		 * when capture is dead, so it cannot be the pixelate effect
		 * (that samples a source texture which may be stale or absent,
		 * and stale content is exactly what must not be shown).
		 */
		v.show = BC_SHOW_PANIC;
	} else if (fail) {
		v.show = BC_SHOW_FAILSAFE;
		v.why  = fail;
	} else if (f->paused || !f->enabled) {
		/*
		 * Master switch and pause are applied by the caller after
		 * detection and tracking have run, so resuming shows the correct
		 * boxes on the very next frame instead of waiting for the model
		 * to rediscover what is on screen.
		 *
		 * These also suppress the fail-closed blank above. Both mean the
		 * user has explicitly asked to see the real screen, so a dead
		 * capture has nothing to fail at: blanking would be surprising
		 * rather than protective. Panic is the opposite -- it outranks
		 * them both.
		 */
		v.show = BC_SHOW_RAW;
	}

	/*
	 * Reverse mode inverts the program's purpose, so every path that exists
	 * to cover the screen must switch it off -- otherwise the single
	 * full-screen blank would be read as one enormous cutout and reveal the
	 * entire display at exactly the moment something has gone wrong. Panic,
	 * the fail-closed deadlines, the pause and the master switch all clear
	 * it; only the ordinary censoring path may turn it on.
	 */
	v.reverse = f->reverse && v.show == BC_SHOW_CENSOR;
	return v;
}
