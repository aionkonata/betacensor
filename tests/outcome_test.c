/*
 * The detector's outcome contract (detect.h, enum bc_det_status).
 *
 * Every way a run can end has to come back to the caller as something it can
 * tell apart: "nothing finished", "finished, nothing there", and "finished,
 * nothing usable". The last two used to share a value, and both live and
 * offline censors read a broken detector as a clean screen because of it --
 * so what is pinned here is exactly that difference, at the two places it is
 * decided: the decoders, and the publish/collect hand-off.
 *
 * Builds the real detect.c, as fusion_test does; no model and no ONNX
 * Runtime session is needed, because neither path touches one.
 */
#include "../src/detect.c"

#include <stdio.h>

static int failures;

static void check(int cond, const char *what)
{
	printf("  %-58s %s\n", what, cond ? "ok" : "FAIL");
	if (!cond)
		failures++;
}

/* Enough of a detector for the hand-off: the lock and the result slots. */
static struct bc_detector *bare(void)
{
	struct bc_detector *d = calloc(1, sizeof *d);
	pthread_mutex_init(&d->lock, NULL);
	return d;
}

int main(void)
{
	struct bc_det out[BC_MAX_DET];
	struct bc_det_result r;

	printf("outcome: the detector's answers\n\n");

	printf("an output the decoder cannot read is a failure\n");
	{
		/* 2 classes, no keypoints, no masks: wants 6 channels. */
		static float raw[8 * 16];
		check(decode(raw, 8, 16, out, BC_MAX_DET, 0.5f, 2, 0, 0) < 0,
		      "too many channels (more classes than the table) is -1");
		check(decode(raw, 5, 16, out, BC_MAX_DET, 0.5f, 2, 0, 0) < 0,
		      "too few channels is -1");
		check(decode(raw, 6, 16, out, BC_MAX_DET, 0.5f, 2, 0, 0) == 0,
		      "the right shape with nothing above the floor is 0");
		check(decode_end2end(raw, 16, 7, out, BC_MAX_DET, 0.5f, 2) < 0,
		      "an end-to-end row that is not 6 wide is -1");
		check(decode_end2end(raw, 16, 6, out, BC_MAX_DET, 0.5f, 2) == 0,
		      "an end-to-end output of zero-score rows is 0");
	}

	printf("\nnothing finished\n");
	{
		struct bc_detector *d = bare();
		r = bc_detect_collect(d, out, BC_MAX_DET);
		check(r.status == BC_DET_PENDING, "a fresh detector is pending");
		check(bc_detect_collect(NULL, out, BC_MAX_DET).status ==
		      BC_DET_PENDING, "no detector at all is pending, not OK");
		free(d);
	}

	printf("\na clean screen and a broken detector are different answers\n");
	{
		struct bc_detector *d = bare();
		publish(d, out, 0, now_ms());
		r = bc_detect_collect(d, out, BC_MAX_DET);
		check(r.status == BC_DET_OK && r.n == 0,
		      "a run that found nothing is OK with n = 0");

		run_failed(d, now_ms());
		r = bc_detect_collect(d, out, BC_MAX_DET);
		check(r.status == BC_DET_FAILED, "a run that failed is FAILED");
		check(r.n == 0, "...and carries no detections");
		check(bc_detect_errors(d) == 1, "...and is counted");
		free(d);
	}

	printf("\neach outcome is handed out once\n");
	{
		struct bc_detector *d = bare();
		run_failed(d, now_ms());
		bc_detect_collect(d, out, BC_MAX_DET);
		r = bc_detect_collect(d, out, BC_MAX_DET);
		check(r.status == BC_DET_PENDING,
		      "a failure already collected is not reported again");

		const struct bc_det one = { .x = 1, .y = 2, .w = 3, .h = 4,
		                            .score = 0.9f, .cls = 1 };
		publish(d, &one, 1, now_ms());
		r = bc_detect_collect(d, out, BC_MAX_DET);
		check(r.status == BC_DET_OK && r.n == 1 && out[0].cls == 1,
		      "a success after a failure delivers its detections");
		r = bc_detect_collect(d, out, BC_MAX_DET);
		check(r.status == BC_DET_PENDING, "...once");
		free(d);
	}

	printf("\nthe latest outcome wins, whichever way round\n");
	{
		struct bc_detector *d = bare();
		const struct bc_det one = { .w = 10, .h = 10, .score = 0.9f };
		publish(d, &one, 1, now_ms());
		run_failed(d, now_ms());
		r = bc_detect_collect(d, out, BC_MAX_DET);
		check(r.status == BC_DET_FAILED,
		      "an uncollected success is replaced by a later failure");

		run_failed(d, now_ms());
		publish(d, &one, 1, now_ms());
		r = bc_detect_collect(d, out, BC_MAX_DET);
		check(r.status == BC_DET_OK && r.n == 1,
		      "an uncollected failure is replaced by a later success");
		free(d);
	}

	printf("\nthe caller's buffer bounds the copy\n");
	{
		struct bc_detector *d = bare();
		struct bc_det many[8] = { 0 };
		publish(d, many, 8, now_ms());
		struct bc_det small[3];
		r = bc_detect_collect(d, small, 3);
		check(r.status == BC_DET_OK && r.n == 3,
		      "eight results into a buffer of three gives three");
		free(d);
	}

	printf("\n%s\n", failures ? "FAILURES." : "All checks passed.");
	return failures ? 1 : 0;
}
