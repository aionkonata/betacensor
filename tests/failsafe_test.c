/*
 * What the overlay may show (failsafe.h).
 *
 * The rule set is small and every part of it has been load-bearing: panic
 * outranks everything, the §6 deadlines blank only while censoring is wanted,
 * pause and the master switch show the real screen, and reverse mode -- which
 * would turn a full-screen blank into a full-screen reveal -- is only ever on
 * along the ordinary path. Until this was its own function it lived in the
 * middle of the live loop, where none of it could be driven without a
 * compositor, a GPU and a detector.
 */
#include "../src/failsafe.c"

#include <stdio.h>

static int failures;

static void check(int cond, const char *what)
{
	printf("  %-58s %s\n", what, cond ? "ok" : "FAIL");
	if (!cond)
		failures++;
}

/* A healthy pipeline at t = 10 s: a frame and a result both just in. */
static struct bc_fs_facts healthy(void)
{
	return (struct bc_fs_facts){
		.now = 10000.0, .last_frame = 9995.0, .last_infer = 9990.0,
		.enabled = 1,
	};
}

static enum bc_fs_show show(struct bc_fs_facts f)
{
	return bc_failsafe_decide(&f).show;
}

int main(void)
{
	struct bc_fs_facts f;

	printf("failsafe: what the overlay may show\n\n");

	printf("a healthy pipeline censors\n");
	check(show(healthy()) == BC_SHOW_CENSOR, "fresh frame and result: covers");

	printf("\nthe deadlines\n");
	f = healthy();
	f.last_frame = f.now - STALL_MS;
	check(show(f) == BC_SHOW_CENSOR, "a frame exactly at the stall deadline is fine");
	f.last_frame = f.now - STALL_MS - 1.0;
	check(show(f) == BC_SHOW_FAILSAFE, "a frame past it blanks");
	check(bc_failsafe_decide(&f).why != NULL, "...and says why");

	f = healthy();
	f.last_infer = f.now - INFER_DEAD_MS;
	check(show(f) == BC_SHOW_CENSOR, "a result exactly at the inference deadline is fine");
	f.last_infer = f.now - INFER_DEAD_MS - 1.0;
	check(show(f) == BC_SHOW_FAILSAFE, "a result past it blanks");

	f = healthy();
	f.capture_dead = 1;
	check(show(f) == BC_SHOW_FAILSAFE, "a lost stream blanks, however fresh the stamps");

	printf("\npanic outranks everything\n");
	f = healthy();
	f.panicked = 1;
	check(show(f) == BC_SHOW_PANIC, "over a healthy pipeline");
	f.capture_dead = 1;
	check(show(f) == BC_SHOW_PANIC, "over a dead one");
	f.paused = 1;
	check(show(f) == BC_SHOW_PANIC, "over pause");
	f.enabled = 0;
	check(show(f) == BC_SHOW_PANIC, "over the master switch");

	printf("\nthe user asking for the real screen\n");
	f = healthy();
	f.paused = 1;
	check(show(f) == BC_SHOW_RAW, "pause shows the screen");
	f.capture_dead = 1;
	f.last_infer = 0.0;
	check(show(f) == BC_SHOW_RAW, "...and is not overruled by a dead pipeline");
	f = healthy();
	f.enabled = 0;
	f.last_frame = 0.0;
	check(show(f) == BC_SHOW_RAW, "the master switch likewise");

	printf("\nreverse mode is only ever on the ordinary path\n");
	f = healthy();
	f.reverse = 1;
	check(bc_failsafe_decide(&f).reverse, "healthy and asked for: on");
	f.panicked = 1;
	check(!bc_failsafe_decide(&f).reverse, "panic: off");
	f = healthy();
	f.reverse = 1;
	f.capture_dead = 1;
	check(!bc_failsafe_decide(&f).reverse, "fail-closed: off");
	f = healthy();
	f.reverse = 1;
	f.paused = 1;
	check(!bc_failsafe_decide(&f).reverse, "paused: off");
	f = healthy();
	f.reverse = 1;
	f.enabled = 0;
	check(!bc_failsafe_decide(&f).reverse, "master switch off: off");
	f = healthy();
	check(!bc_failsafe_decide(&f).reverse, "not asked for: off");

	printf("\n%s\n", failures ? "FAILURES." : "All checks passed.");
	return failures ? 1 : 0;
}
