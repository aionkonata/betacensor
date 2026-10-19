/*
 * betacensor -- the one command to run this thing.
 *
 * Replaces run.sh, and exists mainly so there is a single entry point rather
 * than a growing family of shell scripts that each know a little about how the
 * parts fit together. The parts are: a core (build/bc), a settings panel
 * (build/betacensor-gui), an offline video pass (build/bc-video) and its own
 * panel (build/bc-video-gui). Which of those to start, in what order, and what
 * to do when one of them exits is the knowledge this file holds.
 *
 * Subcommands:
 *
 *   betacensor                  live censor + settings panel (the default)
 *   betacensor live [out] [s]   same, naming an output and a duration
 *   betacensor gui              settings panel only, against a running censor
 *   betacensor video [in] [out] batch a file; with no arguments, its panel
 *   betacensor image [args]     batch images; without -o, its panel
 *   betacensor panic            blank the screen (SIGUSR1); again to undo
 *   betacensor pause            show the real screen (SIGUSR2); again to undo
 *   betacensor stop             stop a running censor
 *   betacensor status           is one running, and on what
 *
 * The signalling subcommands exist because the alternative was telling the user
 * to read a pid out of a log line and type kill -USR1 at it. A pid file removes
 * the copying step without ever introducing a search-by-name kill: the pid is
 * read from the file and then *verified* to still be our own binary before any
 * signal is sent (see live_pid). A recycled pid belonging to something else is
 * the failure this guards against, and it is not hypothetical -- pids wrap.
 */

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define PID_FILE  "betacensor.pid"
#define CONF_FILE "betacensor.conf"

static pid_t core_pid = 0, gui_pid = 0;

/*
 * The directory the launcher was started from, before chdir_to_project()
 * moved it. Paths the user typed are relative to *this*, and the batch tools
 * run from the project directory, so they are made absolute against it first.
 * Without that, `betacensor video clip.mp4 out.mp4` run from ~/Videos looked
 * for clip.mp4 in the project directory.
 */
static char caller_cwd[4096];

/* `p` made absolute against the caller's directory. Returns a static buffer
 * from a small ring, so a handful can be live at once in one argv. */
static char *abs_arg(const char *p)
{
	static char ring[8][4096];
	static int  next;
	if (!p || p[0] == '/' || !caller_cwd[0])
		return (char *)p;
	char *b = ring[next++ % 8];
	snprintf(b, sizeof ring[0], "%s/%s", caller_cwd, p);
	return b;
}

static void say(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fputc('\n', stderr);
}

/* ------------------------------------------------------------------ */

/*
 * Run from the project directory whatever the working directory was, so a
 * desktop shortcut behaves like a shell launch. The binaries live in build/ and
 * the config sits beside it, one level up.
 */
static void chdir_to_project(void)
{
	char exe[4096];
	ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
	if (n <= 0)
		return;
	exe[n] = 0;

	char *slash = strrchr(exe, '/');
	if (!slash)
		return;
	*slash = 0;                       /* .../build */

	slash = strrchr(exe, '/');
	if (slash && !strcmp(slash, "/build"))
		*slash = 0;               /* project root */

	if (chdir(exe) != 0)
		say("warning: could not enter %s (%s)", exe, strerror(errno));
}

static int have(const char *path)
{
	return access(path, X_OK) == 0;
}

/*
 * Build what is missing. Silent when there is nothing to do, which is the
 * common case; noisy only when it actually has to compile something, because a
 * launcher that prints a wall of build output every time trains you to ignore
 * it.
 */
static int ensure(const char *const *bins, const char *target)
{
	int missing = 0;
	for (int i = 0; bins[i]; i++)
		if (!have(bins[i]))
			missing = 1;
	if (!missing)
		return 0;

	say("Building (%s)...", target);
	char cmd[128];
	snprintf(cmd, sizeof cmd, "make %s >/dev/null", target);
	if (system(cmd) != 0) {
		say("build failed -- run 'make %s' to see why", target);
		return -1;
	}
	for (int i = 0; bins[i]; i++)
		if (!have(bins[i])) {
			say("build succeeded but %s is still missing", bins[i]);
			return -1;
		}
	return 0;
}

/* ------------------------------------------------------------------ */

/*
 * The pid file is written and removed by the core itself (bc_main.c), not
 * here. Two writers for one file is a race with nothing to gain: the core
 * knows when it is up and when it is going down, and this process may not even
 * be running -- 'betacensor panic' from a second terminal has to work against
 * a censor this launcher never started.
 */

/*
 * The pid of a running censor, or 0.
 *
 * Verified twice over: the process must exist, and /proc must agree that it is
 * our own binary. Signalling a stale pid file whose number has since been
 * handed to something else is exactly the accident that kill-by-pattern causes
 * from the other direction, and a launcher is the wrong place to be careless
 * about which process gets a signal.
 */
static pid_t live_pid(void)
{
	FILE *f = fopen(PID_FILE, "r");
	if (!f)
		return 0;
	int p = 0;
	int got = fscanf(f, "%d", &p);
	fclose(f);
	if (got != 1 || p <= 1)
		return 0;
	if (kill((pid_t)p, 0) != 0)
		return 0;

	char path[64], comm[64] = { 0 };
	snprintf(path, sizeof path, "/proc/%d/comm", p);
	FILE *c = fopen(path, "r");
	if (!c)
		return 0;
	if (!fgets(comm, sizeof comm, c)) {
		fclose(c);
		return 0;
	}
	fclose(c);
	comm[strcspn(comm, "\n")] = 0;
	return strcmp(comm, "bc") == 0 ? (pid_t)p : 0;
}

static int signal_running(int sig, const char *what)
{
	pid_t p = live_pid();
	if (!p) {
		say("no censor is running");
		return 1;
	}
	if (kill(p, sig) != 0) {
		say("could not signal pid %d: %s", (int)p, strerror(errno));
		return 1;
	}
	say("%s (pid %d)", what, (int)p);
	return 0;
}

/* ------------------------------------------------------------------ */

static void stop_children(void)
{
	/* By pid, never by name. */
	if (gui_pid > 0)
		kill(gui_pid, SIGTERM);
	if (core_pid > 0)
		kill(core_pid, SIGTERM);

	/*
	 * Give them a moment to close down before insisting. The core has a
	 * teardown path that prints its counters; killing it outright throws
	 * away the one summary that says whether the run was clean.
	 *
	 * Progress is judged by reaping, not by asking kill(pid, 0) whether
	 * they are still there. An exited child remains visible to kill() as a
	 * zombie until it is waited for, so polling that way never observed
	 * either one leaving: every shutdown sat out the full timeout and then
	 * SIGKILLed processes that had already finished.
	 */
	for (int i = 0; i < 50 && (gui_pid > 0 || core_pid > 0); i++) {
		pid_t done;
		while ((done = waitpid(-1, NULL, WNOHANG)) > 0) {
			if (done == gui_pid)
				gui_pid = 0;
			if (done == core_pid)
				core_pid = 0;
		}
		if (gui_pid <= 0 && core_pid <= 0)
			break;
		struct timespec ts = { 0, 100000000 };
		nanosleep(&ts, NULL);
	}
	if (gui_pid > 0)
		kill(gui_pid, SIGKILL);
	if (core_pid > 0)
		kill(core_pid, SIGKILL);

	while (waitpid(-1, NULL, WNOHANG) > 0)
		;
}

static void on_signal(int sig)
{
	(void)sig;
	stop_children();
	_exit(0);
}

static pid_t spawn(const char *path, char *const argv[])
{
	pid_t p = fork();
	if (p == 0) {
		execv(path, argv);
		_exit(127);
	}
	return p;
}

/*
 * Run one of our own scripts to completion and hand back its status.
 *
 * Through the shell rather than execv, so it works from a source tree
 * unpacked without the executable bit -- which is exactly what a tarball
 * extracted with the wrong umask, or copied off a FAT stick, looks like.
 */
static int run_script(const char *rel)
{
	if (!have(rel)) {
		say("%s is missing -- is this a complete source tree?", rel);
		return 1;
	}

	char *const argv[] = { (char *)"sh", (char *)rel, NULL };
	pid_t p = spawn("/bin/sh", argv);
	if (p < 0) {
		say("could not run %s", rel);
		return 1;
	}

	int st = 0;
	if (waitpid(p, &st, 0) < 0)
		return 1;
	return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}

/* ------------------------------------------------------------------ */

static int cmd_live(int argc, char **argv)
{
	/*
	 * "auto" rather than a connector name: the launcher has no idea what
	 * this machine calls its monitors, and the core reads "auto" as "the
	 * first one listed". The position still has to be filled, because the
	 * duration is the argument after it.
	 */
	const char *output = argc > 0 ? argv[0] : "auto";
	const char *secs   = argc > 1 ? argv[1] : "86400";
	/*
	 * The model argument is passed as an empty string on purpose. The core
	 * reads empty as "not given" and takes the model from the config, so
	 * the GUI's model picker survives a restart; naming one here would
	 * override that pick on every launch. BC_MODEL still forces one.
	 */
	const char *model = getenv("BC_MODEL");
	if (!model)
		model = "";

	pid_t already = live_pid();
	if (already) {
		say("a censor is already running (pid %d)", (int)already);
		say("stop it with 'betacensor stop', or open its panel with "
		    "'betacensor gui'");
		return 1;
	}

	static const char *const bins[] = { "build/bc", NULL };
	static const char *const guis[] = { "build/betacensor-gui", NULL };
	if (ensure(bins, "bc") != 0 || ensure(guis, "gui") != 0)
		return 1;

	signal(SIGINT,  on_signal);
	signal(SIGTERM, on_signal);
	signal(SIGHUP,  on_signal);

	char *core_argv[] = { (char *)"bc", (char *)output, (char *)model,
	                      (char *)secs, NULL };
	core_pid = spawn("build/bc", core_argv);
	if (core_pid < 0) {
		say("could not start the censor: %s", strerror(errno));
		return 1;
	}

	/*
	 * The core writes the config on first run and the panel needs it to
	 * know the class list, so wait for the file rather than racing it.
	 */
	for (int i = 0; i < 50; i++) {
		if (access(CONF_FILE, R_OK) == 0)
			break;
		if (waitpid(core_pid, NULL, WNOHANG) == core_pid) {
			say("the censor exited during startup -- see the log above");
					return 1;
		}
		struct timespec ts = { 0, 100000000 };
		nanosleep(&ts, NULL);
	}

	char *gui_argv[] = { (char *)"betacensor-gui", (char *)CONF_FILE, NULL };
	gui_pid = spawn("build/betacensor-gui", gui_argv);

	printf("\n"
	       "  Censoring %s.  Control panel is open.\n"
	       "\n"
	       "    betacensor panic    blank the screen     (again to undo)\n"
	       "    betacensor pause    show the real screen (again to undo)\n"
	       "\n"
	       "  Closing the panel, or Ctrl-C here, stops both.\n\n",
	       output);
	fflush(stdout);

	/*
	 * Exit as soon as *either* child does, rather than waiting for both.
	 * If the core died there is nothing left to control; if the panel was
	 * closed, that was the user saying stop.
	 */
	for (;;) {
		int st;
		pid_t done = waitpid(-1, &st, 0);
		if (done < 0) {
			if (errno == EINTR)
				continue;
			break;
		}
		if (done == core_pid) {
			core_pid = 0;
			break;
		}
		if (done == gui_pid) {
			gui_pid = 0;
			break;
		}
	}
	stop_children();
	return 0;
}

static int cmd_gui(void)
{
	static const char *const guis[] = { "build/betacensor-gui", NULL };
	if (ensure(guis, "gui") != 0)
		return 1;
	if (!live_pid())
		say("note: no censor is running -- settings will be saved and "
		    "picked up when one starts");
	char *argv[] = { (char *)"betacensor-gui", (char *)CONF_FILE, NULL };
	execv("build/betacensor-gui", argv);
	say("could not start the panel: %s", strerror(errno));
	return 1;
}

static int cmd_video(int argc, char **argv)
{
	static const char *const vid[] = { "build/bc-video", NULL };
	static const char *const vgui[] = { "build/bc-video-gui", NULL };

	/* --no-harden / --no-badge pass through as they are; the rest are
	 * paths, made absolute. */
	int flags = 0;
	while (flags < argc && !strncmp(argv[flags], "--", 2)) {
		const int end = !strcmp(argv[flags], "--");
		flags++;
		if (end)
			break;
	}
	if (argc - flags >= 2) {
		if (ensure(vid, "video") != 0)
			return 1;
		char *a[16] = { (char *)"bc-video" };
		int n = 1;
		for (int i = 0; i < flags && n < 8; i++)
			a[n++] = argv[i];
		a[n++] = strdup(abs_arg(argv[flags]));
		a[n++] = strdup(abs_arg(argv[flags + 1]));
		if (argc - flags > 2)
			a[n++] = strdup(abs_arg(argv[flags + 2]));
		a[n] = NULL;
		execv("build/bc-video", a);
	} else {
		if (ensure(vgui, "gui") != 0 || ensure(vid, "video") != 0)
			return 1;
		char *a[] = { (char *)"bc-video-gui",
		              argc == 1 ? abs_arg(argv[0]) : NULL, NULL };
		execv("build/bc-video-gui", a);
	}
	say("could not start the video pass: %s", strerror(errno));
	return 1;
}

/*
 * `betacensor image` with an output named (-o/--out) is a batch run handed
 * straight to bc-image; anything else opens the panel, with whatever was
 * named already in its list. Every path argument is made absolute first --
 * options' values for -o and -m, and the inputs -- while -c and -f values are
 * passed as they are.
 */
static int cmd_image(int argc, char **argv)
{
	static const char *const img[]  = { "build/bc-image", NULL };
	static const char *const igui[] = { "build/bc-image-gui", NULL };

	int batch = 0;
	for (int i = 0; i < argc; i++)
		if (!strcmp(argv[i], "--"))
			break;
		else if (!strcmp(argv[i], "-o") || !strcmp(argv[i], "--out") ||
		         !strncmp(argv[i], "--out=", 6) ||
		         !strcmp(argv[i], "--list-classes") ||
		         !strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
			batch = 1;

	char **a = calloc((size_t)argc + 2, sizeof *a);
	if (!a)
		return 1;
	int n = 0, opts_done = 0;
	a[n++] = (char *)(batch ? "bc-image" : "bc-image-gui");
	for (int i = 0; i < argc; i++) {
		const char *s = argv[i];
		char *v = argv[i];
		if (!batch || opts_done) {
			v = abs_arg(s);
		} else if (!strcmp(s, "--")) {
			opts_done = 1;
		} else if (!strncmp(s, "--out=", 6) ||
		           !strncmp(s, "--model=", 8)) {
			const char *eq = strchr(s, '=') + 1;
			char buf[4200];
			snprintf(buf, sizeof buf, "%.*s%s", (int)(eq - s), s,
			         abs_arg(eq));
			a[n++] = strdup(buf);
			continue;
		} else if (s[0] == '-') {
			/* An option. The path-valued ones take the next arg. */
			if ((!strcmp(s, "-o") || !strcmp(s, "--out") ||
			     !strcmp(s, "-m") || !strcmp(s, "--model")) &&
			    i + 1 < argc) {
				a[n++] = v;
				v = abs_arg(argv[++i]);
			} else if (!strcmp(s, "-c") || !strcmp(s, "--classes") ||
			           !strcmp(s, "-f") || !strcmp(s, "--format")) {
				if (i + 1 < argc) {
					a[n++] = v;
					v = argv[++i];
				}
			}
		} else {
			v = abs_arg(s);
		}
		/* abs_arg's ring is small; keep a copy of each. */
		a[n++] = strdup(v);
	}
	a[n] = NULL;

	if (batch) {
		if (ensure(img, "image") != 0)
			return 1;
		execv("build/bc-image", a);
	} else {
		if (ensure(igui, "gui") != 0 || ensure(img, "image") != 0)
			return 1;
		execv("build/bc-image-gui", a);
	}
	say("could not start the image pass: %s", strerror(errno));
	return 1;
}

static int cmd_status(void)
{
	pid_t p = live_pid();
	if (!p) {
		printf("not running\n");
		return 1;
	}
	printf("running (pid %d)\n", (int)p);

	/* Echo the few settings that decide what it is actually doing. */
	FILE *f = fopen(CONF_FILE, "r");
	if (!f)
		return 0;
	char line[512];
	while (fgets(line, sizeof line, f)) {
		if (!strncmp(line, "enabled", 7) || !strncmp(line, "effect", 6) ||
		    !strncmp(line, "model", 5) || !strncmp(line, "reverse", 7))
			printf("  %s", line);
	}
	fclose(f);
	return 0;
}

static int cmd_stop(void)
{
	pid_t p = live_pid();
	if (!p) {
		say("no censor is running");
		return 1;
	}
	kill(p, SIGTERM);
	for (int i = 0; i < 50; i++) {
		if (kill(p, 0) != 0) {
			say("stopped (pid %d)", (int)p);
			return 0;
		}
		struct timespec ts = { 0, 100000000 };
		nanosleep(&ts, NULL);
	}
	say("pid %d did not stop; leaving it alone rather than forcing it", (int)p);
	return 1;
}

static void usage(void)
{
	printf(
	 "betacensor -- run the censor and its tools\n"
	 "\n"
	 "  betacensor                    censor the screen, with the settings panel\n"
	 "  betacensor live [out] [secs]  the same, naming an output and a duration\n"
	 "                                (default: the first output, 24 hours)\n"
	 "  betacensor gui                settings panel on its own\n"
	 "  betacensor video              batch a video file, with its own panel\n"
	 "  betacensor video IN OUT [m]   batch one file without the panel\n"
	 "  betacensor image [IN...]      censor images or folders, with its own\n"
	 "                                panel to pick what to cover\n"
	 "  betacensor image -o DIR IN... the same without the panel; see\n"
	 "                                'betacensor image --help'\n"
	 "\n"
	 "  betacensor panic              blank the screen entirely; run it\n"
	 "                                again to undo (it is a toggle)\n"
	 "  betacensor pause              show the real screen; again to undo\n"
	 "  betacensor stop               stop the running censor\n"
	 "  betacensor status             is one running, and with what settings\n"
	 "  betacensor doctor             what this machine supports, and what\n"
	 "                                that means for the censor\n"
	 "\n"
	 "Missing binaries are built on demand. BC_MODEL overrides the model in\n"
	 "the config for one run.\n"
	 "\n"
	 "Before the live censor works, KWin needs a window rule on class\n"
	 "\"betacensor\" forcing Layer=Overlay and \"Hide from screenshots and\n"
	 "screen recordings\", with \"Match whole window class\" UNCHECKED --\n"
	 "see DESIGN.md section 7, phase 4. Without it the detector sees its own\n"
	 "censor bars and the output flickers.\n");
}

int main(int argc, char **argv)
{
	if (!getcwd(caller_cwd, sizeof caller_cwd))
		caller_cwd[0] = 0;
	chdir_to_project();

	const char *cmd = argc > 1 ? argv[1] : "live";
	int   rest_c = argc > 2 ? argc - 2 : 0;
	char **rest  = argc > 2 ? argv + 2 : argv + argc;

	if (!strcmp(cmd, "-h") || !strcmp(cmd, "--help") || !strcmp(cmd, "help")) {
		usage();
		return 0;
	}
	if (!strcmp(cmd, "live"))
		return cmd_live(rest_c, rest);
	if (!strcmp(cmd, "gui") || !strcmp(cmd, "settings"))
		return cmd_gui();
	if (!strcmp(cmd, "video"))
		return cmd_video(rest_c, rest);
	if (!strcmp(cmd, "image") || !strcmp(cmd, "images"))
		return cmd_image(rest_c, rest);
	/*
	 * Both signals toggle in the core, so each of these is its own undo.
	 * There is deliberately no 'resume': from out here we cannot tell
	 * which of the two is in force, and a command that sent both would
	 * switch on whichever one was off -- causing the state it claims to
	 * clear. The core logs the state on every change; that is the place
	 * to read it.
	 */
	if (!strcmp(cmd, "panic"))
		return signal_running(SIGUSR1, "panic toggled");
	if (!strcmp(cmd, "pause"))
		return signal_running(SIGUSR2, "pause toggled");
	if (!strcmp(cmd, "stop"))
		return cmd_stop();
	if (!strcmp(cmd, "status"))
		return cmd_status();
	/*
	 * Deliberately a script rather than C. Everything it reports is the
	 * output of some other tool -- pkg-config, lspci, pgrep, eglinfo -- and
	 * the one machine that most needs it is the one where this program does
	 * not build, where a compiled subcommand would not exist to run.
	 */
	if (!strcmp(cmd, "doctor"))
		return run_script("tools/doctor.sh");

	/* An unrecognised first argument is almost always an output name. */
	if (cmd[0] != '-')
		return cmd_live(argc - 1, argv + 1);

	usage();
	return 2;
}
