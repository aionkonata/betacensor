/*
 * bc-video -- censor a video file offline (DESIGN.md §7, Beta Blocker parity).
 *
 * File in, censored file out. The detector, the tracker, the config and the
 * effect shaders are the same code the live censor runs; only the two ends
 * differ. Instead of a PipeWire capture feeding a Wayland overlay, frames come
 * from a decoder pipe and go back out to an encoder pipe.
 *
 * Why pipes and not libav*: this program would otherwise gain a large build
 * dependency to do something ffmpeg already does well, and the pipe boundary is
 * raw RGB -- there is no format negotiation to get subtly wrong. ffmpeg becomes
 * a runtime tool, like a shell command, not a link-time dependency. The Makefile
 * does not change.
 *
 * Two properties worth stating because they are design choices, not accidents:
 *
 * [1] Nothing here is real-time. The live path drops frames when inference
 *     falls behind -- it must, there is a screen to keep up with. Here every
 *     frame is inferred and nothing is dropped, so a file gets strictly better
 *     coverage than the same content would get live.
 *
 * [2] It is deterministic. Track timing comes from the frame index and the
 *     frame rate, not from the wall clock, so the same input produces the same
 *     output byte for byte regardless of how loaded the machine is. That is
 *     what makes a censored file reproducible, and what makes a regression here
 *     testable at all. The one deliberate exception is the hardening pattern
 *     (offline.h), which is random per run so it cannot be predicted and
 *     subtracted; BC_HARDEN_SEED pins it when a reproducible file is wanted.
 *     Within a run it is fixed to the frame rather than to time, so averaging
 *     the frames of a still scene does not average it away.
 *
 * Audio is passed through untouched: it is copied from the original by the
 * encoder, never decoded and never re-encoded.
 */

#include "config.h"
#include "detect.h"
#include "track.h"
#include "preprocess.h"
#include "join.h"
#include "offline.h"
#include "util.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>

#define MAX_BOX 64

/* ------------------------------------------------------------------ */
/* Probing the input                                                   */

struct vinfo {
	int    w, h;
	double fps;
	long   frames;      /* 0 if the container does not say */
	int    has_audio;
};

static int probe(const char *path, struct vinfo *v)
{
	char q[4200], cmd[4400], line[256];
	if (bc_off_shq(q, sizeof q, path) != 0) {
		ERR("path too long: %s", path);
		return -1;
	}

	snprintf(cmd, sizeof cmd,
	         "ffprobe -v error -select_streams v:0 -show_entries "
	         "stream=width,height,r_frame_rate,nb_frames "
	         "-of default=nw=1:nk=1 %s 2>/dev/null", q);
	FILE *f = popen(cmd, "r");
	if (!f)
		return -1;

	int  got = 0;
	long nb = 0;
	v->w = v->h = 0;
	v->fps = 0.0;
	while (fgets(line, sizeof line, f)) {
		char *s = line;
		while (*s == ' ')
			s++;
		switch (got++) {
		case 0: v->w = atoi(s); break;
		case 1: v->h = atoi(s); break;
		case 2: {
			/* r_frame_rate is a rational, "30000/1001". */
			long num = 0, den = 1;
			if (sscanf(s, "%ld/%ld", &num, &den) == 2 && den)
				v->fps = (double)num / (double)den;
			else
				v->fps = atof(s);
			break;
		}
		case 3: nb = atol(s); break;
		}
	}
	pclose(f);
	v->frames = nb > 0 ? nb : 0;

	if (v->w <= 0 || v->h <= 0) {
		ERR("ffprobe found no video stream in %s", path);
		return -1;
	}
	/*
	 * A missing or absurd frame rate would silently corrupt track timing
	 * rather than fail, so it is bounded rather than trusted.
	 */
	if (!(v->fps > 0.1 && v->fps < 1000.0)) {
		LOG("WARN: frame rate %.3f is implausible; assuming 30", v->fps);
		v->fps = 30.0;
	}

	snprintf(cmd, sizeof cmd,
	         "ffprobe -v error -select_streams a:0 -show_entries "
	         "stream=index -of default=nw=1:nk=1 %s 2>/dev/null", q);
	v->has_audio = 0;
	if ((f = popen(cmd, "r"))) {
		if (fgets(line, sizeof line, f))
			v->has_audio = 1;
		pclose(f);
	}
	return 0;
}

static void usage(void)
{
	fprintf(stderr,
	    "usage: bc-video [--no-harden] [--no-badge] <input> <output> "
	    "[model.onnx]\n"
	    "\n"
	    "Censors every frame of <input> and writes <output>. Settings come\n"
	    "from betacensor.conf, the same file the live censor and the GUI\n"
	    "use, so an effect set up on screen is the effect written to file.\n"
	    "Audio is copied from the original untouched.\n"
	    "\n"
	    "  --no-harden       leave covers as drawn (by default, covers that\n"
	    "                    sample the picture get a random per-block shift,\n"
	    "                    grain and a faint watermark)\n"
	    "  --no-badge        no \"censored with BetaCensor\" badge\n"
	    "\n"
	    "  BC_VIDEO_CONFIG   config file (default betacensor.conf)\n"
	    "  BC_VIDEO_ENC      encoder flags (default: libx264 -crf 18\n"
	    "                    -preset veryfast)\n"
	    "  BC_HARDEN_SEED    hex seed for the hardening pattern (default:\n"
	    "                    random per run -- keep it secret if set)\n");
}

int main(int argc, char **argv)
{
	int harden = 1, badge = 1;
	while (argc > 1 && argv[1][0] == '-' && argv[1][1] == '-') {
		if (!strcmp(argv[1], "--no-harden"))
			harden = 0;
		else if (!strcmp(argv[1], "--no-badge"))
			badge = 0;
		else if (!strcmp(argv[1], "--")) {
			argv++, argc--;
			break;
		} else {
			usage();
			return 2;
		}
		argv++, argc--;
	}
	if (argc < 3) {
		usage();
		return 2;
	}
	const char *in_path  = argv[1];
	const char *out_path = argv[2];
	const char *model    = argc > 3 ? argv[3] : NULL;
	/* One pattern for the whole file: see [2] above. */
	const uint32_t harden_seed = bc_off_mix_seed(bc_off_run_seed(), 0);
	struct bc_off_badge bdg = { 0 };

	/*
	 * Write to a dead encoder and the default action for SIGPIPE is to
	 * terminate us on the spot -- no summary, no error, exit 141. The one
	 * time that happened, the only thing on screen was the *decoder*
	 * complaining that its pipe had closed, which points at the wrong end
	 * of the program entirely. Ignoring it turns the same event into an
	 * EPIPE from fwrite, which the frame loop already knows how to report.
	 */
	signal(SIGPIPE, SIG_IGN);

	struct vinfo vi;
	if (probe(in_path, &vi) != 0)
		return 1;
	/* Made once: every frame is the same size. */
	if (badge && bc_off_badge_make(&bdg, vi.w, vi.h) != 0) {
		WARN("%dx%d is too small for a legible badge; leaving it off",
		     vi.w, vi.h);
		badge = 0;
	}

	const char *cfg_path = getenv("BC_VIDEO_CONFIG");
	struct bc_config cfg;
	bc_config_load(&cfg, cfg_path ? cfg_path : "betacensor.conf");
	if (!model && cfg.model[0])
		model = cfg.model;
	if (!model) {
		ERR("no model: pass one as the third argument or set it in the config");
		return 1;
	}

	LOG("input  %s: %dx%d @ %.3f fps%s%s", in_path, vi.w, vi.h, vi.fps,
	    vi.frames ? "" : " (length unknown)",
	    vi.has_audio ? ", with audio" : ", no audio");

	/*
	 * 640x640 is only a request: a model with a fixed input overrides it.
	 * One with a dynamic input keeps it, and asking for 0x0 -- as this
	 * used to -- left such a model with no input size at all, so it could
	 * not run here (puryfi's YOLO26 is one).
	 */
	struct bc_detector *det = bc_detect_create(model, 640, 640, 0);
	if (!det) {
		ERR("could not load model '%s'", model);
		return 1;
	}
	/*
	 * Reloading the config now that the detector has published its
	 * taxonomy: class names are looked up by name, and before this point
	 * they resolve against the built-in table rather than the model's own.
	 */
	bc_config_load(&cfg, cfg_path ? cfg_path : "betacensor.conf");
	bc_detect_set_conf(det, cfg.confidence * 0.5f);

	int net_w = bc_detect_net_w(det), net_h = bc_detect_net_h(det);
	LOG("model  %s: net %dx%d, %s", model, net_w, net_h,
	    bc_detect_on_gpu(det) ? "GPU" : "CPU");

	struct bc_off_gl g;
	if (bc_off_gl_init(&g, vi.w, vi.h) != 0)
		return 1;

	struct bc_preproc *pp = bc_preproc_create(net_w, net_h);
	if (!pp) {
		ERR("preprocess init failed");
		return 1;
	}
	float *tensor = malloc(bc_preproc_total_floats(pp) * sizeof *tensor);

	struct bc_tracker tr;
	bc_tracker_init(&tr, cfg.confidence, 0.30f, cfg.sticky_ms);
	tr.same_class = bc_off_classes_similar;
	tr.low_score  = cfg.confidence * 0.5f;

	const int effect = bc_off_effect(cfg.effect);

	/* ---- pipes ---- */
	char qin[4200], qout[4200], cmd[10240];
	if (bc_off_shq(qin, sizeof qin, in_path) != 0 ||
	    bc_off_shq(qout, sizeof qout, out_path) != 0) {
		ERR("path too long");
		return 1;
	}

	snprintf(cmd, sizeof cmd,
	         "ffmpeg -v error -i %s -f rawvideo -pix_fmt rgb24 -", qin);
	FILE *dec = popen(cmd, "r");

	const char *enc_flags = getenv("BC_VIDEO_ENC");
	if (!enc_flags || !*enc_flags)
		enc_flags = "-c:v libx264 -crf 18 -preset veryfast";
	/*
	 * The original is opened a second time purely as an audio source, and
	 * mapped with a trailing '?' so a file with no audio track is not an
	 * error.
	 *
	 * Deliberately no -shortest. It reads as "do not let a stream outrun
	 * the video", but it is symmetric: on a file whose audio is shorter
	 * than its picture -- which is common -- it ends the *video* at the
	 * audio, and four seconds of a six second clip were silently dropped.
	 * The video's length is the frames we feed, full stop. Audio that runs
	 * past the end is left alone; a slightly long tail is not a defect,
	 * and truncated picture is.
	 */
	snprintf(cmd, sizeof cmd,
	         "ffmpeg -v error -y -f rawvideo -pix_fmt rgb24 -s %dx%d "
	         "-framerate %.6f -i - %s%s -map 0:v:0 %s %s -pix_fmt yuv420p "
	         "%s %s",
	         vi.w, vi.h, vi.fps,
	         vi.has_audio ? "-i " : "", vi.has_audio ? qin : "",
	         vi.has_audio ? "-map 1:a:0? -c:a copy" : "",
	         enc_flags, "", qout);
	FILE *enc = popen(cmd, "w");

	if (!dec || !enc) {
		ERR("could not start ffmpeg -- is it installed?");
		return 1;
	}

	const size_t fsz = (size_t)vi.w * vi.h * 3;
	unsigned char *rgb  = malloc(fsz);
	unsigned char *back = malloc(fsz);
	if (!rgb || !back || !tensor) {
		ERR("out of memory for %dx%d frames", vi.w, vi.h);
		return 1;
	}

	int test_box[4] = { 0, 0, 0, 0 };
	if (getenv("BC_VIDEO_TEST_BOX"))
		sscanf(getenv("BC_VIDEO_TEST_BOX"), "%d,%d,%d,%d",
		       &test_box[0], &test_box[1], &test_box[2], &test_box[3]);

	int    write_failed = 0, detector_hung = 0;
	long   frame = 0, censored_frames = 0, blanked_frames = 0;
	double covered_px = 0.0;
	double t0 = bc_off_now_ms();

	while (fread(rgb, 1, fsz, dec) == fsz) {
		/*
		 * Track time comes from the frame index, not the clock: this
		 * pass is not real-time, and timing it by the clock would make
		 * the tracker's memory depend on how fast the machine happened
		 * to be decoding.
		 */
		const double t_ms = (frame * 1000.0) / vi.fps;

		bc_off_upload(&g, rgb);

		struct bc_det dets[BC_MAX_DET];
		int hung = 0;
		struct bc_det_result res =
		    bc_off_detect(&g, pp, det, tensor, dets, BC_MAX_DET, &hung);
		/*
		 * No outcome at all: the detector is hung. Carrying on would hand
		 * its eventual late result to the next frame, so stop here, with
		 * everything written so far still censored.
		 */
		if (hung) {
			ERR("detector gave no answer for frame %ld in 10 s "
			    "-- stopping", frame);
			detector_hung = 1;
			goto done;
		}
		const int n_det = res.status == BC_DET_OK ? res.n : 0;

		/*
		 * §6, offline. No result for this frame -- preprocess failed,
		 * the detector refused it, or the run errored -- means nobody
		 * looked at it, and zero detections would write it out
		 * uncensored while the run still reported SUCCESS. Write it
		 * black instead, and say so at the end.
		 */
		if (res.status != BC_DET_OK) {
			memset(back, 0, fsz);
			blanked_frames++;
			censored_frames++;
			covered_px += (double)vi.w * vi.h;
			goto write_frame;
		}

		/* Network coordinates to frame pixels, with §2's dilation. */
		struct bc_det sd[BC_MAX_DET];
		int ns = 0;
		for (int i = 0; i < n_det && ns < BC_MAX_DET; i++) {
			if (dets[i].cls < 0 || dets[i].cls >= BC_MAX_CLASSES ||
			    !cfg.censor[dets[i].cls])
				continue;
			float x0, y0, x1, y1;
			bc_preproc_to_screen(pp, dets[i].x, dets[i].y, &x0, &y0);
			bc_preproc_to_screen(pp, dets[i].x + dets[i].w,
			                     dets[i].y + dets[i].h, &x1, &y1);
			float w = x1 - x0, h = y1 - y0;
			float dx = w * cfg.dilate / 200.0f;
			float dy = h * cfg.dilate / 200.0f;
			sd[ns] = dets[i];
			sd[ns].x = x0 - dx;
			sd[ns].y = y0 - dy;
			sd[ns].w = w + 2 * dx;
			sd[ns].h = h + 2 * dy;
			ns++;
		}

		bc_tracker_update(&tr, sd, ns, t_ms, cfg.censor, NULL);

		struct bc_box boxes[MAX_BOX];
		int nb = 0;
		for (int k = 0; k < BC_TRK_MAX && nb < MAX_BOX; k++) {
			float x, y, w, h;
			if (!bc_trk_box(&tr.t[k], &x, &y, &w, &h))
				continue;
			if (tr.t[k].cls >= 0 && tr.t[k].cls < BC_MAX_CLASSES &&
			    !cfg.censor[tr.t[k].cls])
				continue;
			/* Clamp to the frame; a scissor outside it draws nothing. */
			float x0 = x < 0.0f ? 0.0f : x;
			float y0 = y < 0.0f ? 0.0f : y;
			float x1 = x + w > vi.w ? (float)vi.w : x + w;
			float y1 = y + h > vi.h ? (float)vi.h : y + h;
			if (x1 - x0 < 1.0f || y1 - y0 < 1.0f)
				continue;
			/* Styled by the class that opened the track. */
			boxes[nb] = bc_off_box(&cfg, tr.t[k].cls,
			                       (int)x0, (int)y0,
			                       (int)(x1 - x0), (int)(y1 - y0));
			covered_px += (double)boxes[nb].w * boxes[nb].h;
			nb++;
		}
		/*
		 * BC_VIDEO_TEST_BOX="x,y,w,h" pins a box regardless of what was
		 * detected, so the render path can be exercised on content that
		 * has nothing to find in it.
		 */
		if (test_box[2] > 0 && nb < MAX_BOX) {
			boxes[nb] = (struct bc_box){
				.x = test_box[0], .y = test_box[1],
				.w = test_box[2], .h = test_box[3],
				.effect = effect, .strength = cfg.strength,
				.text_mode = cfg.text_mode,
				.text_rgb = { cfg.text_rgb[0], cfg.text_rgb[1],
				              cfg.text_rgb[2] }
			};
			snprintf(boxes[nb].text, sizeof boxes[nb].text, "%s",
			         cfg.caption[0] ? cfg.caption : "CENSORED");
			covered_px += (double)boxes[nb].w * boxes[nb].h;
			nb++;
		}
		/* Same joining as the live pass, from the same config key, so
		 * a setting eyeballed on screen is what lands in the file. */
		if (cfg.join && nb > 1)
			nb = bc_join_boxes(boxes, nb, 6, cfg.join_waste);

		if (nb)
			censored_frames++;

		bc_off_draw(&g, boxes, nb, t_ms / 1000.0);
		if (harden)
			bc_off_harden(&g, boxes, nb, harden_seed);
		bc_off_read(&g, back);
		if (badge)
			bc_off_badge_apply(&bdg, back);

write_frame:
		if (fwrite(back, 1, fsz, enc) != fsz) {
			ERR("encoder stopped accepting frames at %ld "
			    "(%s) -- the output is truncated",
			    frame, strerror(errno));
			write_failed = 1;
			goto done;
		}

		frame++;
		if (vi.frames && frame % 100 == 0)
			LOG("  %ld/%ld frames (%.0f%%)", frame, vi.frames,
			    100.0 * frame / vi.frames);
		else if (!vi.frames && frame % 100 == 0)
			LOG("  %ld frames", frame);
	}

done: ;
	double secs = (bc_off_now_ms() - t0) / 1000.0;
	int dec_rc = pclose(dec);
	int enc_rc = pclose(enc);

	LOG("wrote %s", out_path);
	LOG("  frames                  : %ld", frame);
	LOG("  frames with a censor box: %ld (%.1f%%)", censored_frames,
	    frame ? 100.0 * censored_frames / frame : 0.0);
	LOG("  mean covered area       : %.2f%% of frame",
	    frame ? 100.0 * covered_px / ((double)frame * vi.w * vi.h) : 0.0);
	LOG("  detector errors         : %lu", bc_detect_errors(det));
	LOG("  frames blanked (no result): %ld", blanked_frames);
	LOG("  elapsed                 : %.1fs (%.1f fps, %.2fx realtime)",
	    secs, secs > 0 ? frame / secs : 0.0,
	    secs > 0 ? (frame / secs) / vi.fps : 0.0);

	free(rgb);
	bc_off_badge_free(&bdg);
	free(back);
	free(tensor);
	bc_preproc_destroy(pp);
	/* Destroying joins the worker, which is exactly what has hung. */
	if (!detector_hung)
		bc_detect_destroy(det);

	if (frame == 0) {
		ERR("no frames decoded -- is %s a video file?", in_path);
		return 1;
	}
	if (write_failed) {
		ERR("FAILED -- %s is short of the source", out_path);
		return 1;
	}
	if (detector_hung) {
		ERR("FAILED -- the detector hung; %s stops at frame %ld",
		    out_path, frame);
		return 1;
	}
	if (enc_rc != 0) {
		ERR("encoder exited with status %d", enc_rc);
		return 1;
	}
	/*
	 * Only judged on a run that read to the end -- the early exits above
	 * close the pipe on a decoder that is still writing, which fails it.
	 * Here it is the difference between "the file ended" and "the decoder
	 * died partway", and the second used to report SUCCESS on a
	 * truncated output.
	 */
	if (dec_rc != 0) {
		ERR("FAILED -- the decoder exited with status %d after %ld "
		    "frames; %s is short of the source", dec_rc, frame, out_path);
		return 1;
	}
	if (blanked_frames) {
		ERR("FAILED -- %ld of %ld frames had no detection result and were "
		    "written black; see the detector errors above",
		    blanked_frames, frame);
		return 1;
	}
	LOG("SUCCESS -- %ld frames censored.", frame);
	return 0;
}
