/*
 * BetaCensor-linux -- Phase 3, the joined pipeline (DESIGN.md §7).
 *
 *   portal -> PipeWire DMA-BUF -> EGLImage -> GL texture
 *          -> compute-shader letterbox/normalise -> tensor
 *          -> NudeNet (worker thread) -> boxes -> layer-shell overlay
 *
 * Everything runs on one EGLDisplay -- the overlay's -- because EGLImages
 * cannot cross displays. The GL thread never blocks on inference.
 */

#include "portal.h"
#include "capture.h"
#include "gpu.h"
#include "overlay.h"
#include "framering.h"
#include "delay.h"
#include "preprocess.h"
#include "detect.h"
#include "failsafe.h"
#include "config.h"
#include "join.h"
#include "seam.h"
#include "eyebar.h"
#include "track.h"
#include "hold.h"
#include "landmark.h"
#include "util.h"

#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <dirent.h>
#include <time.h>
#include <math.h>

/*
 * No default monitor name. A connector name is the most machine-specific
 * string there is -- DP-3 here is HDMI-A-1 or eDP-1 on the next box -- and
 * naming one that does not exist is a hard failure at startup, which is a
 * terrible first run. NULL means "the first output the compositor lists",
 * which is right on a single-monitor machine and a coin flip on a multi-head
 * one; the name is how you win the coin flip, so it stays available as argv[1]
 * and the startup log prints every output it saw.
 */
#define DEFAULT_OUTPUT  NULL
/*
 * 640m, not the 320n nano model. Measured on CUDA both run at the same ~29
 * inferences/s -- the change gate limits the rate, not the model -- so the
 * larger one is free in practice, and it is visibly steadier: per-detection box
 * jump is 15.8 px against 320n's ~30 px, which is box-coordinate jitter no
 * amount of tracker work removed. 14.8 ms an inference.
 *
 * Note this needs the GPU. On CPU it is 118 ms, and the CUDA EP falls back to
 * CPU with only a warning, so check for [CUDA] in the detector line if it feels
 * slow.
 */
/*
 * Hotscreen's anime-and-real y11s at 640. Chosen on visual judgement over 640m,
 * which wins every counter we have and is still the worse censor: churn, box
 * jump and box-count changes all measure *steadiness*, and a model that finds
 * more distinct regions is penalised by all three. Tracks created on identical
 * content: 5 for 640m, 36 for this. For a censor recall is the property that
 * matters and jitter is the comfort one -- and jitter is tunable (POS_ALPHA,
 * HYST_RATIO, sticky_ms) where recall is not.
 */
#define DEFAULT_MODEL   "vendor/hs-real-anime-y11s-640-fp32.onnx"
#define DEFAULT_RESTORE ".restore_token"
#define DEFAULT_CONFIG  "betacensor.conf"
#define DEFAULT_PIDFILE "betacensor.pid"
#define DEFAULT_SECS    20.0
/*
 * 24 h. This was 600 s while the program only ever ran in test bursts, but an
 * over-limit value falls back to DEFAULT_SECS rather than erroring -- so asking
 * for an hour silently got you twenty seconds, which is a nasty surprise for
 * anyone launching it for real. Still bounded rather than infinite: a censor
 * that has been running unattended for a day has almost certainly been
 * forgotten about.
 */
#define HARD_LIMIT      86400.0

/* How often to stat() the config file. 4 Hz is instant to a human and costs
 * nothing, where checking every frame would be 180 syscalls a second. */
#define CONFIG_POLL_MS  250.0

/*
 * How long to present empty frames before starting the screencast, so KWin has
 * applied the "Hide from screenshots and screen recordings" rule to the mapped
 * surface first. If the stream starts before the rule lands, our own bars enter
 * the captured frame and feed back into the detector.
 */
#define OVERLAY_SETTLE_MS 500.0

/* §6 fail-closed deadlines (STALL_MS, INFER_DEAD_MS): see failsafe.h. */

/* The longest a peek may keep the overlay off the screen. */
#define PEEK_MAX_MS 40.0
/*
 * How long the clean-frame gate may go without a clean frame before it
 * gives up and lets a dirty one through.
 *
 * The gate improves the picture by refusing evidence it does not trust.
 * If clean frames stop arriving -- no peeks granted, no timestamps, a
 * compositor that behaves in some way not anticipated here -- refusing
 * everything means detecting nothing and drawing nothing, which is a
 * blank censor over uncensored content. A degraded detection is worth
 * more than none, so after this long the gate stands down and says so.
 */
#define GATE_BLIND_MS 3000.0

/* §2 mitigations. */
#define TRACK_IOU    0.30f   /* match a new detection to an existing track */
static float track_iou_thresh(void)
{
	const char *e = getenv("BC_TRACK_IOU");
	if (e) { float v = strtof(e, NULL); if (v > 0.f && v < 1.f) return v; }
	return TRACK_IOU;
}
/*
 * The box array's ceiling is BC_MAX_BOXES (overlay.h), not the track count.
 * A MAX_TRACKS spelled as BC_MAX_DET used to size it, which read as correct
 * and was not: a track can emit two boxes, so the array overflowed well before
 * the tracker ran out of tracks.
 */

/*
 * Motion prediction (DESIGN.md §7 phase 4, "tracker").
 *
 * DESIGN.md §3[6] assumed detection runs at 10-30 Hz, which would leave a box
 * 33-100 ms behind the content. Measured, it runs at 68-138 Hz: preprocess is
 * 0.31 ms, inference 5.0 ms, and detections land every ~15 ms under motion, so
 * a displayed box is 5-20 ms stale, ~13 ms on average. At a plausible on-screen
 * speed of 500 px/s that is 6 px -- already inside the 12% dilate margin. So
 * predicting a *freshly detected* box forward buys close to nothing, and the
 * premise the tracker was planned on does not hold on this machine.
 *
 * What is left is the case the plan did not name: a *held* box. Sticky keeps a
 * box alive up to 500 ms past its last detection, and a frozen box over content
 * that is still moving drifts off it entirely -- 500 ms at 500 px/s is 250 px.
 * That is where prediction earns its place, so the horizon below is what it is
 * sized for.
 *
 * The extrapolation *grows* the box to the union of where it was and where it
 * is predicted to be, rather than translating it. Translating is fail-open: if
 * the prediction is wrong the bar has been moved off content that was really
 * there. A union is wrong only by covering too much, which §6 says is the side
 * to be wrong on.
 */
/*
 * Score hysteresis and edge smoothing.
 *
 * The single confidence threshold was the real cause of the flicker-vs-linger
 * trade: the model's score for content it is already looking at oscillates
 * across the threshold between inferences, so a box died on every dip and
 * `sticky_ms` was the only thing papering over it -- which is why no value of
 * sticky could ever be right. Sticky cannot distinguish "the detector blinked"
 * from "the content left". Two thresholds can.
 */
/*
 * Fade-out, adapted from Hotscreen's box transitions (HOTSCREEN.md [3]).
 *
 * Fade *out* only. Hotscreen also fades boxes in -- for its purposes a bar
 * rising slowly out of nothing is the point -- but for a censor that is
 * fail-open: every millisecond of a ramp is content visible through a bar that
 * is not opaque yet. A new bar goes up instantly and at full strength.
 *
 * Fading out is the safe half. A track only expires once nothing has detected
 * it for `sticky_ms`, and at that moment the current code makes the bar vanish
 * between one frame and the next. Ramping it away over the tail of that window
 * covers exactly the same pixels for exactly as long, and merely stops the
 * disappearance being a step change -- which is what reads as flicker.
 */
/*
 * How long the change gate may keep a track alive without a genuine detection.
 * Short on purpose: see the note at its use site.
 */
#define GATE_HOLD_MS  1000.0

/*
 * How long the compositor must send nothing at all before we treat the screen
 * as still. Far above any real inter-frame gap on purpose: see the use site.
 */
#define QUIET_MS      250.0

/*
 * How much of the frame must differ before it counts as a new picture rather
 * than movement within the old one. A third is far above anything an object
 * moving through a scene produces and far below a scroll, which replaces
 * nearly all of it.
 */
#define SCENE_FRAC    0.33f

/*
 * Slack allowed when deciding whether two covers touch. Detections of parts of
 * one body routinely stop a few pixels short of each other; a strict test
 * leaves a hairline of real content between two regions that are obviously one.
 */
#define BC_JOIN_GAP   6

/*
 * How much of the overlap strip two halves of a straddling detection must
 * share before they are treated as one object. Hotscreen's figure. The
 * denominator is what the pair could physically reach rather than their union,
 * so this is a far less delicate number than an IoU threshold would be.
 */
#define SEAM_THRESH   0.4f

#define FADE_OUT_MS   120.0

#define HYST_RATIO    0.5f    /* sustain an existing track at conf*this */
#define POS_ALPHA     0.35f   /* per-edge ease, inward direction only */

#define PRED_ALPHA    0.35f   /* EMA weight on each new velocity sample */
#define PRED_MAX_MS   250.0
#define PRED_MAX_FRAC 0.5f    /* extrapolate at most this much of the box */

/*
 * Eye bar, derived from a face box rather than detected directly.
 *
 * The y11 taxonomies carry an EYE class but never predict it -- measured at 0
 * detections against 386 faces on the same frames -- and Hotscreen's dedicated
 * eye model, while it does fire, produces boxes small and jittery enough that
 * smoothing and motion prediction ratchet them outward until they cover the
 * whole face. Faces, by contrast, are the single most stable thing either model
 * reports.
 *
 * So take the stable measurement and cut the band out of it.
 *
 * The band is symmetric about the face's centre rather than sitting in the upper
 * third where the eyes actually are. An upright face puts the eye line just under
 * 40% down, and a tight band there works -- until the subject is lying down or
 * upside down, where it misses the eyes completely. A face box is axis-aligned
 * and carries no orientation, and aspect ratio cannot recover it either: an
 * inverted face has exactly the same shape as an upright one, so a "top band"
 * lands on the chin.
 *
 * A centred band covers the eye line at any rotation, because whatever the angle
 * the eyes stay near the middle of the head. It covers more than strictly the
 * eyes, which is the trade: correct at every orientation, versus tight but only
 * correct when the subject is straight. Covering too much is the side §6 says to
 * err on.
 *
 * Getting a tight band at any angle needs an actual eye position -- see the note
 * about running eyes-detection-01 as a second detector.
 */
/*
 * The eye band, as a fraction of the face box height from its top. A YOLO face
 * box runs roughly hairline to chin, which puts the eye line just under 40%
 * down.
 *
 * Two alternatives were built and measured against this, and both were worse:
 *
 * - A *centred* band (0.22-0.78) is correct at any rotation, because whatever
 *   the angle the eyes stay near the middle of the head. It is also too tall at
 *   every angle -- it covers most of the face, which is the thing this feature
 *   exists to avoid.
 * - Hotscreen's eyes-detection-01 as a second detector, placing the bar from
 *   actual eye positions and taking the angle from the line between two eye
 *   centres. It works, and it is the only approach that handles rotation
 *   tightly, but the detections are jittery and intermittent enough (341 bars
 *   placed from eyes against 237 falling back over 20 s) that the result reads
 *   worse than a bar that is simply in the right place most of the time.
 *
 * So: the tight geometric band is the default. It is wrong when the subject is
 * lying down or upside down -- a face box carries no orientation and aspect
 * ratio cannot recover it, since an inverted face has exactly the same shape as
 * an upright one. That is a known limitation, accepted deliberately over both
 * alternatives. The eye model remains available behind BC_EYE_MODEL.
 */
#define EYE_MODEL_DEFAULT "vendor/eyes-detection-01-fp32.onnx"
#define EYE_PAD       0.35f   /* grow the eye union by this much of its height */
#define EYE_FRESH_MS  400.0   /* ignore eye boxes older than this */
/*
 * The centred band: 22-78% of the face box, not the 26-52% that used to be
 * here.
 *
 * The tight band is correct only for an upright head, and nothing available
 * can tell us whether a head is upright. Five sources were tried and every one
 * came up empty on drawn content:
 *
 *   body axis          geometrically wrong -- assumes the torso is straight
 *                      below the head, which portrait framing breaks; measured
 *                      35 deg mean tilt on upright subjects
 *   y11 EYE class      0 detections against 386 faces
 *   dedicated eye model  boxes so jittery that smoothing and prediction
 *                      ratchet them out over the whole face
 *   WIDERFACE landmarks  trained on photographs; on drawn faces the box lands
 *                      and the landmarks regress elsewhere, confidently
 *   face box aspect    w/h ran 1.04-1.14 across 111 faces, upright and lying
 *                      down alike -- no signal at all
 *
 * So the choice is a band that is right when the subject happens to be upright
 * and misses the eyes otherwise, or one that covers the eye line at every
 * rotation because the eyes stay near the middle of the head whatever the
 * angle. The second covers more of the face, which is the cost; the first
 * uncovers, which §2 and §6 both refuse.
 *
 * This was measured and rejected once before as "too tall at every angle".
 * That judgement assumed orientation could be recovered some other way. It
 * cannot, and confirmed on video: at 22-78% all three test subjects were
 * covered, at 26-52% only the upright one was.
 *
 * BC_EYEBAR_TOP/BC_EYEBAR_BOT put the tight band back for anyone whose content
 * is reliably upright, or photographic enough for BC_POSE_MODEL to do better.
 */
#define EYEBAR_TOP    BC_EB_BAND_TOP
#define EYEBAR_BOT    BC_EB_BAND_BOT
#define EYEBAR_INSET  BC_EB_INSET   /* trim each side, so the bar sits inside the face */
/*
 * 5-point face models (WIDERFACE/RetinaFace convention) put the two eyes at
 * keypoints 0 and 1. The bar is sized from the eye separation, which scales
 * with the face and, unlike the face box, is already in the eyes' own frame.
 */
#define KPT_MIN_CONF  0.50f   /* both eyes must be this confident to be used */
/*
 * Most faces given to the landmark model in one frame, and the mean heatmap
 * peak below which its answer is not trusted. The peak ran 0.80-0.90 on real
 * faces at every rotation during the export check, so 0.35 rejects only a
 * crop the model has no opinion about -- a bar drawn from a confident wrong
 * answer is worse than the band.
 */
#define BC_LM_MAX_FACES 8
#define BC_LM_MIN_SCORE BC_EB_LM_MIN_SCORE
/* Face box padded by this before cropping, matching the package's
 * BBOX_PADDING: the model was trained on crops with this much context. */
#define BC_LM_PAD     BC_EB_LM_PAD
/* Grow the measured eye extents by this much, so the bar covers lids and
 * lashes rather than stopping at the outermost landmark. */
#define LM_BAR_PAD    BC_EB_LM_BAR_PAD

#define POSE_MIN_IOU  0.30f   /* pose face must overlap the primary face box this much */
#define AXIS_NEAR     0.35f   /* torso closer than this * face height: degenerate */
#define AXIS_FAR      8.0f    /* ...further than this * face height: someone else */   /* extrapolation horizon; beyond this, coast */
#define VEL_MAX_PXMS  6.0f    /* 6000 px/s -- faster than anything real here */
#define VEL_DT_MIN    2.0     /* ignore samples too close together to divide */
#define VEL_DT_MAX    200.0   /* ...and too far apart to still be the same motion */

/*
 * Mean-absolute-difference below which a region counts as unchanged, in
 * normalised [0,1] units. Deliberately small: we are looking for "identical",
 * not "similar".
 */
/*
 * Change gate (DESIGN.md §7 phase 4). Inference costs ~4.8 ms of CPU per run at
 * ~138 runs/s -- about two thirds of a core -- and on a desktop that is not
 * moving, every one of those runs re-derives the answer we already have.
 *
 * The test is deliberately *not* a whole-frame mean. A mean is fail-open: a
 * small region moving inside an otherwise static screen averages down to nearly
 * nothing, and that region is exactly what must not be missed. So the gate asks
 * "did any pixel change at all", with the epsilon below half an 8-bit LSB --
 * frames that are byte-identical pass, and a single changed pixel anywhere
 * fails. Nothing is skipped that could have moved.
 */
#define GATE_EPS     (0.5f / 255.0f)

#define HOLD_MAD     0.010f
#define HOLD_MARGIN  0.25f   /* width of the ring sampled around the box */

/*
 * Absolute ceiling on how long the hold may keep a box alive past its last
 * genuine detection. The ring below is the primary release mechanism; this
 * only guarantees a box can never become permanent, because the one thing
 * worse than a flickering censor bar is one the user cannot get rid of.
 */
#define HOLD_MAX_DEFAULT 30000.0

/*
 * Policy now lives in the config file (bc_config_defaults() holds what used to
 * be the hardcoded table here). The env vars survive as debugging overrides:
 * they are re-applied after every reload, so a value forced on the command line
 * cannot be silently taken back by the GUI writing the file.
 */
static void cfg_env_overrides(struct bc_config *c)
{
	const char *e;

	if (getenv("BC_FILL"))
		c->effect = BC_CFG_FILL;
	if ((e = getenv("BC_STRENGTH"))) {
		float v = strtof(e, NULL);
		if (v > 0.0f && v <= 100.0f)
			c->strength = v;
	}
	if ((e = getenv("BC_CONF"))) {
		float v = strtof(e, NULL);
		if (v > 0.0f && v < 1.0f)
			c->confidence = v;
	}
	if ((e = getenv("BC_STICKY"))) {
		float v = strtof(e, NULL);
		if (v >= 0.0f && v <= 5000.0f)
			c->sticky_ms = v;
	}
}

struct track {
	float  x, y, w, h;    /* screen pixels */
	float  vx, vy;        /* px/ms, EMA over consecutive detections */
	float  angle;         /* radians; bar orientation, 0 = upright */
	/*
	 * The face this bar came from. An eye bar is a claim about a region
	 * *inside* a face, so it is clamped to this at draw time -- dilate,
	 * motion prediction and the outward-instant smoothing all grow a box
	 * with nothing to stop them, and on an eye bar that compounded until it
	 * covered the whole face and kept going.
	 */
	float  fx, fy, fw, fh;
	int    eyebar;
	/*
	 * The body's up-axis, when some other part of the same body was
	 * detected and supplied one. Kept as the vector rather than as the
	 * angle: wrap_bar has a half-turn period, so an angle cannot say which
	 * end of the axis is the head, and the band would sit on the chin as
	 * readily as on the eyes.
	 *
	 * Remembered because body_axis() depends on a *second* detection
	 * landing in the same frame, which comes and goes. Without memory the
	 * bar alternated between the axis geometry and the upright band every
	 * time the torso dropped out for a frame -- a jump in both angle and
	 * size, which is what the eye bar was reported as doing.
	 */
	/*
	 * The bar came from measured eye landmarks, so the containment clamp
	 * below does not apply to it. That clamp exists because the old eye
	 * model's boxes jittered and smoothing ratcheted them out over the
	 * whole face; a landmark bar is anchored to eye positions and cannot
	 * drift that way. Keeping it would shrink rotated bars for a reason
	 * that is an artefact of the clamp's own geometry: it measures the
	 * bar's axis-aligned bounding box against an axis-aligned face box,
	 * and rotation grows the former while leaving the latter alone.
	 */
	int    lm_bar;
	/*
	 * The eye bar, when it is drawn *in addition* to the face rather than
	 * instead of it (cfg.eye_bar_effect). Stored as fractions of the face
	 * box, not as screen pixels: the tracker follows the face, and the bar
	 * has to move with it between detections. Fractions re-apply to
	 * whatever the face box currently is, so the bar tracks for free
	 * instead of needing a filter of its own.
	 */
	int    has_bar;
	float  bar_cx_f, bar_cy_f, bar_w_f, bar_h_f;
	float  ux, uy;
	int    have_axis;
	double axis_at;       /* ms, when that axis was last measured */
	int    cls;
	double last_seen;     /* ms, refreshed by detection OR by the hold */
	double last_detect;   /* ms, refreshed only by a genuine detection */
	double created;
	unsigned long n_detect, n_hold;
	int    used;
};

static volatile sig_atomic_t running = 1;

/*
 * Panic and pause (DESIGN.md §7 phase 4). Wayland gives an ordinary client no
 * global hotkeys, so the "key" is a signal: the user binds a KDE custom
 * shortcut to `kill -USR1 $(cat betacensor.pid)`.
 *
 * The handler only records that the signal arrived. Toggling and logging happen
 * in the loop, because nothing here may call anything that is not
 * async-signal-safe -- LOG() goes through stdio, which is not.
 */
static volatile sig_atomic_t sig_panic = 0, sig_pause = 0;

static void on_signal(int sig) { (void)sig; running = 0; }

static void on_user(int sig)
{
	if (sig == SIGUSR1)
		sig_panic = 1;
	else
		sig_pause = 1;
}

static void on_alarm(int sig)
{
	(void)sig;
	static const char m[] = "[bc] ERROR: watchdog fired; force-exiting.\n";
	ssize_t n = write(2, m, sizeof m - 1);
	(void)n;
	_exit(2);
}

/*
 * Ratio taking a length in one pixel space to another, guarding the case where
 * either is not known yet. Its own function because the guard is the whole
 * point: a zero here would scale every box to nothing, which is a screen that
 * looks censored-by-accident rather than an obvious failure.
 */
static float draw_scale(int from, int to)
{
	if (from <= 0 || to <= 0)
		return 1.0f;
	return (float)to / (float)from;
}

static double now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/*
 * A numeric debug knob, or `dflt` when the variable is unset.
 *
 * One lookup, held in one place. Written out longhand this is
 * `getenv(k) ? atof(getenv(k)) : dflt`, which reads the environment twice and
 * leaves a NULL dereference standing one edit away -- the second call is only
 * safe because nothing between the two can unset the variable, and that is not
 * a property anyone should have to re-verify per knob.
 *
 * An unparseable value yields 0.0, not `dflt`; every caller range-checks the
 * result, and "BC_FOO=nonsense behaves like BC_FOO=0" is easier to explain
 * than a value that silently reverts.
 */
static double env_num(const char *key, double dflt)
{
	const char *v = getenv(key);
	return v ? atof(v) : dflt;
}

/* Running mean/max, so the §2 table can be filled in with measurements. */
struct stat { double sum, max; unsigned long n; };

static void stat_add(struct stat *s, double v)
{
	s->sum += v;
	if (v > s->max)
		s->max = v;
	s->n++;
}
static double stat_mean(const struct stat *s)
{
	return s->n ? s->sum / (double)s->n : 0.0;
}


/*
 * The fail-closed primitive: one opaque black rectangle over the whole output.
 *
 * Solid fill, never pixelate. Pixelate samples the captured source texture, and
 * every situation that gets us here -- panic, a dead capture, a dead detector --
 * is one where that texture may be stale or absent. Showing a stale screen is
 * the exact failure this is meant to prevent, so the blank cannot depend on the
 * capture path being alive.
 */
static void blank_all(struct bc_overlay *ov, struct bc_box *boxes, int *n_boxes)
{
	*n_boxes = 0;
	boxes[(*n_boxes)++] = (struct bc_box){
		.x = 0, .y = 0,
		.w = bc_overlay_width(ov),
		.h = bc_overlay_height(ov),
		.r = 0.0f, .g = 0.0f, .b = 0.0f, .a = 1.0f,
		.effect = BC_EFFECT_FILL, .strength = 0.0f
	};
}

/*
 * Is this tensor identical to the previous one? Returns as soon as it finds a
 * difference, so the expensive case is the one where we are about to skip an
 * inference anyway -- a moving screen exits after a handful of floats.
 */
/*
 * What fraction of the frame is not what it was. The change gate asks whether
 * this is zero; the scene cut asks whether it is nearly everything, and the
 * answer comes from the same comparison so it costs one pass, not two.
 */
/*
 * The screen has not changed since the last inference, so the last results
 * still describe it. Refresh the tracks rather than expiring them: choosing
 * not to infer must not read as "nothing is there", or a still screen would
 * drop every box and uncover itself. Only the clocks move, so the skipped time
 * is never charged against a track's memory.
 *
 * The ceiling still applies. Evidence that nothing changed *since the last
 * frame* may carry a track across a brief still moment -- not across a scene.
 * Without it, boxes from content that had gone sat on screen while new
 * detections stacked on top: 41 tracks created and 0 expired in 30 s.
 *
 * The ceiling reads last_measured, not last_update, and that distinction is
 * the whole mechanism. The hold moves last_update forward, so a ceiling
 * reading it asks "has this track been held recently" -- to which the answer,
 * on the frame right after a hold, is always yes. It could never fire: three
 * separate 30 s runs reported "gate overridden by cap: 0" while tracks piled
 * up, 64 created and 0 expired. A still screen held every track forever, and
 * the boxes stacked until the screen was covered.
 *
 * That is also why only the mouse cleared them. Moving the cursor changes the
 * frame continuously, so nothing is gated, nothing is held, and the tracks
 * finally age out. Movement *inside* the picture comes in bursts with still
 * frames between, and every one of those still frames re-held everything.
 *
 * Returns 1 when a track hit the ceiling, meaning the caller no longer trusts
 * its box and must measure again rather than drop it blind.
 */
static int hold_or_cap(struct bc_tracker *tr, double tg, unsigned long *capped)
{
	int cap = bc_tracker_hold_bounded(tr, tg, GATE_HOLD_MS);
	*capped += (unsigned long)cap;
	return cap;
}

/*
 * Classes that describe the same physical thing, so a detection of one may
 * continue a track of the other (Hotscreen's IsSimilarClass, HOTSCREEN.md [5]).
 *
 * The model flips between these constantly on the same body part -- male vs
 * female is a judgement call it makes per frame, and covered vs exposed hinges
 * on a few pixels of fabric. Under strict class equality every flip fails to
 * match, expires the track and opens a new one, which is churn at best.
 *
 * At worst it is fail-open: FEMALE_BREAST_EXPOSED is censored by default and
 * MALE_BREAST_EXPOSED is not, so one flipped frame used to drop the bar off
 * content that had not changed at all. Pairing them means the track survives
 * and keeps the class it was created with.
 *
 * Matched by name rather than index because the taxonomy now comes from the
 * model, and the two models we have disagree on both order and count.
 */
static const char *SIMILAR_PAIRS[][2] = {
	{ "FACE_FEMALE",           "FACE_MALE"              },
	{ "FEMALE_FACE",           "MALE_FACE"              },  /* y11 naming */
	{ "FEMALE_BREAST_EXPOSED", "MALE_BREAST_EXPOSED"    },
	{ "FEET_COVERED",          "FEET_EXPOSED"           },
};

/*
 * Point the preprocessor and the detector at the same grid.
 *
 * They have to agree: the preprocessor decides how many tensors are produced
 * and the detector decides how many it runs, and a mismatch would either feed
 * the model a tensor built for a different tile or read past the end of the
 * buffer. Keeping the two calls together in one place is the cheapest way to
 * make that impossible to get wrong.
 */
/*
 * The tiles for "split" mode, from the screen's shape and the network input.
 *
 * Fit the screen's height to the network and see how many network-widths the
 * width comes to: `X`. On a 16:9 screen into a square input that is 1.78, so
 * two tiles, each the full height of the screen and square in screen pixels,
 * sharing the 0.22 of a tile they overlap by -- exactly enough that the scale is
 * the same on both axes and nothing is letterboxed. A taller screen splits down
 * instead; one that is nearly square needs no split at all.
 *
 * Why not just ask for a 2x1 grid: the overlap that makes the scales agree
 * depends on the screen, and a fixed one either wastes pixels (too much
 * overlap shrinks everything) or leaves a gap (too little). It is also why this
 * wants a square input: into 640x384 the same two tiles would be letterboxed to
 * a 384 square, which is the waste this exists to remove.
 */
static void split_geometry(int w, int h, int nw, int nh,
                           int *cols, int *rows, float *overlap)
{
	*cols = *rows = 1;
	*overlap = 0.0f;
	if (w <= 0 || h <= 0 || nw <= 0 || nh <= 0)
		return;
	const double x = (double)nh * w / ((double)nw * h);   /* net-widths across */
	const double y = (double)nw * h / ((double)nh * w);   /* net-heights down  */
	const int across = x >= y;
	const double axis = across ? x : y;
	if (axis < 1.3)
		return;                                   /* not worth a second pass */
	int tiles = (int)ceil(axis);
	if (tiles > BC_MAX_GRID)
		tiles = BC_MAX_GRID;
	*(across ? cols : rows) = tiles;
	double ov = ((double)tiles - axis) / (double)(tiles - 1);
	if (ov < 0.0) ov = 0.0;
	if (ov > 0.9) ov = 0.9;
	*overlap = (float)ov;
}

/*
 * The network input to ask a 640-trained model for. A dynamic model takes any
 * size, and the shape should follow what it will be fed: split tiles are
 * square, so a square input wastes none of them; one tile of the whole screen
 * is wide, so the short input saves the rows a letterbox would leave black.
 * Models not named 640 are left at whatever they were.
 */
static void want_net(const char *model, int split, int *w, int *h)
{
	if (strstr(model, "640")) {
		*w = 640;
		*h = split ? 640 : 384;
	}
}

static void apply_sections(struct bc_preproc *p, struct bc_detector *d,
                           const struct bc_config *c, int screen_w, int screen_h)
{
	if (!p)
		return;
	int cols = c->section_cols, rows = c->section_rows;
	float overlap = c->section_overlap;
	if (c->section_split)
		split_geometry(screen_w, screen_h, bc_preproc_net_w(p),
		               bc_preproc_net_h(p), &cols, &rows, &overlap);

	/*
	 * Both calls below reallocate a buffer an inference may be reading:
	 * set_grid() resizes the tensor SSBO, set_tiles() the host copies of it.
	 * That was always true -- see the note on bc_detect_set_tiles() -- but
	 * it used to be a race over a buffer the worker had memcpy'd out of, so
	 * the worst case was one garbled frame. With the tensor handed over in
	 * place it is the buffer the model is reading, unmapped and resized
	 * underneath it.
	 *
	 * Only reachable from a config change, so the cost is one inference of
	 * latency on a keystroke in the GUI.
	 */
	if (d)
		bc_detect_wait_idle(d);

	bc_preproc_set_grid(p, cols, rows, overlap, c->section_full_frame);
	if (d)
		bc_detect_set_tiles(d, bc_preproc_tiles(p));

	/*
	 * Re-taken here because set_grid() reallocates the tensor buffer, and a
	 * registration is against the buffer that existed when it was made. Only
	 * worth having if the other end can read device memory -- on a CPU
	 * session the tensor has to reach host memory regardless.
	 */
	if (d && bc_detect_zero_copy(d))
		bc_preproc_gpu_register(p);
}

/*
 * Hand the current tensor to the detector, without a host copy where both ends
 * can manage it.
 *
 * Ordering is not free choice. The gate's reference frame is copied out of the
 * SSBO first, because from the moment CUDA maps the buffer a GL copy out of it
 * is undefined. And if anything about the device path fails the buffer is given
 * straight back, so the readback below is operating on a buffer GL still owns.
 *
 * `want_host` forces the slow path for the one caller that needs the floats
 * itself (--dump-tensor).
 */
static int submit_tensor(struct bc_preproc *p, struct bc_detector *d,
                         float *host, int want_host, int *host_filled)
{
	if (host_filled)
		*host_filled = 0;
	/*
	 * Both ends have to agree. Either can decline -- a CPU session, a build
	 * without CUDA, a registration the driver refused -- and the readback
	 * below is what either falls back to, so neither is an error.
	 */
	if (!want_host && bc_detect_zero_copy(d) && bc_preproc_zero_copy(p)) {
		bc_preproc_commit(p);
		const void *dev = bc_preproc_device_ptr(p);
		if (dev) {
			if (bc_detect_submit_device(d, dev))
				return 1;
			/*
			 * Committed a frame that was then not inferred.
			 * Advancing the reference without measuring it is the
			 * one direction the gate must not fail in -- the next
			 * frame would compare against something no box was ever
			 * derived from -- so throw the reference away and let
			 * the next frame report "all of it".
			 */
			bc_preproc_gate_reset(p);
			bc_preproc_gpu_release(p);
			return 0;
		}
		/*
		 * Nothing was mapped, so the buffer is still GL's and the
		 * readback below is still a legal thing to do to it.
		 */
		bc_preproc_gate_reset(p);
	}
	if (bc_preproc_readback(p, host) == 0 && bc_detect_submit(d, host)) {
		bc_preproc_commit(p);
		if (host_filled)
			*host_filled = 1;
		return 1;
	}
	return 0;
}

/*
 * Run the hold against the frame just submitted, then keep that frame as the
 * next comparison. Both halves have to happen together: a reference that was
 * never compared, or compared against a frame nobody inferred, answers a
 * different question than the one the hold asks.
 */
static void hold_after_submit(struct bc_tracker *tr, const struct bc_preproc *p,
                              const float *cur, float *prev, int *have_prev,
                              int host_filled, double now, struct stat *mad,
                              unsigned long *released, unsigned long *capped);

/*
 * The occlusion hold: hold every track whose region of the screen has not
 * changed since the last frame we inferred.
 *
 * Why this is not the change gate: the gate asks about the *whole* frame, and
 * on a compositor that cannot hide the overlay the whole frame is never still
 * -- our own bars are in it, moving with the boxes. The question that matters
 * is per box, and it is the one in hold.h: did the picture under this cover
 * change? If it did not, the detector was looking at our bar and its silence
 * about that region is not evidence the content left.
 *
 * Known limitation, worth writing down because it looks like a bug from the
 * outside: the whole box is measured, interior included, whatever effect is
 * drawn over it. That is right for an effect derived from the content --
 * pixelate, blur, glitch -- where the interior is the *best* signal and is
 * where a change shows first. It is wrong for a solid bar, which overwrites
 * its region with a constant: the interior then contributes a guaranteed zero,
 * which drags the difference down and makes a box under a solid bar stickier
 * than it should be. Measuring only a ring around the box would fix that, and
 * nothing here does it yet.
 *
 * Whoever does it should note that the choice is per frame, not per box, and
 * so has to be an AND over every effect that can be drawn -- global, per-class
 * pin, takeover fill, reverse mask, test box. That is the opposite of the
 * reflex, and the asymmetry is the point: one opaque cover wrongly counted as
 * readable pins its own difference at zero and drags the whole frame's verdict
 * with it, while a readable one wrongly excluded costs only the ring's
 * slightly weaker signal.
 *
 * Only for a 1x1 grid. With a tile grid the tensor holds several letterboxes
 * and tile 0's transform is not the screen's, so a box would be compared
 * against the wrong pixels -- and a wrong comparison here reads as "unchanged"
 * about as often as not, which would hold boxes over content that had gone.
 * Sectioned runs simply keep the old detection-driven behaviour.
 */
static void occlusion_hold(struct bc_tracker *tr, const struct bc_preproc *p,
                           const float *cur, const float *prev,
                           double now, double ceiling_ms,
                           struct stat *mad_stat,
                           unsigned long *released, unsigned long *capped)
{
	const int nw = bc_preproc_net_w(p), nh = bc_preproc_net_h(p);
	if (nw <= 0 || nh <= 0 || bc_preproc_tiles(p) != 1)
		return;

	for (int i = 0; i < BC_TRK_MAX; i++) {
		float bx, by, bw, bh;
		if (!bc_trk_box(&tr->t[i], &bx, &by, &bw, &bh))
			continue;

		float nx0, ny0, nx1, ny1;
		bc_preproc_to_net(p, bx, by, &nx0, &ny0);
		bc_preproc_to_net(p, bx + bw, by + bh, &nx1, &ny1);

		const float mad = bc_hold_region_mad(cur, prev, nw, nh, 3,
		                                     (int)nx0, (int)ny0,
		                                     (int)(nx1 + 0.5f),
		                                     (int)(ny1 + 0.5f));
		if (mad < 0.0f)
			continue;          /* nothing measurable; never holds */

		stat_add(mad_stat, mad);
		if (mad >= HOLD_MAD) {
			(*released)++;     /* the content moved: let it age */
			continue;
		}
		*capped += (unsigned long)bc_trk_hold(&tr->t[i], now, ceiling_ms);
	}
}

static void hold_after_submit(struct bc_tracker *tr, const struct bc_preproc *p,
                              const float *cur, float *prev, int *have_prev,
                              int host_filled, double now, struct stat *mad,
                              unsigned long *released, unsigned long *capped)
{
	if (!prev || !cur || !host_filled)
		return;
	if (*have_prev)
		occlusion_hold(tr, p, cur, prev, now, GATE_HOLD_MS,
		               mad, released, capped);
	memcpy(prev, cur, bc_preproc_total_floats(p) * sizeof *prev);
	*have_prev = 1;
}

/*
 * Re-take the hold's reference frame after the tensor it mirrors was resized.
 *
 * Two things resize it under a running censor -- a model swap and a section
 * grid change -- and the hold's copy has to follow both, because the memcpy
 * above writes a whole tensor into it without consulting its length. Left
 * behind, a buffer still sized for the old net is written past its end; on the
 * model path it has also already been freed, so the same memcpy lands in
 * released memory and occlusion_hold() reads it first.
 *
 * A NULL `prev` means the hold was never on -- no allocation, and nothing here
 * should turn it on. Failing to re-allocate is not fatal: the hold goes quiet
 * and censoring falls back to detection-driven, which is what a machine too
 * short of memory for it would have done at startup anyway.
 */
static void hold_resize(float **prev, int *have_prev, const struct bc_preproc *p)
{
	*have_prev = 0;
	if (!*prev)
		return;

	free(*prev);
	*prev = p ? malloc(bc_preproc_total_floats(p) * sizeof **prev) : NULL;
	if (!*prev)
		WARN("no memory for the occlusion hold after a resize -- "
		     "censoring is detection-driven, and may oscillate under "
		     "a captured overlay");
}

/*
 * Push the confidence policy from the config into the detector and tracker.
 *
 * One function because these are one decision spread over three numbers, and
 * they are read at four points -- startup, model swap, config reload, and the
 * tracker rebuild that follows a swap. They drifted apart once already when
 * the reload path set the tracker's thresholds and left the detector's alone.
 *
 * Call it *after* bc_tracker_init(), which resets the per-class overrides.
 */
static void apply_confidence(struct bc_detector *d, struct bc_tracker *tr,
                             const struct bc_config *c, int no_hyst)
{
	const float hyst = no_hyst ? 1.0f : HYST_RATIO;

	/*
	 * The detector runs at the floor of the *lowest* class being censored,
	 * not the global one. A class pinned below `confidence` is otherwise
	 * dropped inside the decoder, before the tracker it was meant to reach
	 * ever sees it -- so the override would work in the direction that
	 * costs coverage and do nothing in the direction that buys it, which is
	 * the wrong way round for every knob in this program.
	 */
	float lowest = c->confidence;
	for (int i = 0; i < bc_class_count(); i++) {
		if (!c->censor[i])
			continue;
		const float s = bc_cfg_class_conf(c, i);
		if (s < lowest)
			lowest = s;
	}
	if (d)
		bc_detect_set_conf(d, lowest * hyst);

	if (!tr)
		return;
	tr->high_score = c->confidence;
	/* The weak band must be the threshold the detector is actually run at,
	 * or the second association pass has nothing to work with. */
	tr->low_score  = c->confidence * hyst;
	for (int i = 0; i < BC_MAX_CLASSES; i++) {
		const float s = (i < bc_class_count()) ? c->conf_for[i] : -1.0f;
		tr->class_score[i] = (s > 0.0f && s < 1.0f) ? s : -1.0f;
	}
}

static int classes_similar(int a, int b)
{
	if (a == b)
		return 1;
	const char *na = bc_class_name(a), *nb = bc_class_name(b);
	for (unsigned i = 0; i < sizeof SIMILAR_PAIRS / sizeof SIMILAR_PAIRS[0]; i++) {
		const char *p = SIMILAR_PAIRS[i][0], *q = SIMILAR_PAIRS[i][1];
		if ((!strcasecmp(na, p) && !strcasecmp(nb, q)) ||
		    (!strcasecmp(na, q) && !strcasecmp(nb, p)))
			return 1;
	}
	return 0;
}


/*
 * A bar is symmetric under a half turn, so every angle has a representative in
 * (-pi/2, pi/2]. Keeping them there means easing never takes the long way round
 * a wrap that is not a real rotation.
 */
static float wrap_bar(float a)
{
	return bc_eyebar_wrap(a);
}

/*
 * Which body parts make a usable orientation reference, nearest the head first.
 * The closer the reference is, the less a bent spine or a turned hip can skew
 * the axis. Feet are deliberately absent: legs bend independently of the head,
 * so a seated subject would read as lying down.
 */
static int torso_rank(const char *n)
{
	if (strstr(n, "BREAST"))    return 1;
	if (strstr(n, "ARMPITS"))   return 2;
	if (strstr(n, "BELLY"))     return 3;
	if (strstr(n, "GENITALIA")) return 4;
	if (strstr(n, "BUTTOCKS") || strstr(n, "ANUS")) return 5;
	/* booru_yolo's taxonomy, same ordering by distance from the head */
	if (!strcmp(n, "BOOB") || !strcmp(n, "BUST") || !strcmp(n, "SIDEB"))
		return 1;
	if (!strcmp(n, "SHLD"))     return 2;
	if (!strcmp(n, "HIP") || !strcmp(n, "NOPAN")) return 4;
	if (!strcmp(n, "BUTT") || !strcmp(n, "ASS"))  return 5;
	return 0;
}

/*
 * Is this class a head? The eye bar and the body axis both need to know, and
 * "contains FACE" stopped being enough once models arrived that call it HEAD --
 * booru_yolo names human and anthro heads separately, and an anthro head is a
 * face for this purpose. Matched exactly rather than by prefix: HIP would
 * otherwise read as a head class.
 */
static int is_head_class(const char *n)
{
	return bc_is_head_class(n);
}

/*
 * Orientation from body geometry.
 *
 * A YOLO detection head gives axis-aligned boxes and no angle, so a face box
 * alone cannot say which way is up -- an inverted face has exactly the shape of
 * an upright one. The other detections on the same person can: a head sits at
 * one end of a torso, so the vector from a torso box to the face is the body's
 * up-axis. A subject lying on their side puts that vector horizontal, and the
 * eye bar turns with it.
 *
 * This reads geometry only, so it uses detections regardless of whether their
 * class is censored -- with faces-only enabled the torso boxes are still there.
 *
 * Writes a unit up-vector in screen coordinates (y down) and returns 1, or
 * returns 0 when nothing near enough was plausibly this person's.
 */
static int body_axis(const struct bc_det *dets, int n, int self,
                     struct bc_preproc *pp, float fcx, float fcy, float fh,
                     float *ux, float *uy)
{
	int   best_rank = 99;
	float near2 = (fh * AXIS_NEAR) * (fh * AXIS_NEAR);
	float far2  = (fh * AXIS_FAR)  * (fh * AXIS_FAR);
	float cx[BC_MAX_DET], cy[BC_MAX_DET];
	int   rk[BC_MAX_DET], n_cand = 0;

	for (int j = 0; j < n; j++) {
		if (j == self)
			continue;
		int rank = torso_rank(bc_class_name(dets[j].cls));
		if (!rank)
			continue;
		float ax, ay, bx, by;
		bc_preproc_to_screen(pp, dets[j].x, dets[j].y, &ax, &ay);
		bc_preproc_to_screen(pp, dets[j].x + dets[j].w,
		                     dets[j].y + dets[j].h, &bx, &by);
		float mx = (ax + bx) * 0.5f, my = (ay + by) * 0.5f;
		float dx = mx - fcx, dy = my - fcy;
		float d2 = dx * dx + dy * dy;
		/* Too close and the direction is noise; too far and it belongs
		 * to somebody else in frame. */
		if (d2 < near2 || d2 > far2)
			continue;
		cx[n_cand] = mx; cy[n_cand] = my; rk[n_cand] = rank;
		n_cand++;
		if (rank < best_rank)
			best_rank = rank;
	}
	if (best_rank == 99)
		return 0;

	/*
	 * Average every reference of the best rank rather than taking the
	 * nearest one. A single breast box sits off to one side of the body,
	 * so on its own it tilts the axis even for an upright subject; the two
	 * together average back onto the centreline. Mixing ranks would not
	 * help -- parts at different heights are supposed to differ.
	 */
	float sx = 0.0f, sy = 0.0f;
	int   used = 0;
	for (int j = 0; j < n_cand; j++) {
		if (rk[j] != best_rank)
			continue;
		sx += cx[j]; sy += cy[j];
		used++;
	}
	float bdx = sx / used - fcx;
	float bdy = sy / used - fcy;
	float len = sqrtf(bdx * bdx + bdy * bdy);
	if (len < fh * AXIS_NEAR)
		return 0;   /* averaging collapsed it onto the face */
	/* face->torso points down the body, so up is its negation */
	*ux = -bdx / len;
	*uy = -bdy / len;
	return 1;
}

/*
 * How long a remembered body axis stays usable, and how far from its own face
 * it may be reused.
 *
 * The duration is Hotscreen's lost-track window (HOTSCREEN.md [5]) for the same
 * reason: it is long enough to bridge a torso that drops out for a few
 * inferences, short enough that a subject who has actually turned or left is
 * not still being covered on last frame's geometry.
 */
#define AXIS_MEM_MS   500.0
#define AXIS_MEM_NEAR 1.5f    /* x face height, from the remembered face centre */

/*
 * The axis this face had recently, when this frame has no second detection to
 * derive one from. Matched on position rather than identity, because the bar is
 * computed before association runs and there is no track id yet.
 *
 * Nearest wins, which is what keeps two people in frame from trading axes: a
 * face is far closer to where it was 100 ms ago than to anybody else.
 */
static int remembered_axis(const struct track *meta, int n_meta,
                           float fcx, float fcy, float fh, double now,
                           float *ux, float *uy)
{
	float near2 = (fh * AXIS_MEM_NEAR) * (fh * AXIS_MEM_NEAR);
	float best2 = near2;
	int   best  = -1;

	for (int k = 0; k < n_meta; k++) {
		if (!meta[k].eyebar || !meta[k].have_axis)
			continue;
		if (now - meta[k].axis_at > AXIS_MEM_MS)
			continue;
		float mx = meta[k].fx + meta[k].fw * 0.5f;
		float my = meta[k].fy + meta[k].fh * 0.5f;
		float dx = mx - fcx, dy = my - fcy;
		float d2 = dx * dx + dy * dy;
		if (d2 <= best2) {
			best2 = d2;
			best  = k;
		}
	}
	if (best < 0)
		return 0;
	*ux = meta[best].ux;
	*uy = meta[best].uy;
	return 1;
}

/*
 * Floor on the gap between segmentation runs, in ms.
 *
 * Left to fill whatever the worker had spare it ran at 63/s against a primary
 * detector managing 1.2/s -- about half the GPU spent refining the *shape* of
 * covers whose position was being updated fifty times less often.
 *
 * The ceiling on how slow this may go is staleness, not cost: the silhouette
 * clips a cover that follows a tracked box, so if the box moves and the mask
 * does not, the clip cuts into content the box still covers. At 30/s a subject
 * crossing the screen at 500 px/s moves ~16 px between silhouettes, which the
 * default 2% margin (~28 px here) already absorbs. That is why the margin and
 * this interval have to be read together.
 */
#define SEG_MIN_MS 33.0

/*
 * Grow a silhouette outward by `r` pixels, separably: a max filter along each
 * axis in turn, which for a binary mask is a square dilation.
 *
 * This is not a nicety. Clipping a cover to a mask is the one thing here that
 * *removes* covered area, and a silhouette that traces the subject exactly
 * leaves a rim of them on the wrong side of the line -- the model's edge is
 * approximate, and it is approximate in both directions. Growing it first means
 * the error has to exceed the margin before any of it becomes exposed skin.
 *
 * `tmp` is scratch of the same size as `src`.
 */
static void dilate_mask(const unsigned char *src, unsigned char *dst,
                        unsigned char *tmp, int w, int h, int r)
{
	if (r <= 0) {
		memcpy(dst, src, (size_t)w * h);
		return;
	}

	for (int y = 0; y < h; y++) {
		const unsigned char *row = src + (size_t)y * w;
		unsigned char *out = tmp + (size_t)y * w;
		for (int x = 0; x < w; x++) {
			int lo = x - r, hi = x + r;
			if (lo < 0) lo = 0;
			if (hi >= w) hi = w - 1;
			unsigned char v = 0;
			for (int i = lo; i <= hi && !v; i++)
				v = row[i];
			out[x] = v;
		}
	}
	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++) {
			int lo = y - r, hi = y + r;
			if (lo < 0) lo = 0;
			if (hi >= h) hi = h - 1;
			unsigned char v = 0;
			for (int i = lo; i <= hi && !v; i++)
				v = tmp[(size_t)i * w + x];
			dst[(size_t)y * w + x] = v;
		}
	}
}

/* --- hand-drawn outlines --------------------------------------------------
 *
 * A freehand outline is covered by drawing its *bounding box* with the usual
 * effect and clipping that box to a mask rasterised from the points -- the
 * same mask machinery the silhouette clip uses, pointed at a shape the user
 * drew instead of one a model guessed.
 *
 * The direction of every approximation here is outward. Simplification cuts
 * corners, rasterising quantises, and both can only be trusted to be within a
 * pixel or two -- so the mask is dilated before it clips anything, and any
 * failure to build it at all leaves the box unclipped, i.e. covering the whole
 * bounding box. Nothing in this path can end in less coverage than intended.
 */

/* Samples in one stroke. One per frame, so this is about six seconds of drag. */
#define BC_PATH_MAX 1024
/*
 * Mask resolution on the long side.
 *
 * This is the quantum for everything drawn through the mask -- the traced
 * edge, the outline band, and the width of the line while it is being drawn.
 * At 1024 across a 2560-wide screen one mask pixel is 2.5 screen pixels, which
 * made the drawing line noticeably fatter than the thing being traced round.
 */
#define BC_MASK_LONG 2048
/* Mask pixels the finished shape is grown by, per the note above. */
#define BC_MASK_GROW 2

/*
 * Even-odd scanline fill of a closed polygon.
 *
 * The ring is implied: the segment from the last point back to the first is
 * included, so an unclosed point list still fills as though it had been
 * closed. That is deliberate -- an outline that failed to close is refused
 * long before it reaches here, and if one ever did, filling it is the
 * covering answer and leaving it hollow is not.
 */
static void fill_poly(unsigned char *m, int mw, int mh,
                      const float *xs, const float *ys, int n)
{
	if (n < 3)
		return;

	float cross[BC_PATH_MAX];

	for (int y = 0; y < mh; y++) {
		const float sy = ((float)y + 0.5f) / (float)mh;
		int nc = 0;

		for (int i = 0; i < n && nc < BC_PATH_MAX; i++) {
			const int j = (i + 1) % n;
			float y0 = ys[i], y1 = ys[j];
			float x0 = xs[i], x1 = xs[j];
			/*
			 * Half-open in y: a vertex exactly on the scanline is
			 * counted by the edge below it and not the one above,
			 * which is what stops a vertex producing two crossings
			 * and inverting the rest of the row.
			 */
			if ((y0 <= sy && y1 > sy) || (y1 <= sy && y0 > sy)) {
				const float t = (sy - y0) / (y1 - y0);
				cross[nc++] = x0 + t * (x1 - x0);
			}
		}
		if (nc < 2)
			continue;

		for (int a = 1; a < nc; a++) {
			const float v = cross[a];
			int b = a - 1;
			while (b >= 0 && cross[b] > v) {
				cross[b + 1] = cross[b];
				b--;
			}
			cross[b + 1] = v;
		}

		unsigned char *row = m + (size_t)y * mw;
		for (int a = 0; a + 1 < nc; a += 2) {
			int lo = (int)(cross[a] * (float)mw);
			int hi = (int)(cross[a + 1] * (float)mw + 0.5f);
			if (hi < 0 || lo >= mw)
				continue;
			if (lo < 0) lo = 0;
			if (hi > mw) hi = mw;
			for (int x = lo; x < hi; x++)
				row[x] = 255;
		}
	}
}

/*
 * Stamp a path as a thick line, for previewing a stroke that is not closed yet
 * and so has no inside to fill.
 */
static void stroke_path(unsigned char *m, int mw, int mh,
                        const float *xs, const float *ys, int n, int r)
{
	if (n < 1)
		return;
	if (r < 1)
		r = 1;

	for (int i = 0; i < n; i++) {
		const int j = i + 1 < n ? i + 1 : i;
		const float ax = xs[i] * (float)mw, ay = ys[i] * (float)mh;
		const float bx = xs[j] * (float)mw, by = ys[j] * (float)mh;
		const float dx = bx - ax, dy = by - ay;
		float len = sqrtf(dx * dx + dy * dy);
		int steps = (int)len + 1;
		if (steps > 4096)
			steps = 4096;
		for (int s = 0; s <= steps; s++) {
			const float t = steps ? (float)s / (float)steps : 0.0f;
			const int cx = (int)(ax + dx * t);
			const int cy = (int)(ay + dy * t);
			for (int y = cy - r; y <= cy + r; y++) {
				if (y < 0 || y >= mh)
					continue;
				unsigned char *row = m + (size_t)y * mw;
				for (int x = cx - r; x <= cx + r; x++)
					if (x >= 0 && x < mw)
						row[x] = 255;
			}
		}
	}
}

/*
 * Ramer-Douglas-Peucker, iteratively.
 *
 * A stroke arrives with one sample per frame and almost all of it is hand
 * tremor; the shape is in a few dozen of those points. Written with an
 * explicit stack rather than recursively because the input is up to
 * BC_PATH_MAX long and the worst case for the recursion is one frame per
 * point.
 *
 * Returns the number of points kept, writing them in order.
 */
static int dp_simplify(const float *xs, const float *ys, int n, float eps,
                       float *ox, float *oy, int max)
{
	if (n <= 2 || max < 2) {
		int k = n < max ? n : max;
		for (int i = 0; i < k; i++) { ox[i] = xs[i]; oy[i] = ys[i]; }
		return k;
	}

	unsigned char keep[BC_PATH_MAX];
	memset(keep, 0, (size_t)n);
	keep[0] = keep[n - 1] = 1;

	int stack[BC_PATH_MAX][2];
	int sp = 0;
	stack[sp][0] = 0;
	stack[sp][1] = n - 1;
	sp++;

	while (sp > 0) {
		sp--;
		const int a = stack[sp][0], b = stack[sp][1];
		if (b <= a + 1)
			continue;

		const float ax = xs[a], ay = ys[a];
		const float bx = xs[b], by = ys[b];
		const float dx = bx - ax, dy = by - ay;
		const float len = sqrtf(dx * dx + dy * dy);

		float worst = -1.0f;
		int   at = -1;
		for (int i = a + 1; i < b; i++) {
			float d;
			if (len < 1e-6f) {
				const float ex = xs[i] - ax, ey = ys[i] - ay;
				d = sqrtf(ex * ex + ey * ey);
			} else {
				d = fabsf((xs[i] - ax) * dy - (ys[i] - ay) * dx) / len;
			}
			if (d > worst) { worst = d; at = i; }
		}
		if (worst > eps && at > a && at < b) {
			keep[at] = 1;
			if (sp + 2 <= BC_PATH_MAX) {
				stack[sp][0] = a; stack[sp][1] = at; sp++;
				stack[sp][0] = at; stack[sp][1] = b; sp++;
			}
		}
	}

	int k = 0;
	for (int i = 0; i < n && k < max; i++)
		if (keep[i]) { ox[k] = xs[i]; oy[k] = ys[i]; k++; }
	return k;
}

/*
 * Simplify until the result fits, raising the tolerance until it does.
 *
 * A single tolerance cannot serve both a shape drawn in one quick sweep and
 * one traced slowly around something detailed: too tight and the point list
 * overflows, too loose and the shape is a triangle. Doubling from something
 * small converges in a handful of passes and lands just inside the limit.
 */
static int simplify_to_fit(const float *xs, const float *ys, int n,
                           float *ox, float *oy, int max)
{
	float eps = 0.0015f;   /* fraction of the screen; ~2 px at 1440p */
	for (int pass = 0; pass < 12; pass++) {
		const int k = dp_simplify(xs, ys, n, eps, ox, oy, BC_MAX_POLY_PTS);
		if (k <= max)
			return k;
		eps *= 2.0f;
	}
	return dp_simplify(xs, ys, n, eps, ox, oy, max);
}

/*
 * A cheap fingerprint of the outline set, so the mask is rasterised when it
 * changes and not once a frame. Filling a megabyte of mask every frame for a
 * shape that has not moved since it was drawn would be the most expensive
 * thing in the loop that does nothing.
 */
static unsigned long regions_stamp(const struct bc_config *c)
{
	unsigned long h = 1469598103934665603UL;
	for (int i = 0; i < c->n_regions; i++) {
		const struct bc_region *r = &c->regions[i];
		if (r->n_pts < 3)
			continue;
		const unsigned char *p = (const unsigned char *)r->px;
		const unsigned char *q = (const unsigned char *)r->py;
		for (size_t b = 0; b < sizeof(float) * (size_t)r->n_pts; b++) {
			h = (h ^ p[b]) * 1099511628211UL;
			h = (h ^ q[b]) * 1099511628211UL;
		}
	}
	return h;
}

/*
 * Config effect -> overlay effect. The two enums do not share a numbering, so
 * this maps and never casts. Was inline at the one place a frame needed it;
 * it is a function now because per-class effects need it once per box.
 */
static int overlay_effect(int cfg_effect)
{
	switch (cfg_effect) {
	case BC_CFG_FILL:     return BC_EFFECT_FILL;
	case BC_CFG_BLUR:     return BC_EFFECT_BLUR;
	case BC_CFG_NOISE:    return BC_EFFECT_NOISE;
	case BC_CFG_GLITCH:   return BC_EFFECT_GLITCH;
	case BC_CFG_RGBSHIFT: return BC_EFFECT_RGBSHIFT;
	case BC_CFG_TAPE:     return BC_EFFECT_TAPE;
	case BC_CFG_HALFTONE: return BC_EFFECT_HALFTONE;
	case BC_CFG_SOBEL:    return BC_EFFECT_SOBEL;
	case BC_CFG_TRIANGLE: return BC_EFFECT_TRIANGLE;
	case BC_CFG_CLUSTER:  return BC_EFFECT_CLUSTER;
	case BC_CFG_SPLATTER: return BC_EFFECT_SPLATTER;
	case BC_CFG_SHAPE:    return BC_EFFECT_SHAPE;
	case BC_CFG_CAPTION:  return BC_EFFECT_CAPTION;
	case BC_CFG_WALL:     return BC_EFFECT_WALL;
	default:              return BC_EFFECT_PIXELATE;
	}
}

/*
 * Any usable detector in vendor/, for when the configured one is not there.
 *
 * "Usable" excludes the auxiliary heads by name. Those are loaded deliberately
 * through their own config keys (seg_model, landmark_model) or through
 * BC_EYE_MODEL/BC_POSE_MODEL, and none of them is a nudity detector: picking
 * one here would produce a run that loads, reports classes, and covers nothing
 * -- the silent-failure shape this program keeps having to design against.
 *
 * Alphabetical order, because readdir order is filesystem-dependent and a
 * fallback that picks a different model on two machines with the same files is
 * a support problem rather than a convenience.
 */
static const char *pick_any_model(char *buf, size_t n)
{
	static const char *skip[] = { "seg", "landmark", "pose", "eyes", NULL };

	DIR *d = opendir("vendor");
	if (!d)
		return NULL;

	char best[1024] = { 0 };
	struct dirent *e;
	while ((e = readdir(d))) {
		const char *dot = strrchr(e->d_name, '.');
		if (!dot || strcasecmp(dot, ".onnx") != 0)
			continue;

		int aux = 0;
		for (int i = 0; skip[i] && !aux; i++)
			if (strcasestr(e->d_name, skip[i]))
				aux = 1;
		if (aux)
			continue;

		if (!best[0] || strcmp(e->d_name, best) < 0)
			snprintf(best, sizeof best, "%s", e->d_name);
	}
	closedir(d);

	if (!best[0])
		return NULL;
	snprintf(buf, n, "vendor/%s", best);
	return buf;
}

/*
 * The live loop's state. Everything main() used to keep in locals that the
 * loop shares across frames, or across the stages of one frame, is here, so
 * each stage can be a function of its own (the step_*() below). A variable a
 * single stage uses within one frame stays a local of that stage.
 *
 * Static storage, so everything starts at zero; main() sets the rest, where it
 * used to declare them.
 */
struct bc_live {
	double secs;
	int net_w;
	int net_h;
	int net_req_h;                         /* the height last asked of the model, which a fixed one ignores */
	struct bc_overlay *ov;
	struct bc_gpu *gpu;
	struct bc_capture *cap;
	struct bc_preproc *pp;
	struct bc_detector *det;
	struct bc_detector *eye_det;
	struct bc_preproc *eye_pp;
	float *eye_tensor;
	/* Most recent eye boxes, in screen pixels, with the time they landed. */
	struct { float x0, y0, x1, y1; } eyes[BC_MAX_DET];
	int n_eyes;
	double eyes_at;
	/*
	 * Segmentation, for clipping covers to the subject's outline. A third
	 * model on its own preprocessor, full-frame at whatever input it wants,
	 * and entirely optional -- absent, every cover is the rectangle it has
	 * always been.
	 */
	struct bc_detector *seg_det;
	struct bc_preproc *seg_pp;
	float *seg_tensor;
	unsigned char *seg_mask;                /* as published by the detector */
	unsigned char *seg_dil;                 /* the same, grown by seg_margin */
	int seg_have;                           /* a silhouette has arrived */
	int seg_shadowed;                       /* outlines took the mask; warned once */
	/*
	 * The rasterised outline mask, and the scratch its dilation needs.
	 * Allocated on first use: most runs never draw anything, and this is a
	 * megabyte that would otherwise be reserved for all of them. Declared
	 * up here with the other buffers because the teardown frees it, and
	 * every early exit between here and the loop goes through that.
	 */
	unsigned char *lasso_mask;
	unsigned char *lasso_tmp;
	unsigned char *lasso_dil;
	unsigned char *lasso_band;              /* the shape grown by the outline width */
	int lasso_mw;
	int lasso_mh;
	int lasso_have;                         /* the mask holds outlines right now */
	int lasso_uploaded;
	unsigned char *lasso_base;              /* committed shapes, dilated, cached */
	/*
	 * How much of the bounding box the outline mask actually removes.
	 *
	 * Reported because a clip that turns out to be geometrically inert is
	 * not visible from the outside: it runs, it costs, and it changes
	 * nothing. That is exactly how the silhouette clip failed, and it went
	 * unnoticed until the kept-area fraction was measured and came back
	 * 100.0%. A traced outline that reports 100% is a rectangle.
	 */
	double lasso_kept_pct;
	int lasso_measured;
	unsigned long seg_runs;
	double seg_cov_sum;
	double seg_last;
	/*
	 * How much of a cover the outline actually keeps. The silhouette being
	 * built says nothing about whether it *does* anything: a body-part box
	 * usually sits well inside a person, and a mask can only trim where the
	 * two disagree. Near 100% means the clip is a no-op and the feature is
	 * invisible however correct it is.
	 */
	double seg_keep_sum;
	unsigned long seg_keep_n;
	unsigned long regions_drawn;
	struct bc_landmark *lm_det;
	struct bc_preproc *lm_pp;
	float *lm_tensor;
	struct bc_lm_face lm_out[BC_LM_MAX_FACES];
	float lm_rects[BC_LM_MAX_FACES * 4];
	int lm_face_det[BC_LM_MAX_FACES];
	int n_lm;
	unsigned long lm_runs;
	unsigned long lm_used;
	unsigned long lm_weak;
	double lm_ms_sum;
	int pose_mode;
	struct { float x0, y0, x1, y1; float ex[2], ey[2], es[2]; } pfaces[BC_MAX_DET];
	int n_pfaces;
	double pfaces_at;
	float *tensor;
	int have_prev;
	/*
	 * Up here with `tensor` rather than beside the code that fills it,
	 * because the cleanup at `out:` frees it and every `goto out` above
	 * jumps over this point -- a declaration left further down would be
	 * uninitialised on exactly those paths. See the occlusion hold below
	 * for what it is for.
	 */
	float *prev_tensor;
	int hold_have_prev;
	/*
	 * Config lives next to the binary's working directory by default rather
	 * than in ~/.config: this is still a development tree, and keeping the
	 * file inside the project means a stray write cannot land outside it.
	 * BC_CONFIG points elsewhere once that stops being true.
	 */
	struct bc_config cfg;
	int no_hyst;
	int no_smooth;
	float alpha_env;
	/*
	 * Geometry and lifetime live in the tracker (src/track.c). What stays
	 * here is the part it has no business knowing: an eye bar's angle, the
	 * face it is clamped inside, and the eased box we actually draw. Indexed
	 * by track slot, so the two stay in step.
	 */
	struct bc_tracker tracker;
	struct track meta[BC_TRK_MAX];
	struct bc_box boxes[BC_MAX_BOXES];
	struct bc_det dets[BC_MAX_DET];
	/*
	 * Screen delay (cfg.delay_ms, DESIGN: delayed screen). The capture is kept
	 * in a GPU ring and shown `delay` late, so each frame can be covered from
	 * detections on both sides of it; `dh` is the record of what the censor
	 * had decided to cover each time it looked. `show` is the failsafe's
	 * verdict for this frame: only an ordinary censoring frame is delayed.
	 */
	struct bc_framering *ring;
	struct bc_dh *dh;
	float  delay_applied;                  /* ms the ring is sized for; 0 = off */
	double ring_interval;                  /* ms between frames kept */
	double ring_last_push;
	unsigned long ring_seq;
	double submit_t;                       /* capture time of the frame in flight */
	int    snap_pending;
	double snap_t;
	int    shown_valid;
	unsigned long shown_seq;
	struct bc_box shown_boxes[BC_MAX_BOXES];
	int    shown_n;
	enum bc_fs_show show;
	unsigned long delay_shown, delay_held, delay_blank;
	double play_t, play_last, last_stall;  /* the display clock; see delay_prepare() */
	int    play_have;
	unsigned long delay_stalled, delay_distinct;
	unsigned long doubled_bar_frames;      /* frames with two eye bars on one spot */
	unsigned long delay_doubled;           /* the same, of the delayed frames shown */
	struct stat s_delay_lag, s_delay_gap;
	struct stat s_pre;
	struct stat s_present;
	struct stat s_infer;
	struct stat s_mad;
	unsigned long held_released;
	unsigned long held_capped;
	/*
	 * A hold cap has fired, so the next frame must be inferred even if the
	 * change gate would skip it. See the use site: without this, capping a
	 * track on a motionless screen is a fail-open deadlock.
	 */
	int force_infer;
	unsigned long gate_forced;
	unsigned long eyebars;
	unsigned long eyebar_fallback;
	unsigned long eyebar_axis;
	unsigned long eyebar_axis_mem;
	unsigned long eyebar_over;              /* bars drawn on top of a covered face */
	unsigned long eyebar_solo;              /* boxes that are themselves the band */
	unsigned long eyebar_kpt;
	unsigned long kpt_reject;
	unsigned long pose_matched;
	unsigned long pose_unmatched;
	int no_kpt;
	int debug_lm;
	float lm_bar_pad;
	int lm_nofloor;
	int lm_wide_crop;
	int axis_bar;
	float eyebar_top;
	float eyebar_bot;
	double axis_deg_sum;
	double axis_deg_max;
	unsigned long axis_over_30;
	/*
	 * Raw detections per class, counted before the censor policy is applied.
	 * Answers "is the model even reporting this class?", which no other
	 * counter here can: every downstream number is already filtered by what
	 * the user chose to censor, so a class that is enabled but never
	 * detected looks identical to one that is disabled.
	 */
	unsigned long det_by_class[BC_MAX_CLASSES];
	unsigned long submitted;
	unsigned long skipped;
	unsigned long presents;
	/* Log the capture->overlay mapping once, on the first frame that
	 * establishes it; it is the first thing to check when covers land in
	 * the wrong place. */
	int logged_draw_scale;
	const char *dump_path;
	int debug_all;
	int dumped;
	int no_hold;
	/*
	 * The hold reads the tensor on the host, so with a CUDA session -- which
	 * hands the detector a device pointer and never fills the host buffer --
	 * it has to ask for the readback it would otherwise skip. That is a
	 * deliberate exception to §0's "nothing crosses PCIe per frame", taken
	 * only where the alternative is a censor that oscillates: a compositor
	 * that cannot hide the overlay, on a machine using the GPU. On the CPU
	 * path the readback already happens and this costs nothing.
	 */
	int want_host;
	int host_filled;
	/*
	 * The peek: the only way to learn whether content under a cover is
	 * still there when the detector can see our cover.
	 *
	 * peek_start_ns marks the moment the screen was uncovered; a captured
	 * frame whose header timestamp is after it was composited with nothing
	 * of ours on screen, and only such a frame may be used as grounds to
	 * *drop* a cover. Everything else feeds tracking exactly as before.
	 *
	 * The screen stays uncovered until that frame arrives or PEEK_MAX_MS
	 * passes, whichever is first -- self-terminating, so a capture that
	 * stops delivering cannot leave the screen bare.
	 */
	int peeking;
	double peek_started;                    /* ms, our clock */
	int64_t peek_start_ns;                  /* CLOCK_MONOTONIC, the capture's clock */
	int clean_pending;                      /* a clean frame is captured, not yet inferred */
	int submitted_clean;                    /* ...and has been handed to the detector */
	unsigned long peeks_granted;
	unsigned long peeks_refused;
	unsigned long clean_frames;
	unsigned long clean_drops;
	/* Consecutive clean samples that failed to see each track. Two are
	 * required before a cover is allowed to go: one clean miss can be the
	 * detector blinking, and uncovering on a blink is the failure this
	 * program exists to prevent. */
	unsigned char clean_miss[BC_TRK_MAX];
	int peek_enabled;
	/*
	 * Area guard. Coverage and churn both *improve* when boxes grow to
	 * swallow the screen, so neither can tell "covering the right things"
	 * from "covering everything" -- which is how a defect that grew every
	 * cover without bound read as a 10x improvement. Area is the metric
	 * that separates them, and it is reported whether or not anything is
	 * wrong.
	 */
	struct stat s_cover_frac;               /* covered fraction of the screen */
	float max_box_growth;                   /* largest area / its area when first drawn */
	float first_area[BC_TRK_MAX];
	double last_clean_ms;
	int warned_blind;
	int trace;
	int no_gate;
	unsigned long fade_frames;
	int no_fade;
	/*
	 * Per-track churn. The global on/off transition counter is close to
	 * useless as a flicker metric: it only fires when the *total* box count
	 * crosses zero, so with several boxes on screen one of them blinking in
	 * and out registers as nothing at all. What a viewer actually perceives
	 * is individual boxes appearing and vanishing, and boxes jumping in
	 * size/position between detections -- so count those instead.
	 */
	unsigned long boxcount_changes;
	int last_n_boxes;
	/* Which detector is loaded now, so a config change can be seen. */
	char cur_model[512];
	unsigned long swaps;
	struct stat s_jitter;
	unsigned long gated;
	unsigned long quiet_holds;              /* ticks with no frame from a still screen */
	unsigned long scene_cuts;               /* frames where the picture was replaced */
	unsigned long joins_made;               /* boxes absorbed into another region */
	/*
	 * Reverse only: frames where the mask covered everything and revealed
	 * nothing. Not an error -- it is the safe direction for this mode --
	 * but it used to be the frame that flashed the screen black, so it is
	 * worth being able to see how often it happens.
	 */
	unsigned long mask_no_cutouts;
	/* Halves of a straddling detection rejoined across a tile seam. */
	unsigned long seam_merges;
	unsigned long eyes_under_bar;   /* EYE detections left to a face's bar */
	/* Frames whose tensor was never built because the worker was busy. */
	unsigned long readbacks_skipped;
	unsigned long takeovers;
	unsigned long pred_frames;
	double next_cfg_poll;
	unsigned long reloads;
	int panicked;
	int paused;
	unsigned long panic_frames;
	unsigned long paused_frames;
	/*
	 * Region drawing. `drawing` follows cfg.draw, which the GUI button sets
	 * and this loop clears -- so the file always ends up holding what
	 * actually happened, and the GUI can end the mode without needing the
	 * overlay to have keyboard focus. That last part is the important one:
	 * while drawing, this surface swallows every click on the output, so an
	 * exit that depends on us receiving input is an exit that does not exist
	 * on the day the compositor never focuses us.
	 *
	 * Two ways out, and they fail independently. Escape is the intended one
	 * and needs the compositor to have given us keyboard focus. The config
	 * flag does not -- the GUI is its own window on its own surface, so
	 * pressing the button there ends the mode whatever this surface is or
	 * is not receiving. There was a third, an idle timeout, removed at the
	 * user's request: it ended a trace that was merely being taken slowly,
	 * which is the normal way to use the feature rather than a fault.
	 */
	int drawing;
	int have_anchor;
	int anchor_x;
	int anchor_y;
	/*
	 * The freehand stroke being traced, in output pixels. Kept in pixels
	 * rather than normalised because every decision made about it -- has it
	 * come back to its start, did it travel far enough to be a trace rather
	 * than a click -- is a distance the user judged by eye on this screen.
	 */
	float path_x[BC_PATH_MAX];
	float path_y[BC_PATH_MAX];
	int n_path;
	int stroking;
	int path_closed;
	float path_len;
	float path_min_d;
	unsigned long lasso_stamp;              /* of the region set the mask was built from */
	int lasso_built;
	/*
	 * §6 fail-closed state. `capture_dead` latches: once the stream is gone
	 * it does not come back on its own, and the reconnect-with-restore-token
	 * path is not built yet, so the honest behaviour is to stay blanked and
	 * say so rather than pretend a recovery happened.
	 */
	int capture_dead;
	int failsafe;
	unsigned long failsafe_frames;
	unsigned long failsafe_trips;
	unsigned long det_errors;
	/*
	 * The latest outcome the detector handed back was a failure. Set and
	 * cleared only by bc_detect_collect() in step 3, so the gate in step 2
	 * sees it one iteration late -- at most one frame held over the failed
	 * one, never more, and never past a success.
	 */
	int det_failing;
	double last_frame;
	double last_infer;
	double max_frame_gap;
	double fault_stall;
	double fault_dead;
	double fault_nodet;
	int test_box[4];
	/* Flicker metric: how often the censored set goes empty <-> non-empty. */
	unsigned long transitions;
	unsigned long frames_with_boxes;
	int prev_had_boxes;
	double start;

	/* Set afresh every frame by one stage and read by a later one. */
	double t;
	int got;
	double tnow;
	int effect;
	float strength;
	int n_boxes;
	float frame_cover;
};

/*
 * The stages of one frame, in the order the loop in main() runs them. Each
 * reads and writes the shared state in struct bc_live; the two that can end
 * the frame early say so with their return value.
 */
enum bc_next {
	BC_GO_ON = 0,     /* run the next stage */
	BC_NEXT_FRAME,    /* abandon the rest of this frame */
	BC_STOP,          /* end the run */
};

/* Act on any signal that arrived since the last frame. */
static void step_signals(struct bc_live *lv)
{
	if (sig_panic) {
		sig_panic = 0;
		lv->panicked = !lv->panicked;
		LOG("PANIC %s", lv->panicked ? "ON -- output blanked"
		                         : "off");
	}
	if (sig_pause) {
		sig_pause = 0;
		lv->paused = !lv->paused;
		LOG("censoring %s", lv->paused ? "PAUSED" : "resumed");
	}
}

/* Pick up edits made by the GUI (or by hand) while running. */
static void step_config(struct bc_live *lv)
{
	if (now_ms() >= lv->next_cfg_poll) {
		lv->next_cfg_poll = now_ms() + CONFIG_POLL_MS;
		if (bc_config_poll(&lv->cfg)) {
			cfg_env_overrides(&lv->cfg);
			/*
			 * Model swap. Not a reload: the taxonomy, the
			 * class table, the tensor geometry and every
			 * class index in flight all change together, so
			 * the detector, the preprocessor and the tensors
			 * are rebuilt and the tracks dropped -- a track
			 * holds a class index, and index 3 means
			 * something different under the new model.
			 *
			 * Blank while it happens. Building an ORT
			 * session takes long enough to be visible, there
			 * is no detector to ask during it, and §6 says an
			 * unknown screen is covered, not shown. A second
			 * of black is the honest rendering of "I cannot
			 * see right now"; carrying the old boxes forward
			 * would be a guess about content we are not
			 * looking at.
			 */
			/* A change of tiling can change the shape the network wants
			 * (split tiles are square), which is rebuilt exactly as a
			 * model swap is: the input size is fixed when the session
			 * is made. */
			int net_wanted_w = lv->net_w, net_wanted_h = lv->net_req_h;
			want_net(lv->cfg.model, lv->cfg.section_split,
			         &net_wanted_w, &net_wanted_h);
			if (lv->cfg.model[0] && (strcmp(lv->cfg.model, lv->cur_model) ||
			                         net_wanted_h != lv->net_req_h)) {
				LOG("model: swapping to %s", lv->cfg.model);
				int nb = 0;
				/* Reverse would read the single
				 * full-screen box as one enormous
				 * cutout and reveal the whole display,
				 * exactly as it does on the panic
				 * path. */
				bc_overlay_set_reverse(lv->ov, 0);
				blank_all(lv->ov, lv->boxes, &nb);
				bc_overlay_present(lv->ov, lv->boxes, nb);

				/*
				 * The same filename guess as at startup.
				 * Without it a swap asked for the *old*
				 * model's size, which a fixed-input model
				 * ignores but a dynamic one keeps: going from
				 * 320n to a dynamic 640-trained model ran
				 * the new one at 320x192.
				 */
				int req_w = lv->net_w, req_h = lv->net_h;
				want_net(lv->cfg.model, lv->cfg.section_split,
				         &req_w, &req_h);
				lv->net_req_h = req_h;
				struct bc_detector *nd =
					bc_detect_create(lv->cfg.model, req_w,
					                 req_h, 4);
				if (!nd) {
					/* Keep running on the old model:
					 * a typo in the file must not
					 * take the censor down. */
					WARN("model: %s failed to load -- "
					     "staying on %s",
					     lv->cfg.model, lv->cur_model);
					snprintf(lv->cfg.model, sizeof lv->cfg.model,
					         "%s", lv->cur_model);
				} else {
					bc_detect_destroy(lv->det);
					lv->det = nd;
					lv->net_w = bc_detect_net_w(lv->det);
					lv->net_h = bc_detect_net_h(lv->det);

					bc_preproc_destroy(lv->pp);
					lv->pp = bc_preproc_create(lv->net_w, lv->net_h);
					apply_sections(lv->pp, lv->det, &lv->cfg, bc_overlay_width(lv->ov),
					               bc_overlay_height(lv->ov));
					free(lv->tensor);
					lv->tensor = lv->pp ? malloc(bc_preproc_total_floats(lv->pp)
					                     * sizeof *lv->tensor) : NULL;
					if (!lv->pp || !lv->tensor) {
						ERR("model: out of memory "
						    "rebuilding the pipeline");
						running = 0;
					}
					lv->have_prev = 0;
					hold_resize(&lv->prev_tensor,
					            &lv->hold_have_prev, lv->pp);

					/* A track holds a class index, and
					 * index 3 means something else
					 * under the new taxonomy. */
					bc_tracker_init(&lv->tracker, lv->cfg.confidence,
					                track_iou_thresh(),
					                lv->cfg.sticky_ms);
					lv->tracker.same_class = classes_similar;
					/* The new model has its own
					 * taxonomy, so the per-class
					 * floors have to be re-resolved
					 * against it. */
					apply_confidence(lv->det, &lv->tracker,
					                 &lv->cfg, lv->no_hyst);
					memset(lv->meta, 0, sizeof lv->meta);
					memset(lv->det_by_class, 0,
					       sizeof lv->det_by_class);
					lv->last_n_boxes = -1;
					lv->swaps++;

					/* Same reason as at startup: the
					 * file's per-class flags were
					 * resolved against the previous
					 * table and are now indexed
					 * wrong. Unconditional -- what
					 * changed is the names, and two
					 * taxonomies of the same size
					 * look identical by count. This
					 * reload is also what rewrites
					 * the file's class list, which is
					 * the only way the GUI learns the
					 * swap happened. */
					LOG("config: re-reading against "
					    "the model's %d-class taxonomy",
					    bc_class_count());
					/* Copy the path out first: load
					 * resets the struct this would
					 * point into, so passing
					 * cfg.path reads it back
					 * empty. */
					char cpath[sizeof lv->cfg.path];
					snprintf(cpath, sizeof cpath,
					         "%s", lv->cfg.path);
					bc_config_load(&lv->cfg, cpath);
					cfg_env_overrides(&lv->cfg);
					snprintf(lv->cur_model, sizeof lv->cur_model,
					         "%s", lv->cfg.model);
					LOG("model: now %s (%dx%d, %d classes)",
					    lv->cur_model, lv->net_w, lv->net_h,
					    bc_class_count());
					/* The swap is not a detector
					 * stall; do not let the watchdog
					 * read it as one. */
					lv->last_infer = now_ms();
				}
			}
			apply_confidence(lv->det, &lv->tracker, &lv->cfg, lv->no_hyst);

			/*
			 * The grid may have been changed in the GUI.
			 * Changing it changes how many tensors a frame
			 * produces, so the staging buffer has to grow or
			 * shrink with it. The gate's own previous frame
			 * lives on the GPU and is dropped by the grid
			 * change itself.
			 */
			{
				const size_t was = bc_preproc_total_floats(lv->pp);
				apply_sections(lv->pp, lv->det, &lv->cfg, bc_overlay_width(lv->ov),
					               bc_overlay_height(lv->ov));
				const size_t now = bc_preproc_total_floats(lv->pp);
				if (now != was) {
					float *nt = malloc(now * sizeof *nt);
					if (nt) {
						free(lv->tensor);
						lv->tensor = nt;
						lv->have_prev = 0;
						/* The hold mirrors the
						 * tensor whole, so its
						 * copy has to grow too. */
						hold_resize(&lv->prev_tensor,
						            &lv->hold_have_prev,
						            lv->pp);
					} else {
						ERR("sections: out of memory; "
						    "staying on the old grid");
						bc_preproc_set_grid(lv->pp,
						    lv->cfg.section_cols = 1,
						    lv->cfg.section_rows = 1,
						    lv->cfg.section_overlap,
						    lv->cfg.section_full_frame);
						bc_detect_set_tiles(lv->det,
						    bc_preproc_tiles(lv->pp));
					}
				}
			}

			/*
			 * Drop tracks whose class was just switched off.
			 *
			 * Un-ticking a class did not uncover anything: its
			 * boxes stayed until the model was reloaded. The
			 * class flag is only read where detections are
			 * turned into tracks, and an existing track does
			 * not need a *censored* detection to survive --
			 * any detection that matches it refreshes it (see
			 * crossclass_sustained below). That is deliberate,
			 * so a face detection can sustain an eye bar, but
			 * it also meant the now-unwanted class kept
			 * matching its own track and renewing it forever.
			 *
			 * Worst on booru_yolo, whose 26 classes overlap
			 * heavily by design -- HEAD with the per-species
			 * heads, BOOB with BUST and SIDEB, ASS with BUTT
			 * and HIP -- so almost every switched-off class
			 * had an overlapping detection holding it up.
			 *
			 * Only the track's *own* class is checked. Cross
			 * class sustaining stays exactly as it was.
			 */
			int dropped = 0;
			for (int k = 0; k < BC_TRK_MAX; k++)
				if (lv->tracker.t[k].state != BC_TRK_FREE &&
				    !lv->cfg.censor[lv->tracker.t[k].cls]) {
					lv->tracker.t[k].state = BC_TRK_FREE;
					dropped++;
				}
			/* The thresholds and the memory are config, so a reload
			 * has to reach them too -- they were set once at start.
			 * apply_confidence() above already moved the scores;
			 * the memory below is the part it does not own. */
			/*
			 * Not cfg.sticky_ms raw: the detection-gap floor
			 * below is what keeps a short sticky window from
			 * starving on slow inference, and assigning the
			 * config value here dropped that floor until the
			 * next detection arrived. A reload happens while
			 * the GUI is being used, which is exactly when a
			 * dropped floor would show as a flicker.
			 */
			if (lv->tracker.memory_ms < lv->cfg.sticky_ms)
				lv->tracker.memory_ms = lv->cfg.sticky_ms;
			if (dropped)
				LOG("config: dropped %d track(s) whose class "
				    "is no longer censored", dropped);

			/*
			 * And switching a class *on* must not wait for the
			 * screen to move. The change gate skips inference
			 * while the frame is identical, and the class flag
			 * is only consulted when fresh detections arrive,
			 * so on still content a newly ticked class covered
			 * nothing until something happened to move.
			 */
			lv->force_infer = 1;
			lv->reloads++;
			LOG("config: %s, effect=%s, conf=%.2f",
			    lv->cfg.enabled ? "enabled" : "DISABLED",
			    lv->cfg.effect == BC_CFG_PIXELATE ? "pixelate"
			                                  : "fill",
			    lv->cfg.confidence);
		}
	}
}

/* Region drawing mode: enter, follow the pointer, commit, leave. */
static void step_drawing(struct bc_live *lv)
{
	/*
	 * 0c. Region drawing.
	 *
	 * Driven from cfg.draw every frame rather than only on a reload,
	 * so that whatever writes the flag -- the GUI, a hand edit, or
	 * the commit below -- is answered the same way. The state on
	 * screen is then a function of the file, with no transition that
	 * only fires if it is noticed at the right moment.
	 */
	if (lv->cfg.draw && !lv->drawing) {
		bc_overlay_set_input(lv->ov, 1);
		lv->drawing       = 1;
		lv->have_anchor   = 0;
		lv->n_path        = 0;
		lv->stroking      = 0;
		LOG("drawing: on -- drag to trace an outline, or Enter "
		    "twice for a rectangle; Escape finishes");
	} else if (!lv->cfg.draw && lv->drawing) {
		bc_overlay_set_input(lv->ov, 0);
		lv->drawing     = 0;
		lv->have_anchor = 0;
		/*
		 * Write the flag back down, always -- not only when we
		 * were the one to clear it.
		 *
		 * Without this the file still says draw = true, and the
		 * next poll reads it back and re-enters the mode we
		 * just left. Escape would then release the pointer for
		 * a quarter of a second at a time, forever. Saving
		 * unconditionally costs one redundant rewrite when the
		 * GUI is what cleared it, and closes that loop.
		 */
		bc_config_save(&lv->cfg);
		bc_config_poll(&lv->cfg);
		cfg_env_overrides(&lv->cfg);
		LOG("drawing: off (%d region(s))", lv->cfg.n_regions);
	}

	if (lv->drawing) {
		/*
		 * Panic and pause both mean "show me the real screen",
		 * and a mode that is eating the pointer is not that.
		 * Leaving here rather than at the panic path below so
		 * input is handed back on the same frame the user asked
		 * for their screen back.
		 */
		if (lv->panicked || lv->paused) {
			LOG("drawing: cancelled by panic/pause");
			lv->cfg.draw = 0;
		}

		/*
		 * Freehand tracing.
		 *
		 * Sampled once per frame from the pointer rather than
		 * from every motion event. At display rate that is more
		 * than a hundred samples a second, which is far finer
		 * than the hand it is following, and it means the path
		 * cannot outrun the buffer holding it while the loop is
		 * busy doing everything else.
		 */
		int b_press = 0, b_release = 0;
		const int b_held = bc_overlay_button(lv->ov, &b_press, &b_release);
		int cur_x = 0, cur_y = 0;
		const int have_ptr = bc_overlay_pointer(lv->ov, &cur_x, &cur_y);
		const int draw_w = bc_overlay_width(lv->ov);
		const int draw_h = bc_overlay_height(lv->ov);
		const int shortest = draw_w < draw_h ? draw_w : draw_h;
		/*
		 * How near the start the stroke must come back to count
		 * as closed, and how far a press may travel and still be
		 * a click rather than a trace. Both scale with the
		 * screen: a tolerance in raw pixels means something
		 * different on a 1080p panel than on this one.
		 */
		const float close_tol = 0.04f * (float)shortest;
		const float click_slop = 0.006f * (float)shortest;

		if (b_press && have_ptr) {
			lv->stroking   = 1;
			lv->n_path     = 0;
			lv->path_len   = 0.0f;
			lv->path_min_d = 1.0f;
			lv->path_x[lv->n_path] = (float)cur_x;
			lv->path_y[lv->n_path] = (float)cur_y;
			lv->n_path++;
		}

		if (lv->stroking && b_held && have_ptr && lv->n_path > 0) {
			const float dx = (float)cur_x - lv->path_x[lv->n_path - 1];
			const float dy = (float)cur_y - lv->path_y[lv->n_path - 1];
			const float d  = sqrtf(dx * dx + dy * dy);
			/*
			 * A point per path_min_d of travel at most.
			 * Standing still would otherwise fill the buffer
			 * with copies of one position and end the stroke
			 * without the hand having moved.
			 */
			if (d >= lv->path_min_d) {
				/*
				 * Out of room: halve the resolution
				 * instead of stopping.
				 *
				 * A fixed spacing makes the buffer a
				 * *travel* limit -- at one point per
				 * pixel, BC_PATH_MAX ran out after about
				 * a thousand pixels of movement, which is
				 * less than one lap around a face on this
				 * screen. The stroke then froze mid-trace
				 * and, because its last point stopped
				 * following the pointer, could never come
				 * back to its start, so the shape was
				 * refused as unclosed. Both halves of
				 * that were this one line.
				 *
				 * Dropping every other sample and
				 * doubling the spacing keeps the trace
				 * going with the same memory and costs
				 * only resolution, of which there is far
				 * more than needed: the result is thinned
				 * to 64 points regardless.
				 */
				if (lv->n_path >= BC_PATH_MAX) {
					int k = 0;
					for (int i = 0; i < lv->n_path; i += 2) {
						lv->path_x[k] = lv->path_x[i];
						lv->path_y[k] = lv->path_y[i];
						k++;
					}
					lv->n_path = k;
					lv->path_min_d *= 2.0f;
					/* Logged because this is invisible
					 * otherwise, and its predecessor --
					 * silently giving up here -- cost a
					 * whole trace with no clue why. */
					LOG("drawing: long trace, sampling "
					    "every %.0f px now", lv->path_min_d);
				}
				if (lv->n_path < BC_PATH_MAX) {
					lv->path_x[lv->n_path] = (float)cur_x;
					lv->path_y[lv->n_path] = (float)cur_y;
					lv->n_path++;
					lv->path_len += d;
				}
			}
		}

		/*
		 * Is the loop closed *right now*? Read every frame, not
		 * only on release, because it is what the preview below
		 * shows: the shape fills in as soon as releasing would
		 * be accepted, so the rule is visible before it is
		 * enforced rather than after.
		 */
		lv->path_closed = 0;
		if (lv->stroking && lv->n_path >= 8 && lv->path_len > close_tol * 4.0f) {
			const float dx = lv->path_x[lv->n_path - 1] - lv->path_x[0];
			const float dy = lv->path_y[lv->n_path - 1] - lv->path_y[0];
			lv->path_closed = sqrtf(dx * dx + dy * dy) <= close_tol;
		}

		/*
		 * A click and a trace arrive through the same button.
		 * They are told apart by how far the pointer travelled
		 * between press and release, which is the distinction
		 * the hand is already making -- a click is aimed and
		 * still, a trace is the whole point of moving.
		 */
		int synth_enter = 0;
		if (b_release && lv->stroking) {
			lv->stroking = 0;
			if (lv->path_len <= click_slop) {
				synth_enter = 1;
				lv->n_path = 0;
			} else if (!lv->path_closed) {
				/*
				 * Refused, as asked: an open curve has
				 * no inside, and guessing where the
				 * missing piece went would be inventing
				 * the boundary of a censor.
				 */
				WARN("drawing: outline is not closed "
				     "-- finish where you started "
				     "(within %.0f px) and it will be "
				     "taken", (double)close_tol);
				lv->n_path = 0;
			} else if (lv->cfg.n_regions >= BC_MAX_REGIONS) {
				WARN("drawing: already at the %d region "
				     "limit -- not saved", BC_MAX_REGIONS);
				lv->n_path = 0;
			} else if (draw_w <= 0 || draw_h <= 0) {
				lv->n_path = 0;
			} else {
				/*
				 * Thin the stroke down to something a
				 * config file can hold. The tolerance
				 * is in screen fractions, so the path is
				 * normalised first and the simplifier
				 * never has to know the resolution.
				 */
				float ux[BC_PATH_MAX], uy[BC_PATH_MAX];
				for (int i = 0; i < lv->n_path; i++) {
					ux[i] = lv->path_x[i] / (float)draw_w;
					uy[i] = lv->path_y[i] / (float)draw_h;
				}
				float sx[BC_MAX_POLY_PTS], sy[BC_MAX_POLY_PTS];
				const int k = simplify_to_fit(
					ux, uy, lv->n_path, sx, sy,
					BC_MAX_POLY_PTS);
				if (k < 3) {
					WARN("drawing: outline collapsed "
					     "to %d point(s) -- not saved", k);
				} else {
					float x0 = sx[0], y0 = sy[0];
					float x1 = sx[0], y1 = sy[0];
					for (int i = 1; i < k; i++) {
						if (sx[i] < x0) x0 = sx[i];
						if (sy[i] < y0) y0 = sy[i];
						if (sx[i] > x1) x1 = sx[i];
						if (sy[i] > y1) y1 = sy[i];
					}
					struct bc_region *r =
						&lv->cfg.regions[lv->cfg.n_regions++];
					r->x = x0; r->y = y0;
					r->w = x1 - x0; r->h = y1 - y0;
					r->effect = -1;
					r->n_pts = k;
					for (int i = 0; i < k; i++) {
						r->px[i] = sx[i];
						r->py[i] = sy[i];
					}
					/* A traced shape supersedes a
					 * corner left pending; the user
					 * has plainly moved on. */
					lv->have_anchor = 0;
					LOG("drawing: outline %d, %d points "
					    "from %d samples",
					    lv->cfg.n_regions, k, lv->n_path);
					bc_config_save(&lv->cfg);
					bc_config_poll(&lv->cfg);
					cfg_env_overrides(&lv->cfg);
				}
				lv->n_path = 0;
			}
		}

		int key;
		while ((key = bc_overlay_next_key(lv->ov)) != BC_KEY_NONE ||
		       synth_enter) {
			if (key == BC_KEY_NONE) {
				key = BC_KEY_ENTER;
				synth_enter = 0;
			}
			if (key == BC_KEY_ESC) {
				/* A stroke in progress is the nearest
				 * thing to abandon, before a pending
				 * corner and before the mode itself. */
				if (lv->stroking || lv->n_path > 0) {
					lv->stroking = 0;
					lv->n_path   = 0;
					LOG("drawing: outline abandoned");
					continue;
				}
				/* First Escape abandons a half-placed
				 * rectangle; a second leaves. Otherwise
				 * a mis-aimed first corner would cost
				 * the whole mode. */
				if (lv->have_anchor) {
					lv->have_anchor = 0;
					LOG("drawing: corner cleared");
				} else {
					lv->cfg.draw = 0;
				}
				continue;
			}

			int px, py;
			if (!bc_overlay_pointer(lv->ov, &px, &py)) {
				WARN("drawing: pointer is not on this "
				     "output -- nothing placed");
				continue;
			}
			if (!lv->have_anchor) {
				lv->anchor_x = px;
				lv->anchor_y = py;
				lv->have_anchor = 1;
				LOG("drawing: corner at %d,%d", px, py);
				continue;
			}

			/* Second corner: commit. */
			lv->have_anchor = 0;
			const int ow = bc_overlay_width(lv->ov);
			const int oh = bc_overlay_height(lv->ov);
			int x0 = lv->anchor_x < px ? lv->anchor_x : px;
			int y0 = lv->anchor_y < py ? lv->anchor_y : py;
			int x1 = lv->anchor_x < px ? px : lv->anchor_x;
			int y1 = lv->anchor_y < py ? py : lv->anchor_y;
			if (x1 - x0 < 2 || y1 - y0 < 2 || ow <= 0 || oh <= 0) {
				WARN("drawing: rectangle has no area "
				     "-- not saved");
				continue;
			}
			if (lv->cfg.n_regions >= BC_MAX_REGIONS) {
				WARN("drawing: already at the %d region "
				     "limit -- not saved", BC_MAX_REGIONS);
				continue;
			}
			/*
			 * Stored normalised, like every other region, so
			 * it still covers the same thing after a
			 * resolution change. The effect is left at -1:
			 * the user drew a shape, not a style, and a
			 * region pinned to today's global effect would
			 * stop following it.
			 */
			struct bc_region *r = &lv->cfg.regions[lv->cfg.n_regions++];
			r->x = (float)x0 / (float)ow;
			r->y = (float)y0 / (float)oh;
			r->w = (float)(x1 - x0) / (float)ow;
			r->h = (float)(y1 - y0) / (float)oh;
			r->effect = -1;
			LOG("drawing: region %d = %.4f,%.4f,%.4f,%.4f",
			    lv->cfg.n_regions, r->x, r->y, r->w, r->h);
			/*
			 * Written out now rather than on the way out.
			 * The region is already covering the screen, and
			 * a crash between here and the end of the mode
			 * would otherwise lose a cover the user has
			 * every reason to think is saved.
			 */
			bc_config_save(&lv->cfg);
			bc_config_poll(&lv->cfg);
			cfg_env_overrides(&lv->cfg);
		}

		/* The flag may have just been cleared; answer it on the
		 * next frame through the transition above. */
	}
}

/* Rasterise the drawn outlines into the clip mask. */
static void step_outlines(struct bc_live *lv)
{
	/*
	 * 0d. Rasterise the outlines into the clip mask.
	 *
	 * Rebuilt when the outline set changes, and every frame while a
	 * stroke is being traced -- that is the only time the shape moves.
	 *
	 * Failure at any step here leaves the mask off, and the mask
	 * being off means every box is covered whole (see the shader's
	 * clip). So an allocation that does not happen costs a shape its
	 * precision and never its coverage.
	 */
	{
		int want_poly = 0;
		for (int i = 0; i < lv->cfg.n_regions; i++)
			if (lv->cfg.regions[i].n_pts >= 3)
				want_poly = 1;
		const int want = want_poly || (lv->drawing && lv->n_path > 0);

		if (want && !lv->lasso_mask) {
			const int ow = bc_overlay_width(lv->ov);
			const int oh = bc_overlay_height(lv->ov);
			if (ow > 0 && oh > 0) {
				if (ow >= oh) {
					lv->lasso_mw = ow < BC_MASK_LONG
					         ? ow : BC_MASK_LONG;
					lv->lasso_mh = (int)((float)lv->lasso_mw *
					                 (float)oh / (float)ow + 0.5f);
				} else {
					lv->lasso_mh = oh < BC_MASK_LONG
					         ? oh : BC_MASK_LONG;
					lv->lasso_mw = (int)((float)lv->lasso_mh *
					                 (float)ow / (float)oh + 0.5f);
				}
				if (lv->lasso_mw < 1) lv->lasso_mw = 1;
				if (lv->lasso_mh < 1) lv->lasso_mh = 1;
				const size_t px = (size_t)lv->lasso_mw * lv->lasso_mh;
				lv->lasso_mask = malloc(px);
				lv->lasso_tmp  = malloc(px);
				lv->lasso_dil  = malloc(px);
				lv->lasso_band = malloc(px);
				lv->lasso_base = malloc(px);
				if (!lv->lasso_mask || !lv->lasso_tmp ||
				    !lv->lasso_dil || !lv->lasso_band ||
				    !lv->lasso_base) {
					WARN("outlines: out of memory for a "
					     "%dx%d mask -- shapes will cover "
					     "their whole bounding box",
					     lv->lasso_mw, lv->lasso_mh);
					free(lv->lasso_mask); free(lv->lasso_tmp);
					free(lv->lasso_dil);  free(lv->lasso_band);
					free(lv->lasso_base);
					lv->lasso_mask = lv->lasso_tmp = NULL;
					lv->lasso_dil  = lv->lasso_band = NULL;
					lv->lasso_base = NULL;
				}
			}
		}

		if (want && lv->lasso_mask) {
			const unsigned long stamp = regions_stamp(&lv->cfg);
			const int live = lv->drawing && lv->n_path > 0;
			const size_t mpx = (size_t)lv->lasso_mw * lv->lasso_mh;
			/*
			 * The committed shapes are rebuilt only when
			 * they change; a frame of live drawing copies
			 * that result and stamps the line onto it.
			 *
			 * Rasterising and dilating the whole mask every
			 * frame is what the loop used to do, and it is
			 * far and away the most expensive thing here --
			 * a dilation touches every pixel five times per
			 * axis, so it grows with the mask's area and
			 * would have set a ceiling on how fine the mask
			 * could be. A memcpy does not care.
			 */
			if (!lv->lasso_built || stamp != lv->lasso_stamp) {
				memset(lv->lasso_mask, 0, mpx);
				for (int i = 0; i < lv->cfg.n_regions; i++) {
					const struct bc_region *r = &lv->cfg.regions[i];
					if (r->n_pts >= 3)
						fill_poly(lv->lasso_mask,
						          lv->lasso_mw, lv->lasso_mh,
						          r->px, r->py, r->n_pts);
				}
				/*
				 * Two levels out of one shape: 255 for
				 * what is covered, 128 for the outline
				 * band around it, 0 for the rest. The
				 * band is the shape grown by the outline
				 * width minus the shape itself, which is
				 * what makes an outline follow a traced
				 * edge instead of the rectangle it is
				 * drawn inside.
				 *
				 * Both start from the same dilation the
				 * §2 margin needs, so the outline lies
				 * outside the grown cover and never eats
				 * into it.
				 */
				dilate_mask(lv->lasso_mask, lv->lasso_dil, lv->lasso_tmp,
				            lv->lasso_mw, lv->lasso_mh, BC_MASK_GROW);
				if (lv->cfg.outline > 0.0f && lv->lasso_band) {
					/* Outline width is in screen
					 * pixels; the mask is not at the
					 * screen's resolution. */
					const int ow2 = bc_overlay_width(lv->ov);
					int band = (int)(lv->cfg.outline *
						(float)lv->lasso_mw /
						(ow2 > 0 ? (float)ow2 : 1.0f)
						+ 0.5f);
					if (band < 1)
						band = 1;
					/* Grown from the *shape*, not from
					 * the already-grown copy: chaining
					 * the two dilations would add their
					 * radii and draw an outline wider
					 * than the one asked for. */
					dilate_mask(lv->lasso_mask, lv->lasso_band,
					            lv->lasso_tmp, lv->lasso_mw,
					            lv->lasso_mh,
					            BC_MASK_GROW + band);
					for (size_t i = 0;
					     i < (size_t)lv->lasso_mw * lv->lasso_mh; i++)
						lv->lasso_dil[i] = lv->lasso_dil[i]
						             ? 255
						             : (lv->lasso_band[i] ? 128 : 0);
				}
				/* Measured on the finished mask, against
				 * the boxes it is about to clip. */
				{
					double box_px = 0.0;
					for (int i = 0; i < lv->cfg.n_regions; i++) {
						const struct bc_region *r =
							&lv->cfg.regions[i];
						if (r->n_pts < 3)
							continue;
						box_px += (double)(r->w * lv->lasso_mw) *
						          (double)(r->h * lv->lasso_mh);
					}
					long on = 0;
					for (size_t i = 0; i < mpx; i++)
						on += lv->lasso_dil[i] == 255;
					if (box_px > 0.0) {
						lv->lasso_kept_pct = 100.0 *
							(double)on / box_px;
						lv->lasso_measured = 1;
					}
				}
				memcpy(lv->lasso_base, lv->lasso_dil, mpx);
				lv->lasso_stamp = stamp;
				lv->lasso_built = 1;
				lv->lasso_uploaded = 0;
			}

			/*
			 * This frame's mask: the committed shapes, plus
			 * whatever is being traced right now.
			 *
			 * The line is stamped one mask pixel wide and
			 * *not* dilated. Dilation is the §2 margin on a
			 * shape that is about to censor something; the
			 * line being traced censors nothing yet, it is a
			 * cursor. Running it through the same growth
			 * made it several times the width it is drawn
			 * at -- too coarse to trace an edge with. What
			 * the shape finally encloses is unaffected: the
			 * fill comes from the points, not from the
			 * pixels of the line.
			 */
			if (live) {
				const int ow4 = bc_overlay_width(lv->ov);
				const int oh4 = bc_overlay_height(lv->ov);
				float ux[BC_PATH_MAX], uy[BC_PATH_MAX];
				for (int i = 0; i < lv->n_path; i++) {
					ux[i] = lv->path_x[i] / (float)ow4;
					uy[i] = lv->path_y[i] / (float)oh4;
				}
				memcpy(lv->lasso_dil, lv->lasso_base, mpx);
				/*
				 * Once the loop would be accepted it is
				 * shown filled, and until then only as a
				 * line. The preview is the rule: if it
				 * is solid, let go and it is taken.
				 */
				if (lv->path_closed)
					fill_poly(lv->lasso_dil,
					          lv->lasso_mw, lv->lasso_mh,
					          ux, uy, lv->n_path);
				stroke_path(lv->lasso_dil, lv->lasso_mw, lv->lasso_mh,
				            ux, uy, lv->n_path, 1);
				lv->lasso_uploaded = 0;
			} else if (!lv->lasso_uploaded) {
				memcpy(lv->lasso_dil, lv->lasso_base, mpx);
			}

			if (!lv->lasso_uploaded || live) {
				/*
				 * v runs the other way: GL's origin is
				 * the bottom of the screen and the mask
				 * was rasterised from the top, so the
				 * screen's bottom edge is the mask's
				 * last row.
				 */
				bc_overlay_set_silhouette(lv->ov, lv->lasso_dil,
				                          lv->lasso_mw, lv->lasso_mh,
				                          0.0f, 1.0f, 1.0f, 0.0f);
				lv->lasso_uploaded = 1;
				lv->lasso_have = 1;
			}
		} else if (lv->lasso_have && !want) {
			bc_overlay_set_silhouette(lv->ov, NULL, 0, 0, 0, 0, 1, 1);
			lv->lasso_have     = 0;
			lv->lasso_built    = 0;
			lv->lasso_uploaded = 0;
		}
	}
}

/* Capture. Non-blocking; imports the newest frame if any. */

/*
 * Screen delay: the ring and the record of boxes are sized from the config,
 * and rebuilt when it changes. A delay of 0 is the live overlay and costs
 * nothing -- no ring, no record, no opaque background.
 *
 * The ring holds the delay plus 400 ms, so a slow detection pass lengthens the
 * lag (the screen waits for the frame it can vouch for) rather than stalling
 * it. It keeps a frame every `ring_interval` ms, not every captured frame: at
 * 140 captured frames a second the full history would be a gigabyte of VRAM,
 * and the picture only needs to be as smooth as the video in it.
 */
static void delay_apply(struct bc_live *lv)
{
	float want = lv->cfg.delay_ms;
	const char *e = getenv("BC_DELAY_MS");
	if (e)
		want = (float)atof(e);
	if (!(want >= 0.0f))
		want = 0.0f;
	if (want > 1000.0f)
		want = 1000.0f;
	if (want == lv->delay_applied)
		return;

	if (lv->ring) { bc_framering_destroy(lv->ring); lv->ring = NULL; }
	if (lv->dh)   { bc_dh_destroy(lv->dh); lv->dh = NULL; }
	lv->delay_applied = want;
	lv->shown_valid = 0;
	lv->play_have = 0;
	lv->snap_pending = 0;
	bc_overlay_set_background(lv->ov, 0);
	if (want <= 0.0f) {
		LOG("screen delay: off");
		return;
	}
	const double window = (double)want + 400.0;
	lv->ring_interval = window / 50.0 > 16.0 ? window / 50.0 : 16.0;
	const int slots = (int)ceil(window / lv->ring_interval) + 3;
	lv->ring = bc_framering_create(bc_overlay_width(lv->ov),
	                               bc_overlay_height(lv->ov), slots);
	lv->dh = bc_dh_create(512);
	if (!lv->ring || !lv->dh) {
		ERR("screen delay unavailable (no frame ring) -- staying live");
		if (lv->ring) { bc_framering_destroy(lv->ring); lv->ring = NULL; }
		if (lv->dh)   { bc_dh_destroy(lv->dh); lv->dh = NULL; }
		return;
	}
	LOG("screen delay: %.0f ms, one frame kept every %.0f ms", want,
	    lv->ring_interval);
}

/* Keep a copy of the frame just captured, before the compositor reuses it. */
static void delay_push_frame(struct bc_live *lv)
{
	const unsigned tex = bc_gpu_texture(lv->gpu);
	if (!lv->ring || !tex)
		return;
	if (lv->last_frame - lv->ring_last_push < lv->ring_interval - 1.0)
		return;
	if (bc_framering_push(lv->ring, tex, ++lv->ring_seq, lv->last_frame) == 0) {
		lv->ring_last_push = lv->last_frame;
		/* The copy must keep the colours: debug only, it reads pixels back. */
		if (getenv("BC_RING_CHECK") && lv->ring_seq == 20)
			bc_framering_selftest(lv->ring, tex);
	}
}

/*
 * The last step before presenting: with the delay on, replace "the covers for
 * now over the screen of now" with "the covers for then over the screen of
 * then".
 *
 * `then` is the delay ago, but never later than the newest thing the censor
 * has looked at -- a frame nothing at or after it has been checked against is
 * not shown, so a slow detector makes the screen lag further instead of
 * letting an unchecked frame through. If even the oldest frame the ring holds
 * is not vouched for, the last frame shown is held; and if there is none,
 * black. Never an unchecked frame (§6).
 *
 * Only an ordinary censoring frame is delayed. Panic, pause, the master
 * switch, the fail-closed blank and hand-drawing all mean "act on what is
 * real now", and show the live screen or the blank exactly as before.
 */
static int has_doubled_bar(const struct bc_box *bx, int n);

static void delay_prepare(struct bc_live *lv)
{
	if (!lv->ring || !lv->dh)
		return;
	if (lv->show != BC_SHOW_CENSOR || lv->cfg.draw) {
		bc_dh_reset(lv->dh);
		lv->snap_pending = 0;
		lv->shown_valid = 0;
		lv->play_have = 0;
		bc_overlay_set_background(lv->ov, 0);
		return;
	}
	if (lv->snap_pending) {
		bc_dh_push(lv->dh, lv->snap_t, lv->boxes, lv->n_boxes);
		lv->snap_pending = 0;
	}

	/*
	 * The display clock. It runs at real speed and is held back only by what
	 * has been checked: it may not pass the newest record. Capping a target
	 * of `now - delay` at the newest record instead makes the picture advance
	 * in jumps the size of the detection interval -- 3 frames a second for a
	 * heavy model on a tile grid -- because the newest record moves in jumps.
	 * Here a stall costs the delay a little more lag and then plays on at full
	 * speed, so the lag grows to the smallest value that does not stall and
	 * stays there; it drifts back toward the delay asked for, slowly, only
	 * when nothing has stalled for a while.
	 */
	const double want = (double)lv->delay_applied;
	const double newest = bc_dh_newest(lv->dh);
	if (!lv->play_have) {
		lv->play_t = lv->t - want;
		lv->play_have = 1;
	} else {
		/* 12% fast closes a second of excess lag in under ten, and only
		 * once the clock has been free of the horizon for a while. */
		const double rate = (lv->t - lv->last_stall > 1500.0 &&
		                     lv->t - lv->play_t > want + 30.0) ? 1.12 : 1.0;
		lv->play_t += (lv->t - lv->play_last) * rate;
	}
	lv->play_last = lv->t;
	if (lv->play_t > lv->t - want)
		lv->play_t = lv->t - want;                  /* never less lag than asked */
	if (lv->play_t < lv->t - (want + 350.0))
		lv->play_t = lv->t - (want + 350.0);        /* the ring's reach */
	if (lv->play_t > newest) {
		lv->play_t = newest;                        /* nothing has checked past here */
		lv->last_stall = lv->t;
		lv->delay_stalled++;
	}
	const double t_sel = lv->play_t;

	struct bc_box tmp[BC_MAX_BOXES];
	unsigned long seq = 0;
	double tf = 0.0, gap = 0.0;
	unsigned tex = bc_framering_at(lv->ring, t_sel, &seq, &tf);
	int n = tex ? bc_dh_boxes(lv->dh, tf, tmp, BC_MAX_BOXES, &gap) : -1;

	if (tex && n >= 0) {
		memcpy(lv->boxes, tmp, (size_t)n * sizeof tmp[0]);
		lv->n_boxes = n;
		memcpy(lv->shown_boxes, tmp, (size_t)n * sizeof tmp[0]);
		lv->shown_n = n;
		lv->delay_doubled += (unsigned long)has_doubled_bar(lv->boxes, n);
		if (!lv->shown_valid || lv->shown_seq != seq)
			lv->delay_distinct++;
		lv->shown_seq = seq;
		lv->shown_valid = 1;
		lv->delay_shown++;
		stat_add(&lv->s_delay_lag, lv->t - tf);
		stat_add(&lv->s_delay_gap, gap);
	} else if (lv->shown_valid &&
	           (tex = bc_framering_get(lv->ring, lv->shown_seq)) != 0) {
		memcpy(lv->boxes, lv->shown_boxes,
		       (size_t)lv->shown_n * sizeof lv->boxes[0]);
		lv->n_boxes = lv->shown_n;
		lv->delay_held++;
	} else {
		blank_all(lv->ov, lv->boxes, &lv->n_boxes);
		lv->delay_blank++;
		bc_overlay_set_background(lv->ov, 0);
		return;
	}
	bc_overlay_set_source(lv->ov, tex, bc_gpu_width(lv->gpu),
	                      bc_gpu_height(lv->gpu));
	bc_overlay_set_background(lv->ov, 1);
}

/* BC_DEBUG_TENSOR writes the detector's input once; BC_DEBUG_TENSOR_AFTER_MS
 * holds it back until that long into the run, so it can show what the
 * detector sees once the overlay is up (the first frame never can). */
static int dump_due(const struct bc_live *lv)
{
	if (!lv->dump_path || lv->dumped)
		return 0;
	const char *e = getenv("BC_DEBUG_TENSOR_AFTER_MS");
	return !e || now_ms() - lv->start >= atof(e);
}

static void step_capture(struct bc_live *lv)
{
	delay_apply(lv);
	double since_start = now_ms() - lv->start;
	int faked_stall = lv->fault_stall > 0.0 && since_start > lv->fault_stall;
	int faked_dead  = lv->fault_dead  > 0.0 && since_start > lv->fault_dead;

	lv->got = (lv->capture_dead || faked_dead) ? -1
	        : faked_stall ? 0
	        : bc_capture_poll(lv->cap);
	if (lv->got < 0) {
		/*
		 * Do not break. Exiting here tears down the overlay and
		 * leaves the screen uncensored, which is precisely the
		 * fail-open outcome §6 forbids. Latch instead and keep
		 * presenting -- the loop below blanks the output.
		 */
		if (!lv->capture_dead)
			ERR("capture stream lost -- blanking output");
		lv->capture_dead = 1;
	}
	if (lv->got > 0) {
		double gap = now_ms() - lv->last_frame;
		if (gap > lv->max_frame_gap)
			lv->max_frame_gap = gap;
		lv->last_frame = now_ms();
		bc_overlay_set_source(lv->ov, bc_gpu_texture(lv->gpu),
		                      bc_gpu_width(lv->gpu), bc_gpu_height(lv->gpu));
		delay_push_frame(lv);
	}
}

/* Preprocess and hand the frame to the detectors, if they are free. */
static enum bc_next step_submit(struct bc_live *lv)
{
	/*
	 * 2. Preprocess + hand off, but only if the worker is free.
	 *
	 * Readback stalls the GL pipeline, so building a tensor for a
	 * frame that will be dropped is pure waste. This comment said
	 * so for a long time while the condition below did not check
	 * it: measured at a 2x1 grid, 841 readbacks fed 38 inferences,
	 * so 95% of the traffic -- 4.9 MB per tile, each way -- was
	 * produced and thrown away.
	 *
	 * The change gate is unaffected in the case it exists for. A
	 * still screen leaves the worker idle, so the gate still sees
	 * every frame; a busy worker means an inference is already in
	 * flight, which is exactly when there is nothing to decide.
	 * The gate's reference frame becomes "the last one we
	 * considered", which is the question it is actually asking.
	 */
	/*
	 * Segmentation, on its own cadence and its own worker.
	 *
	 * Gated only on that worker being free, so it runs as often as it
	 * can and no oftener; it is a second full-frame inference and has
	 * no business pacing itself off the primary detector's grid. The
	 * silhouette persists between runs the same way boxes do -- it
	 * describes where subjects are, which does not stop being roughly
	 * true between inferences.
	 */
	if (lv->seg_det && lv->got > 0 && bc_gpu_texture(lv->gpu) &&
	    !bc_detect_busy(lv->seg_det) &&
	    now_ms() - lv->seg_last >= SEG_MIN_MS) {
		lv->seg_last = now_ms();
		if (bc_preproc_run(lv->seg_pp, bc_gpu_texture(lv->gpu),
		                   bc_gpu_width(lv->gpu),
		                   bc_gpu_height(lv->gpu)) == 0 &&
		    bc_preproc_readback(lv->seg_pp, lv->seg_tensor) == 0)
			bc_detect_submit(lv->seg_det, lv->seg_tensor);
	}
	if (lv->seg_det) {
		const int mw = bc_detect_mask_w(lv->seg_det);
		const int mh = bc_detect_mask_h(lv->seg_det);
		if (bc_detect_take_mask(lv->seg_det, lv->seg_mask,
		                        (size_t)mw * mh) > 0) {
			/*
			 * Margin is a percentage of the screen's short
			 * side, so it means the same thing whatever the
			 * model's input size; convert it into mask
			 * pixels through the letterbox scale.
			 */
			const int sw = bc_gpu_width(lv->gpu);
			const int sh = bc_gpu_height(lv->gpu);
			const int shortest = sw < sh ? sw : sh;
			const float px_screen =
				lv->cfg.seg_margin * 0.01f * (float)shortest;
			const int r = (int)(px_screen *
				bc_preproc_scale(lv->seg_pp) *
				(float)mw / (float)bc_preproc_net_w(lv->seg_pp)
				+ 0.5f);

			/* seg_mask doubles as the scratch buffer: it is
			 * consumed by the dilation that reads it. */
			dilate_mask(lv->seg_mask, lv->seg_dil, lv->seg_mask, mw, mh, r);

			long on = 0;
			for (size_t i = 0; i < (size_t)mw * mh; i++)
				on += lv->seg_dil[i] != 0;
			lv->seg_cov_sum += 100.0 * (double)on /
			               ((double)mw * mh);
			lv->seg_runs++;

			/* Where the screen's corners land in the mask.
			 * GL's origin is bottom-left, so the first pair
			 * is the *bottom* of the screen. */
			float nx0, ny0, nx1, ny1;
			bc_preproc_to_net(lv->seg_pp, 0.0f, (float)sh,
			                  &nx0, &ny0);
			bc_preproc_to_net(lv->seg_pp, (float)sw, 0.0f,
			                  &nx1, &ny1);
			const float nw = (float)bc_preproc_net_w(lv->seg_pp);
			const float nh = (float)bc_preproc_net_h(lv->seg_pp);
			/*
			 * One mask texture, one uv mapping -- so the
			 * silhouette and the hand-drawn outlines cannot
			 * both have it, and the hand-drawn ones win.
			 * They are an explicit instruction about a
			 * specific shape; the silhouette is a guess
			 * about where a subject is. Deferring to the
			 * guess would let a model erase a boundary the
			 * user drew on purpose.
			 */
			if (lv->lasso_have) {
				/*
				 * And drop any silhouette already
				 * standing: it describes a texture that
				 * is no longer loaded, so leaving it set
				 * would clip every detection cover to a
				 * shape the user drew somewhere else.
				 */
				lv->seg_have = 0;
				if (!lv->seg_shadowed) {
					WARN("segmentation: outlines are "
					     "drawn, so covers clip to those "
					     "and not to the silhouette");
					lv->seg_shadowed = 1;
				}
			} else {
				bc_overlay_set_silhouette(lv->ov, lv->seg_dil, mw, mh,
				                    nx0 / nw, ny0 / nh,
				                    nx1 / nw, ny1 / nh);
				lv->seg_have = 1;
			}
		}
	}

	if (lv->got > 0 && bc_gpu_texture(lv->gpu) && bc_detect_busy(lv->det)) {
		lv->readbacks_skipped++;
	} else if (lv->got > 0 && bc_gpu_texture(lv->gpu)) {
		/*
		 * Take the tensor buffer back from CUDA before writing
		 * a new frame into it.
		 *
		 * This is the only point in the loop where that is
		 * known to be safe, and it is safe for the same reason
		 * the branch above exists: getting here means the
		 * detector is idle, so no inference is still reading
		 * it. Zero-copy removed the memcpy in
		 * bc_detect_submit() that used to make the question
		 * moot -- the tensor the worker runs on is now the very
		 * buffer the next dispatch overwrites, so the exclusion
		 * has to be real rather than incidental. A no-op on a
		 * CPU session, and on any frame that did not submit.
		 */
		bc_preproc_gpu_release(lv->pp);

		lv->t = now_ms();
		if (bc_preproc_run(lv->pp, bc_gpu_texture(lv->gpu),
		                   bc_gpu_width(lv->gpu), bc_gpu_height(lv->gpu)) == 0) {
			{
				/*
				 * Occlusion-aware hold. This, not the sticky
				 * timer, is what actually breaks the Phase 2
				 * feedback loop.
				 *
				 * We draw an opaque bar, then capture our own
				 * output, so the detector is blind exactly where
				 * we censor and always reports "nothing there".
				 * A sticky timer only sets how often that
				 * blindness lifts the box -- it cannot stop it,
				 * which is the ~500 ms flicker.
				 *
				 * So: if the region under a box is *unchanged*
				 * since the last frame, we are looking at our own
				 * bar and have learned nothing; keep the box.
				 * Only once the region actually changes does the
				 * sticky timer get to expire it.
				 */
				/*
				 * Gate before the memcpy below overwrites
				 * the frame we are comparing against.
				 */
				/*
				 * The same comparison answers the
				 * opposite question too: not "did
				 * nothing change" but "did everything".
				 * A scroll replaces the picture, which
				 * invalidates every track's motion model
				 * at once -- see bc_tracker_scene_cut().
				 */
				/*
				 * Answered on the GPU, where the tensor
				 * already is: four bytes come back rather
				 * than the whole thing. See
				 * bc_preproc_change_frac().
				 */
				const float chg =
					bc_preproc_change_frac(lv->pp, GATE_EPS);
				stat_add(&lv->s_pre, now_ms() - lv->t);

				if (lv->have_prev && chg > SCENE_FRAC) {
					bc_tracker_scene_cut(&lv->tracker);
					lv->scene_cuts++;
				}
				/*
				 * The reference is committed when a frame
				 * is *submitted*, so after a run that
				 * failed it is a frame nothing was ever
				 * measured on. "Unchanged since then" would
				 * hold whatever boxes predate it and feed
				 * the inference deadline below -- so new
				 * content arriving on the failed frame
				 * stayed uncovered for as long as the
				 * screen was still, and the detector could
				 * die without §6 ever noticing. Until a run
				 * succeeds, nothing counts as unchanged:
				 * every frame retries, and the deadline
				 * only hears about real results.
				 */
				int unchanged =
					lv->have_prev && !lv->no_gate &&
					chg == 0.0f && !lv->det_failing;
				lv->have_prev = 1;

				if (unchanged && lv->force_infer) {
					/*
					 * A hold cap fired, so some track is
					 * about to be dropped for want of a
					 * detection -- and on a motionless
					 * screen the gate would never let one
					 * happen. That is a deadlock, and it
					 * fails open: identical frames keep
					 * arriving (so no stall deadline),
					 * the gate keeps skipping inference,
					 * and the bar comes off content that
					 * provably has not moved. It stayed
					 * off until something changed on
					 * screen -- moving the mouse was
					 * enough, which is how it was found.
					 *
					 * A cap means "I no longer trust this
					 * box", which is a reason to measure
					 * again, not to drop it blind. So the
					 * cap overrides the gate: infer this
					 * frame even though it is identical.
					 *
					 * Cost is bounded and small. A cap can
					 * only fire once per track per cap
					 * window, and a detection resets the
					 * window, so a static screen with
					 * content on it costs about one
					 * inference per second -- ~15 ms of
					 * GPU. If the content really is gone
					 * the forced inference returns
					 * nothing, the track expires for a
					 * real reason, and with no tracks
					 * left no cap fires and the gate goes
					 * quiet again.
					 */
					lv->gate_forced++;
					if (submit_tensor(lv->pp, lv->det, lv->tensor, lv->want_host, &lv->host_filled)) {
						lv->submit_t = lv->last_frame;
						lv->submitted++;
						lv->force_infer = 0;
						hold_after_submit(&lv->tracker, lv->pp,
						    lv->tensor, lv->prev_tensor, &lv->hold_have_prev,
						    lv->host_filled, now_ms(), &lv->s_mad,
						    &lv->held_released, &lv->held_capped);
					} else {
						/* worker busy -- retry next frame */
						lv->skipped++;
					}
				} else if (unchanged) {
					/*
					 * Nothing on screen moved, so the
					 * last results still describe it.
					 * Two consequences, both required
					 * or the gate is a hole:
					 *
					 * Tracks must be refreshed. The
					 * sticky timer would otherwise
					 * expire boxes after 500 ms of a
					 * screen that still has the same
					 * content on it -- skipping the
					 * inference must not also drop the
					 * censoring it was paying for. The
					 * ceiling still applies, so no box
					 * becomes permanent.
					 *
					 * The fail-closed inference
					 * deadline must not fire either:
					 * we chose not to infer, which is
					 * not the detector dying.
					 */
					double tg = now_ms();
					/* see hold_or_cap() */
					if (hold_or_cap(&lv->tracker, tg,
					                &lv->held_capped))
						lv->force_infer = 1;
					lv->last_infer = tg;
					lv->gated++;
					/* Nothing moved since the last frame looked at: that
					 * frame, and this one, are covered by what is drawn. */
					lv->snap_pending = 1;
					if (lv->last_frame > lv->snap_t)
						lv->snap_t = lv->last_frame;
				} else if (lv->peek_enabled && !lv->clean_pending &&
				           now_ms() - lv->last_clean_ms < GATE_BLIND_MS) {
					/*
					 * The clean-frame gate.
					 *
					 * Between peeks the capture is a
					 * photograph of our own covers, and
					 * a detection made from it is a
					 * detection of a censor bar: boxes
					 * on bar edges, boxes a little off
					 * the subject, boxes that come and
					 * go while the content underneath
					 * has not moved at all. Feeding
					 * those to the tracker is what
					 * stacked covers on one subject and
					 * placed them wrong on a screen
					 * that was perfectly still.
					 *
					 * So they are not made. The peeked
					 * frames are the only evidence, at
					 * ~2 Hz, which a still picture does
					 * not need more than -- and which
					 * costs about an eighth of the
					 * inference this path was spending
					 * on frames of itself.
					 *
					 * Everything live is frozen while
					 * we wait: held against expiry, and
					 * not predicted, because a guess
					 * about content we cannot see is
					 * what grew the boxes before.
					 */
					double tg = now_ms();
					for (int k = 0; k < BC_TRK_MAX; k++)
						bc_trk_freeze(&lv->tracker.t[k], tg);
					/* Waiting on a clean frame is not
					 * evidence of a working detector
					 * when its last run failed. */
					if (!lv->det_failing)
						lv->last_infer = tg;
					lv->gated++;
				} else if (lv->peek_enabled && !lv->clean_pending &&
				           !lv->warned_blind) {
					lv->warned_blind = 1;
					WARN("no clean frame in %.0f s -- "
					     "the peek gate is standing "
					     "down and inferring from "
					     "captured frames that contain "
					     "the overlay; expect the "
					     "oscillation it exists to "
					     "prevent", GATE_BLIND_MS / 1000.0);
					return BC_NEXT_FRAME;
				} else if (lv->fault_nodet > 0.0 &&
				           now_ms() - lv->start > lv->fault_nodet) {
					/* injected: pretend the detector
					 * is permanently busy */
					lv->skipped++;
				} else if (submit_tensor(lv->pp, lv->det, lv->tensor,
				                         dump_due(lv) || lv->want_host,
				                         &lv->host_filled)) {
					lv->submit_t = lv->last_frame;
					/*
					 * The only place the tensor can
					 * cross the bus, and only when it
					 * has to: with a CUDA session the
					 * detector reads it where it lies
					 * and nothing crosses at all.
					 */
					/* The ordinary path, and so the
					 * one the hold lives on. */
					if (lv->clean_pending) {
						lv->submitted_clean = 1;
						lv->clean_pending = 0;
					}
					hold_after_submit(&lv->tracker, lv->pp,
					    lv->tensor, lv->prev_tensor, &lv->hold_have_prev,
					    lv->host_filled, now_ms(), &lv->s_mad,
					    &lv->held_released, &lv->held_capped);
					if (dump_due(lv)) {
						bc_preproc_dump_ppm(lv->pp, lv->tensor,
						                    lv->dump_path);
						lv->dumped = 1;
					}
					lv->submitted++;
					/*
					 * Same frame, second model. Only on
					 * frames the primary runs on: an eye
					 * position is only ever used to place
					 * a bar inside a face from the same
					 * frame, so running it more often
					 * would buy nothing.
					 */
					if (lv->eye_det && lv->eye_pp &&
					    bc_preproc_run(lv->eye_pp,
					                   bc_gpu_texture(lv->gpu),
					                   bc_gpu_width(lv->gpu),
					                   bc_gpu_height(lv->gpu)) == 0 &&
					    bc_preproc_readback(lv->eye_pp,
					                        lv->eye_tensor) == 0)
						bc_detect_submit(lv->eye_det, lv->eye_tensor);
				} else {
					lv->skipped++;
				}
			}
		}
	} else if (lv->got == 0 && !lv->capture_dead && lv->have_prev &&
	           now_ms() - lv->last_frame > QUIET_MS) {
		/*
		 * 2b. No frame arrived at all.
		 *
		 * PipeWire is damage-driven: the compositor sends a
		 * buffer when something on the monitor is redrawn and
		 * says nothing when nothing is. The change gate above
		 * only runs on a frame, so a screen that goes
		 * completely still stops feeding the tracker
		 * altogether -- and silence was charged against the
		 * tracks as if it were absence. They went LOST, coasted
		 * on the filter, and the box drifted off content that
		 * had not moved. It only corrected when the mouse gave
		 * the compositor a reason to send a frame, which is why
		 * this looked like boxes sticking until you wiggle the
		 * cursor.
		 *
		 * No frame *is* the evidence: the screen is unchanged,
		 * so this is the same situation the gate already
		 * handles, and it gets the same answer.
		 *
		 * This cannot touch moving content. It needs a quarter
		 * second in which the compositor drew nothing at all --
		 * two orders of magnitude longer than the gap between
		 * frames of anything animating, and longer than a frame
		 * of even a 10 fps video. While anything moves,
		 * frames keep arriving, got > 0, and the branch above
		 * runs exactly as it did before.
		 */
		double tg = now_ms();
		lv->quiet_holds++;
		/*
		 * After a failed run the stillness proves nothing (see
		 * det_failing above), so it neither feeds the deadline
		 * nor waits for a frame that may never come: the last
		 * frame is re-measured, as for a cap. One transient
		 * error then costs one retry rather than a blank that
		 * lasts until something on screen moves.
		 */
		if (hold_or_cap(&lv->tracker, tg, &lv->held_capped) || lv->det_failing) {
			/*
			 * A track hit the ceiling. On a frame this
			 * latches force_infer and the next frame pays
			 * it -- but here there is no next frame, and
			 * waiting for one is the deadlock this whole
			 * branch exists to break. Re-measure the last
			 * frame we captured instead; it is by
			 * construction what is still on the screen.
			 */
			/* No host copy is kept any more; the GPU still
			 * holds the last frame it preprocessed, which is
			 * by construction what is still on screen. */
			if (submit_tensor(lv->pp, lv->det, lv->tensor, lv->want_host, &lv->host_filled)) {
						lv->submit_t = lv->last_frame;
				hold_after_submit(&lv->tracker, lv->pp, lv->tensor,
				    lv->prev_tensor, &lv->hold_have_prev, lv->host_filled,
				    now_ms(), &lv->s_mad, &lv->held_released,
				    &lv->held_capped);
				lv->submitted++;
				lv->gate_forced++;
				lv->force_infer = 0;
			} else {
				lv->skipped++;
			}
		}
		if (!lv->det_failing)
			lv->last_infer = tg;
	}
	return BC_GO_ON;
}

/* Fold any finished detections into the tracks. */
static void step_eyes(struct bc_live *lv)
{
	if (lv->eye_det && lv->pose_mode) {
		/*
		 * Pose faces, kept whole: the box so it can be matched
		 * against the primary model's head boxes, and the two
		 * eye keypoints so a matched face can be given an angle
		 * and a bar. Nothing here is censored on its own -- an
		 * unmatched pose face means the primary model saw no
		 * head there, and this model's opinion does not override
		 * that.
		 */
		struct bc_det ed[BC_MAX_DET];
		const struct bc_det_result er =
			bc_detect_collect(lv->eye_det, ed, BC_MAX_DET);
		const int ne = er.n;
		if (er.status == BC_DET_OK) {
			lv->n_pfaces = 0;
			for (int i = 0; i < ne && lv->n_pfaces < BC_MAX_DET; i++) {
				if (ed[i].n_kpt < 2)
					continue;
				float ax, ay, bx2, by2;
				bc_preproc_to_screen(lv->eye_pp, ed[i].x, ed[i].y,
				                     &ax, &ay);
				bc_preproc_to_screen(lv->eye_pp,
				                     ed[i].x + ed[i].w,
				                     ed[i].y + ed[i].h,
				                     &bx2, &by2);
				lv->pfaces[lv->n_pfaces].x0 = ax;
				lv->pfaces[lv->n_pfaces].y0 = ay;
				lv->pfaces[lv->n_pfaces].x1 = bx2;
				lv->pfaces[lv->n_pfaces].y1 = by2;
				for (int e = 0; e < 2; e++) {
					float sx, sy;
					bc_preproc_to_screen(lv->eye_pp,
					                     ed[i].kx[e],
					                     ed[i].ky[e],
					                     &sx, &sy);
					lv->pfaces[lv->n_pfaces].ex[e] = sx;
					lv->pfaces[lv->n_pfaces].ey[e] = sy;
					lv->pfaces[lv->n_pfaces].es[e] = ed[i].ks[e];
				}
				lv->n_pfaces++;
			}
			lv->pfaces_at = now_ms();
		}
	} else if (lv->eye_det) {
		struct bc_det ed[BC_MAX_DET];
		const struct bc_det_result er =
			bc_detect_collect(lv->eye_det, ed, BC_MAX_DET);
		const int ne = er.n;
		if (er.status == BC_DET_OK && ne > 0) {
			lv->n_eyes = 0;
			for (int i = 0; i < ne && lv->n_eyes < BC_MAX_DET; i++) {
				float ax, ay, bx2, by2;
				bc_preproc_to_screen(lv->eye_pp, ed[i].x, ed[i].y,
				                     &ax, &ay);
				bc_preproc_to_screen(lv->eye_pp,
				                     ed[i].x + ed[i].w,
				                     ed[i].y + ed[i].h,
				                     &bx2, &by2);
				lv->eyes[lv->n_eyes].x0 = ax; lv->eyes[lv->n_eyes].y0 = ay;
				lv->eyes[lv->n_eyes].x1 = bx2; lv->eyes[lv->n_eyes].y1 = by2;
				lv->n_eyes++;
			}
			lv->eyes_at = now_ms();
		} else if (er.status == BC_DET_OK) {
			lv->n_eyes = 0;
			lv->eyes_at = now_ms();
		}
	}
}

/* Collect the primary detector's answer and fold it into the tracks. */
/*
 * With the eye bar on, an EYE detection inside a face that is getting a bar
 * is the bar's business, not a cover of its own.
 *
 * Models whose EYE class never fired made this moot. puryfi YOLO26m's does
 * fire, and its eyes were tracked and drawn like any censored class -- with
 * the class's own effect, *after* the bar -- so a solid bar came out with a
 * pixelated window over each eye: the eyes shown through the one cover
 * meant to hide them best.
 *
 * "A face that is getting a bar" is a head detection in this set that its
 * class would let open a track, or a live head track already on screen --
 * the second because a track outlives a weak frame, and an eye dropped on
 * that frame must still be under a bar. An eye with no such face around it
 * is kept: there is no bar to cover it, and §2 says cover it anyway.
 *
 * Returns the new count; dropped detections are compacted out.
 */
static int drop_eyes_under_bars(struct bc_live *lv, struct bc_det *d, int n)
{
	if (!lv->cfg.eye_bar)
		return n;
	const int eye = bc_class_index("EYE");
	if (eye < 0)
		return n;

	int out = 0;
	for (int i = 0; i < n; i++) {
		int under = 0;
		if (d[i].cls == eye) {
			const float cx = d[i].x + d[i].w * 0.5f;
			const float cy = d[i].y + d[i].h * 0.5f;
			for (int j = 0; j < n && !under; j++) {
				const int c = d[j].cls;
				if (j == i || c < 0 || c >= BC_MAX_CLASSES ||
				    !lv->cfg.censor[c] ||
				    !is_head_class(bc_class_name(c)) ||
				    d[j].score < bc_cfg_class_conf(&lv->cfg, c))
					continue;
				under = cx >= d[j].x && cx <= d[j].x + d[j].w &&
				        cy >= d[j].y && cy <= d[j].y + d[j].h;
			}
			for (int k = 0; k < BC_TRK_MAX && !under; k++) {
				const struct bc_trk *t = &lv->tracker.t[k];
				float x, y, w, h;
				if (!bc_trk_box(t, &x, &y, &w, &h) ||
				    t->cls < 0 || t->cls >= BC_MAX_CLASSES ||
				    !lv->cfg.censor[t->cls] ||
				    !is_head_class(bc_class_name(t->cls)))
					continue;
				under = cx >= x && cx <= x + w &&
				        cy >= y && cy <= y + h;
			}
		}
		if (under) {
			lv->eyes_under_bar++;
			continue;
		}
		d[out++] = d[i];
	}
	return out;
}

static void step_detections(struct bc_live *lv)
{
	const struct bc_det_result res =
		bc_detect_collect(lv->det, lv->dets, BC_MAX_DET);
	/* n < 0: no fresh result to fold in, whether nothing finished
	 * or the run failed; the failure is acted on below. */
	int n = res.status == BC_DET_OK ? res.n : -1;
	lv->tnow = now_ms();
	if (n >= 0) {
		/* A look at the frame that was submitted, which is not this one. */
		lv->snap_pending = 1;
		if (lv->submit_t > lv->snap_t)
			lv->snap_t = lv->submit_t;
	}
	if (n > 0) {
		/*
		 * Into screen space, here and once.
		 *
		 * With a tile grid every detection is in *its own*
		 * tile's coordinates, so nothing below can compare two
		 * of them until they share a frame of reference. Doing
		 * it up front also means the seam merge and everything
		 * after it work in the units they are actually about.
		 */
		for (int i = 0; i < n; i++) {
			float x0, y0, x1, y1;
			const int t = lv->dets[i].tile;
			bc_preproc_tile_to_screen(lv->pp, t, lv->dets[i].x,
			                          lv->dets[i].y, &x0, &y0);
			bc_preproc_tile_to_screen(lv->pp, t,
			                          lv->dets[i].x + lv->dets[i].w,
			                          lv->dets[i].y + lv->dets[i].h,
			                          &x1, &y1);
			lv->dets[i].x = x0;
			lv->dets[i].y = y0;
			lv->dets[i].w = x1 - x0;
			lv->dets[i].h = y1 - y0;
			for (int k = 0; k < lv->dets[i].n_kpt &&
			                k < BC_MAX_KPT; k++) {
				float kx, ky;
				bc_preproc_tile_to_screen(lv->pp, t,
				                          lv->dets[i].kx[k],
				                          lv->dets[i].ky[k],
				                          &kx, &ky);
				lv->dets[i].kx[k] = kx;
				lv->dets[i].ky[k] = ky;
			}
		}

		if (bc_preproc_tiles(lv->pp) > 1) {
			float ox, oy;
			bc_preproc_overlap_px(lv->pp, &ox, &oy);
			const int before = n;
			n = bc_seam_merge(lv->dets, n, bc_preproc_cols(lv->pp),
			                  bc_preproc_full_tile(lv->pp),
			                  ox, oy, SEAM_THRESH,
			                  classes_similar);
			lv->seam_merges += (unsigned long)(before - n);
		}

		/* After the seam merge, so an eye cut by a tile edge is judged
		 * whole; before landmarks and association, so a dropped eye
		 * neither opens a track nor sustains one. */
		n = drop_eyes_under_bars(lv, lv->dets, n);

		/*
		 * Landmarks, once per detection set rather than once per
		 * displayed frame: the crops come from these boxes, and
		 * between detections the tracker carries the angle
		 * forward anyway. Runs here because the boxes are in
		 * screen space and the captured frame is still the one
		 * they were found in -- cropping a later frame would put
		 * the eyes where the face used to be.
		 */
		lv->n_lm = 0;
		if (lv->lm_det && lv->lm_pp && lv->lm_tensor) {
			for (int i = 0; i < n && lv->n_lm < BC_LM_MAX_FACES; i++) {
				if (!is_head_class(bc_class_name(lv->dets[i].cls)))
					continue;
				/*
				 * A head whose class is not censored draws
				 * nothing, so it earns no inference.
				 *
				 * Without this, a second detector finding
				 * the same face under its own name -- booru
				 * calls it HCAT where a NudeNet model calls
				 * it FACEFEMALE -- gets a landmark run of
				 * its own. The anime landmark model was not
				 * trained on that framing, comes back weak,
				 * and the bar falls back to a band across
				 * the whole box: a second, wrong bar over a
				 * face that already had a correct one, on a
				 * class the user had switched off.
				 *
				 * Costs ~29 ms of GPU a frame and produces
				 * a cover nobody asked for. Both go away by
				 * asking the question the draw path asks.
				 */
				if (lv->dets[i].cls < 0 ||
				    lv->dets[i].cls >= BC_MAX_CLASSES ||
				    !lv->cfg.censor[lv->dets[i].cls])
					continue;
				/* Pad as the model was trained, and keep
				 * the crop square: a square crop cannot
				 * letterbox, so no part of the input is
				 * spent on grey. */
				float cx = lv->dets[i].x + lv->dets[i].w * 0.5f;
				float cy = lv->dets[i].y + lv->dets[i].h * 0.5f;
				/*
				 * Sized by the box's *height*, not its
				 * longest side.
				 *
				 * The landmark model does not care which
				 * detector found the head -- it is handed a
				 * crop and returns a heatmap -- so what
				 * decides whether it works is how much of
				 * that crop the face fills. Different
				 * models draw very different head boxes for
				 * the identical subject: a face box is
				 * portrait and tight (452x477 measured),
				 * while a booru head box is landscape and
				 * takes in ears and hair (716x537). Sizing
				 * the square by the longer side made the
				 * second crop 50% wider than the first, the
				 * face too small in frame, and every peak
				 * too weak to accept -- so the bar fell back
				 * to a band across the whole head.
				 *
				 * Height is the dimension that tracks the
				 * face rather than the ears, so it brings
				 * both crops to nearly the same scale. It
				 * also leaves the already-working case
				 * exactly as it was: for a portrait face box
				 * the height *is* the longer side.
				 */
				float side = lv->dets[i].h * BC_LM_PAD;
				if (lv->lm_wide_crop) {
					side = (lv->dets[i].w > lv->dets[i].h
					        ? lv->dets[i].w : lv->dets[i].h)
					       * BC_LM_PAD;
				}
				if (!(side > 1.0f))
					continue;
				lv->lm_rects[lv->n_lm * 4 + 0] = cx - side * 0.5f;
				lv->lm_rects[lv->n_lm * 4 + 1] = cy - side * 0.5f;
				lv->lm_rects[lv->n_lm * 4 + 2] = side;
				lv->lm_rects[lv->n_lm * 4 + 3] = side;
				lv->lm_face_det[lv->n_lm] = i;
				lv->n_lm++;
			}
			if (lv->n_lm > 0 &&
			    bc_preproc_set_rects(lv->lm_pp, lv->lm_rects, lv->n_lm) == 0 &&
			    /* The capture's own dimensions, not the
			     * overlay's: the crop rectangles came from
			     * detections, and those are in the space
			     * bc_preproc_tile_to_screen() maps to, which
			     * is the captured texture. The two sizes are
			     * usually equal and silently need not be. */
			    bc_preproc_run(lv->lm_pp, bc_gpu_texture(lv->gpu),
			                   bc_gpu_width(lv->gpu),
			                   bc_gpu_height(lv->gpu)) == 0 &&
			    bc_preproc_readback(lv->lm_pp, lv->lm_tensor) == 0) {
				const int got = bc_landmark_run(lv->lm_det, lv->lm_tensor,
				                                lv->n_lm, lv->lm_out,
				                                BC_LM_MAX_FACES);
				lv->n_lm = got > 0 ? got : 0;
				if (lv->n_lm) {
					lv->lm_runs++;
					lv->lm_ms_sum += bc_landmark_last_ms(lv->lm_det);
				}
			} else {
				lv->n_lm = 0;
			}
		}
	}
	if (n >= 0) {
		/*
		 * Detections are converted to screen space, dilated and
		 * turned into eye bars *before* being associated, so the
		 * tracker sees the boxes we will actually draw. Staged
		 * here rather than matched one at a time: association has
		 * to consider all of them together to be one-to-one.
		 */
		struct bc_det sdets[BC_MAX_DET];
		struct track  smeta[BC_MAX_DET];
		int           det_track[BC_MAX_DET];
		int           ns = 0;
		for (int i = 0; i < n; i++) {
			if (lv->dets[i].cls >= 0 && lv->dets[i].cls < BC_MAX_CLASSES)
				lv->det_by_class[lv->dets[i].cls]++;

			if (lv->debug_all)
				LOG("  det %-26s %.2f  tile %d  screen(%.0f,%.0f %.0fx%.0f)%s",
				    bc_class_name(lv->dets[i].cls), lv->dets[i].score,
				    lv->dets[i].tile,
				    lv->dets[i].x, lv->dets[i].y, lv->dets[i].w, lv->dets[i].h,
				    lv->cfg.censor[lv->dets[i].cls] ? "  [CENSOR]" : "");
			/*
			 * Deliberately not `continue` here. A class we
			 * do not censor may still be the same physical
			 * thing as a track we do -- that is the point of
			 * classes_similar() -- so it has to reach the
			 * association, which is where the policy is
			 * applied: an uncensored class may sustain a
			 * track but never open one.
			 */

			/* Already screen pixels: converted above, before
			 * the seam merge needed them comparable. */
			float x0 = lv->dets[i].x, y0 = lv->dets[i].y;
			float w = lv->dets[i].w, h = lv->dets[i].h;
			float angle = 0.0f;
			/* Axis used for this bar, staged into smeta
			 * below so the track can remember it. */
			float axis_ux = 0.0f, axis_uy = 0.0f;
			int   have_axis_out = 0, axis_fresh = 0;

			/*
			 * Eye bar. A face box is stable but says nothing
			 * about orientation; the eye model is jittery but
			 * says exactly where the eyes are. Together they
			 * give a tight bar at any rotation: the line
			 * between two eye centres *is* the angle, which
			 * is the one thing an axis-aligned face box can
			 * never supply.
			 */
			float fx = x0, fy = y0, fw = w, fh = h;
			int   is_eyebar = 0;
			int   lm_bar_used = 0;
			int   has_bar = 0;
			float bar_cx_f = 0, bar_cy_f = 0;
			float bar_w_f = 0, bar_h_f = 0;
			if (lv->cfg.eye_bar &&
			    is_head_class(bc_class_name(lv->dets[i].cls))) {
				is_eyebar = 1;
				/*
				 * Best case: the face model has a Pose
				 * head and gave us the eyes directly.
				 * Keypoints are predicted jointly with
				 * the box, so unlike a second detector
				 * they cannot disagree with the face they
				 * belong to -- position and angle both
				 * come from one consistent prediction.
				 */
				int kpt_bar = 0;
				/*
				 * Where this face's eyes are, from the
				 * primary model if it has a pose head,
				 * otherwise from the opt-in pose model
				 * matched to this face by overlap.
				 *
				 * Matched rather than trusted on its own:
				 * the pose model's own boxes never become
				 * covers, so a face it invents where the
				 * primary model saw nothing changes
				 * nothing, and a face the primary model
				 * found without a match keeps the band.
				 * The censor policy stays entirely with
				 * the primary taxonomy.
				 */
				int   have_kp = 0;
				float kp_x0 = 0, kp_y0 = 0;
				float kp_x1 = 0, kp_y1 = 0;
				/* Eye extents in screen pixels when the
				 * landmark model supplied them; 0 means
				 * fall back to scaling by separation. */
				float lm_along = 0.0f, lm_across = 0.0f;
				/*
				 * The anime landmark model first, when it
				 * ran for this detection. It is the only
				 * source trained on drawn faces, and it
				 * gives six points per eye -- so its eye
				 * centre is an average, where every other
				 * source here is one point taken on trust.
				 */
				for (int m = 0; !lv->no_kpt && m < lv->n_lm; m++) {
					if (lv->lm_face_det[m] != i)
						continue;
					if (lv->lm_out[m].eye_score < BC_LM_MIN_SCORE) {
						lv->lm_weak++;
						break;
					}
					/* Crop pixels -> screen. The crop
					 * was tile m of lm_pp, so its own
					 * inverse maps it back. */
					float ax, ay, bx2, by2;
					bc_preproc_tile_to_screen(lv->lm_pp, m,
					        lv->lm_out[m].lx, lv->lm_out[m].ly,
					        &ax, &ay);
					bc_preproc_tile_to_screen(lv->lm_pp, m,
					        lv->lm_out[m].rx, lv->lm_out[m].ry,
					        &bx2, &by2);
					kp_x0 = ax;  kp_y0 = ay;
					kp_x1 = bx2; kp_y1 = by2;
					have_kp = 1;
					/*
					 * Size the bar from where the eye
					 * points actually reach, scaled
					 * into screen pixels by the same
					 * factor the crop used. Separation
					 * alone assumes photographic
					 * proportions: drawn eyes are much
					 * larger for the same spacing, and
					 * a turned head brings the centres
					 * together without shrinking the
					 * eyes, so a separation-scaled bar
					 * came out too small on exactly the
					 * rotated faces this exists for.
					 */
					{
						float px0, py0, px1, py1;
						bc_preproc_tile_to_screen(lv->lm_pp, m,
						        0.0f, 0.0f, &px0, &py0);
						bc_preproc_tile_to_screen(lv->lm_pp, m,
						        1.0f, 1.0f, &px1, &py1);
						const float k = px1 - px0;
						lm_along  = lv->lm_out[m].ext_along  * k;
						lm_across = lv->lm_out[m].ext_across * k;
					}
					lm_bar_used = 1;
					lv->lm_used++;
					/*
					 * BC_DEBUG_LM: where the eyes
					 * landed relative to the face
					 * box, as fractions of it. An
					 * eye should sit inside its own
					 * face, left of centre and right
					 * of centre respectively; values
					 * outside 0..1 mean the crop or
					 * the inverse mapping is wrong,
					 * which no amount of looking at
					 * the bar would distinguish from
					 * the model simply being poor.
					 */
					if (lv->debug_lm) {
						LOG("lm: face %.0f,%.0f %.0fx%.0f  "
						    "L(%.2f,%.2f) R(%.2f,%.2f) "
						    "score %.2f",
						    fx, fy, fw, fh,
						    fw > 0 ? (ax - fx) / fw : 0.0f,
						    fh > 0 ? (ay - fy) / fh : 0.0f,
						    fw > 0 ? (bx2 - fx) / fw : 0.0f,
						    fh > 0 ? (by2 - fy) / fh : 0.0f,
						    lv->lm_out[m].eye_score);
					}
					break;
				}
				if (!have_kp && !lv->no_kpt &&
				    lv->dets[i].n_kpt >= 2 &&
				    lv->dets[i].ks[0] > KPT_MIN_CONF &&
				    lv->dets[i].ks[1] > KPT_MIN_CONF) {
					have_kp = 1;
					kp_x0 = lv->dets[i].kx[0];
					kp_y0 = lv->dets[i].ky[0];
					kp_x1 = lv->dets[i].kx[1];
					kp_y1 = lv->dets[i].ky[1];
				} else if (!lv->no_kpt && lv->pose_mode &&
				           (lv->tnow - lv->pfaces_at) < EYE_FRESH_MS) {
					int   best = -1;
					float best_iou = POSE_MIN_IOU;
					for (int p = 0; p < lv->n_pfaces; p++) {
						float ix0 = fx > lv->pfaces[p].x0 ? fx : lv->pfaces[p].x0;
						float iy0 = fy > lv->pfaces[p].y0 ? fy : lv->pfaces[p].y0;
						float ix1 = fx + fw < lv->pfaces[p].x1 ? fx + fw : lv->pfaces[p].x1;
						float iy1 = fy + fh < lv->pfaces[p].y1 ? fy + fh : lv->pfaces[p].y1;
						if (ix1 <= ix0 || iy1 <= iy0)
							continue;
						float inter = (ix1 - ix0) * (iy1 - iy0);
						float ua = fw * fh +
						           (lv->pfaces[p].x1 - lv->pfaces[p].x0) *
						           (lv->pfaces[p].y1 - lv->pfaces[p].y0) - inter;
						float iou = ua > 0.0f ? inter / ua : 0.0f;
						if (iou > best_iou) {
							best_iou = iou;
							best = p;
						}
					}
					if (best >= 0 &&
					    lv->pfaces[best].es[0] > KPT_MIN_CONF &&
					    lv->pfaces[best].es[1] > KPT_MIN_CONF) {
						have_kp = 1;
						kp_x0 = lv->pfaces[best].ex[0];
						kp_y0 = lv->pfaces[best].ey[0];
						kp_x1 = lv->pfaces[best].ex[1];
						kp_y1 = lv->pfaces[best].ey[1];
						lv->pose_matched++;
					} else if (lv->n_pfaces > 0) {
						lv->pose_unmatched++;
					}
				}
				if (have_kp) {
					float ex0 = kp_x0;
					float ey0 = kp_y0;
					float ex1 = kp_x1;
					float ey1 = kp_y1;
					/*
					 * Confidence is the model's opinion of
					 * its own output and cannot catch it
					 * being confidently wrong -- which is
					 * what a WIDERFACE model does on anime,
					 * where it finds the box but regresses
					 * the landmarks somewhere else on the
					 * face. The geometry is checked instead:
					 * eyes belong inside their own face box,
					 * separated by a sane fraction of it.
					 * Failing that falls through to the
					 * body axis, which is the behaviour
					 * that was claimed for the confidence
					 * gate and never actually happened.
					 *
					 * The rules live in eyebar.c, shared
					 * with the offline passes: one copy of
					 * "which answers are believable".
					 */
					struct bc_eyebar eb;
					const int er = bc_eyebar_from_eyes(
					    ex0, ey0, ex1, ey1, lm_along, lm_across,
					    lv->lm_bar_pad, lv->lm_nofloor,
					    fx, fy, fw, fh, &eb);
					if (er < 0)
						lv->kpt_reject++;
					if (er > 0) {
						const float sep = hypotf(ex1 - ex0, ey1 - ey0);
						angle = eb.angle;
						x0 = eb.cx - eb.w * 0.5f;
						y0 = eb.cy - eb.h * 0.5f;
						w  = eb.w;
						h  = eb.h;
						kpt_bar = 1;
						lv->eyebar_kpt++;
						if (lv->debug_lm)
							LOG("lm: bar %.0f,%.0f %.0fx%.0f "
							    "eyemid %.0f,%.0f sep %.0f "
							    "ang %.0f",
							    x0, y0, w, h,
							    (ex0 + ex1) * 0.5f,
							    (ey0 + ey1) * 0.5f,
							    sep, angle * 57.29578f);
					}
				}

				int   found = 0;
				if (kpt_bar)
					goto eyebar_done;
				float ecx[BC_MAX_DET], ecy[BC_MAX_DET];
				float ehh = 0.0f, ehw = 0.0f;
				int fresh = (now_ms() - lv->eyes_at) < EYE_FRESH_MS;
				for (int e = 0; fresh && e < lv->n_eyes; e++) {
					float cx = (lv->eyes[e].x0 + lv->eyes[e].x1) * 0.5f;
					float cy = (lv->eyes[e].y0 + lv->eyes[e].y1) * 0.5f;
					/* Margin: a face box is tight, and an
					 * eye centre near its edge is still
					 * that face's eye. */
					float mx = w * 0.12f, my = h * 0.12f;
					if (cx < x0 - mx || cx > x0 + w + mx ||
					    cy < y0 - my || cy > y0 + h + my)
						continue;   /* not this face */
					ecx[found] = cx;
					ecy[found] = cy;
					float ew = lv->eyes[e].x1 - lv->eyes[e].x0;
					float eh = lv->eyes[e].y1 - lv->eyes[e].y0;
					if (ew * 0.5f > ehw) ehw = ew * 0.5f;
					if (eh * 0.5f > ehh) ehh = eh * 0.5f;
					found++;
				}

				if (found >= 2) {
					/* The two furthest apart are the
					 * outer pair; a third detection is
					 * usually a duplicate between them. */
					int a = 0, b2 = 1;
					float best = -1.0f;
					for (int p = 0; p < found; p++)
						for (int q = p + 1; q < found; q++) {
							float dx = ecx[q] - ecx[p];
							float dy = ecy[q] - ecy[p];
							float d2 = dx * dx + dy * dy;
							if (d2 > best) {
								best = d2;
								a = p; b2 = q;
							}
						}
					float dx = ecx[b2] - ecx[a];
					float dy = ecy[b2] - ecy[a];
					float len = sqrtf(dx * dx + dy * dy);
					/* screen y grows downward, GL's grows
					 * up, so negate to get a CCW angle */
					angle = atan2f(-dy, dx);
					float barw = len + 2.0f * ehw * (1.0f + EYE_PAD);
					float barh = 2.0f * ehh * (1.0f + EYE_PAD);
					float cx = (ecx[a] + ecx[b2]) * 0.5f;
					float cy = (ecy[a] + ecy[b2]) * 0.5f;
					x0 = cx - barw * 0.5f;
					y0 = cy - barh * 0.5f;
					w  = barw;
					h  = barh;
					lv->eyebars++;
				} else if (found == 1) {
					/* One eye: position is known, angle
					 * is not. Cover it upright and a
					 * little wide. */
					float barw = 2.0f * ehw * (1.0f + EYE_PAD);
					float barh = 2.0f * ehh * (1.0f + EYE_PAD);
					x0 = ecx[0] - barw * 0.5f;
					y0 = ecy[0] - barh * 0.5f;
					w  = barw;
					h  = barh;
					lv->eyebars++;
				} else {
					/*
					 * No eye positions: place the band
					 * geometrically inside the face box.
					 * If another part of the same body was
					 * detected it supplies the up-axis, so
					 * the band can be turned to match a
					 * subject lying down or inverted -- the
					 * case a face box alone cannot see.
					 */
					float ux, uy;
					/*
					 * This frame's axis if a second
					 * part of the body was detected,
					 * otherwise the one this face had
					 * a moment ago. The remembered arm
					 * is what stops the bar flipping
					 * to the upright band and back as
					 * the torso comes and goes.
					 */
					int fresh_axis = lv->axis_bar && body_axis(
					        lv->dets, n, i, lv->pp,
					        x0 + w * 0.5f,
					        y0 + h * 0.5f, h,
					        &ux, &uy);
					int got_axis = fresh_axis;
					if (lv->axis_bar && !got_axis &&
					    remembered_axis(lv->meta, BC_TRK_MAX,
					                    x0 + w * 0.5f,
					                    y0 + h * 0.5f, h,
					                    lv->tnow, &ux, &uy)) {
						got_axis = 1;
						lv->eyebar_axis_mem++;
					}
					if (got_axis) {
						axis_ux = ux;
						axis_uy = uy;
						axis_fresh = fresh_axis;
						have_axis_out = 1;
						/* bar runs across the body axis;
						 * screen y grows down, GL's up */
						angle = wrap_bar(atan2f(-uy, ux)
						                 - 1.57079633f);
						/*
						 * Recover the face's own extents
						 * from the axis-aligned box:
						 *   bw = W|cos| + L|sin|
						 *   bh = W|sin| + L|cos|
						 * with W across the body and L
						 * along it. That inverts cleanly
						 * except near 45 degrees, where the
						 * two are nearly equal anyway.
						 */
						float c = fabsf(cosf(angle));
						float s = fabsf(sinf(angle));
						float dt = c * c - s * s;
						float L, W;
						if (fabsf(dt) < 0.30f) {
							L = W = (w + h) * 0.5f;
						} else {
							L = (c * h - s * w) / dt;
							W = (c * w - s * h) / dt;
							if (L < 1.0f) L = 1.0f;
							if (W < 1.0f) W = 1.0f;
						}
						float barw = W * (1.0f - 2.0f * EYEBAR_INSET);
						float barh = L * (lv->eyebar_bot - lv->eyebar_top);
						/* band centre sits above the face
						 * centre, along the body axis */
						float off = (0.5f - (lv->eyebar_top +
						                     lv->eyebar_bot) * 0.5f) * L;
						float cx = x0 + w * 0.5f + ux * off;
						float cy = y0 + h * 0.5f + uy * off;
						x0 = cx - barw * 0.5f;
						y0 = cy - barh * 0.5f;
						w  = barw;
						h  = barh;
						lv->eyebar_axis++;
					{
						double d = fabs(angle) * 57.29578;
						lv->axis_deg_sum += d;
						if (d > lv->axis_deg_max)
							lv->axis_deg_max = d;
						if (d > 30.0) lv->axis_over_30++;
					}
					} else {
						y0 += h * lv->eyebar_top;
						h  *= (lv->eyebar_bot - lv->eyebar_top);
						x0 += w * EYEBAR_INSET;
						w  *= (1.0f - 2.0f * EYEBAR_INSET);
					}
					lv->eyebar_fallback++;
				}
eyebar_done: ;
				/*
				 * Additive mode: the bar is drawn *as
				 * well as* the face, so the detection
				 * that goes to the tracker stays the
				 * face box and the bar is recorded
				 * against it. Recorded as fractions so
				 * it follows the tracked face between
				 * detections -- the face is the stable
				 * thing here, and giving the bar its own
				 * track would mean two filters
				 * disagreeing about one object.
				 */
				/*
				 * Keep the face as the tracked box, and
				 * record the bar against it, only when
				 * the face is itself being covered. With
				 * eye_bar_face off the bar *is* the
				 * cover, so the box handed on below stays
				 * the band and nothing is drawn over the
				 * rest of the face -- which is the case
				 * that could not be asked for while this
				 * was gated on eye_bar_effect.
				 */
				if (lv->cfg.eye_bar_face &&
				    fw > 0.0f && fh > 0.0f) {
					bar_cx_f = (x0 + w * 0.5f - fx) / fw;
					bar_cy_f = (y0 + h * 0.5f - fy) / fh;
					bar_w_f  = w / fw;
					bar_h_f  = h / fh;
					has_bar  = 1;
					/* Hand the face back to the
					 * tracker, not the bar. */
					x0 = fx; y0 = fy;
					w  = fw; h  = fh;
				}
			}

			/* §2: dilate to cover motion since the frame was grabbed */
			float dil = lv->cfg.dilate / 100.0f;
			x0 -= w * dil * 0.5f;
			y0 -= h * dil * 0.5f;
			w  *= 1.0f + dil;
			h  *= 1.0f + dil;

			/*
			 * Stage it. Everything that made this box what it
			 * is -- screen space, dilate, the eye bar and its
			 * angle -- has happened; association is next, and
			 * needs the whole frame's worth at once.
			 */
			if (ns < BC_MAX_DET) {
				sdets[ns] = lv->dets[i];
				sdets[ns].x = x0;
				sdets[ns].y = y0;
				sdets[ns].w = w;
				sdets[ns].h = h;
				smeta[ns] = (struct track){ 0 };
				smeta[ns].angle  = angle;
				smeta[ns].eyebar = is_eyebar;
				smeta[ns].lm_bar = lm_bar_used;
				smeta[ns].has_bar  = has_bar;
				smeta[ns].bar_cx_f = bar_cx_f;
				smeta[ns].bar_cy_f = bar_cy_f;
				smeta[ns].bar_w_f  = bar_w_f;
				smeta[ns].bar_h_f  = bar_h_f;
				/* The face is what is tracked in additive
				 * mode, so it must not also be clamped as
				 * though it were a bar. */
				if (has_bar)
					smeta[ns].eyebar = 0;
				smeta[ns].fx = fx;
				smeta[ns].fy = fy;
				smeta[ns].fw = fw;
				smeta[ns].fh = fh;
				smeta[ns].ux = axis_ux;
				smeta[ns].uy = axis_uy;
				smeta[ns].have_axis = have_axis_out;
				/*
				 * Only a freshly measured axis restamps
				 * the clock. Letting a remembered one do
				 * it would let the memory refresh itself
				 * every frame and never expire -- a face
				 * that turned would keep its old axis for
				 * as long as it stayed on screen.
				 */
				smeta[ns].axis_at = axis_fresh ? lv->tnow : 0.0;
				ns++;
			}
		}
		/*
		 * One association for the whole frame. Two passes inside:
		 * confident detections claim their tracks first, then the
		 * weak ones take what is left -- which is what the score
		 * hysteresis here used to approximate.
		 */
		/*
		 * A track has to outlive the gap between inferences, or
		 * it expires waiting for the detection that would have
		 * renewed it.
		 *
		 * sticky_ms is a fixed 400 ms, which is fine at the rate
		 * inference runs on moving content. On a *static* screen
		 * the change gate suppresses almost everything -- 98.7%
		 * of frames identical -- and the only inferences left are
		 * the ~1.8/s the hold cap forces, so they arrive ~550 ms
		 * apart. Any object a single inference misses is then
		 * already past its 400 ms sticky window before the next
		 * one could confirm it, and the bar drops. Measured that
		 * way: box count swinging 5 -> 2 -> 1 -> 2 -> ... -> 5 on
		 * a screen that was not changing at all.
		 *
		 * So the window has a floor of a few inference intervals,
		 * measured rather than assumed. On moving content the gap
		 * is small and cfg.sticky_ms still wins, so this changes
		 * nothing there; it only stretches the memory in exactly
		 * the case that starved it.
		 */
		static double last_det_t, det_gap_ms;
		if (last_det_t > 0.0) {
			const double gap = lv->tnow - last_det_t;
			det_gap_ms = det_gap_ms > 0.0
			           ? det_gap_ms * 0.8 + gap * 0.2
			           : gap;
		}
		last_det_t = lv->tnow;

		const double gap_floor = det_gap_ms * 2.5;
		lv->tracker.memory_ms = lv->cfg.sticky_ms > gap_floor
		                  ? lv->cfg.sticky_ms : gap_floor;

		bc_tracker_update(&lv->tracker, sdets, ns, lv->tnow, lv->cfg.censor,
		                  det_track);

		/*
		 * A clean sample -- taken while the screen was
		 * uncovered -- is the only evidence that may retire a
		 * cover, and two consecutive ones are required. The
		 * detector misses; uncovering on a single blink is the
		 * failure this whole program exists to prevent, and the
		 * cost of the second sample is at most another half
		 * second of a cover that had nothing left under it.
		 *
		 * Anything less than two, and the track is held
		 * outright: it has not been disproved, and under a
		 * captured overlay silence is not disproof.
		 */
		if (lv->peek_enabled && !lv->no_hold) {
			int matched[BC_TRK_MAX] = { 0 };
			for (int d = 0; d < ns; d++)
				if (det_track[d] >= 0 &&
				    det_track[d] < BC_TRK_MAX)
					matched[det_track[d]] = 1;

			for (int k = 0; k < BC_TRK_MAX; k++) {
				if (lv->tracker.t[k].state == BC_TRK_FREE) {
					lv->clean_miss[k] = 0;
					lv->first_area[k] = 0.0f;
					continue;
				}
				if (matched[k]) {
					lv->clean_miss[k] = 0;
				} else if (lv->submitted_clean) {
					if (lv->clean_miss[k] < 255)
						lv->clean_miss[k]++;
					if (lv->clean_miss[k] == 2)
						lv->clean_drops++;
				}
				if (lv->clean_miss[k] < 2)
					bc_trk_freeze(&lv->tracker.t[k], lv->tnow);
			}
			lv->submitted_clean = 0;
		}

		/*
		 * Carry each detection's eye-bar state across to the track
		 * it landed on. A track the tracker just created starts
		 * from nothing; one it matched keeps its eased angle, or a
		 * bar would twitch every time the eye pair jittered.
		 */
		for (int i = 0; i < ns; i++) {
			const int k = det_track[i];
			if (k < 0)
				continue;
			if (lv->tracker.t[k].hits <= 1) {
				lv->meta[k] = (struct track){ 0 };
				lv->meta[k].created = lv->tnow;
				lv->meta[k].angle   = smeta[i].angle;
				if (lv->trace)
					LOG("trace: track %d OPENED %s "
					    "%.0fx%.0f at %.0f,%.0f "
					    "score %.2f", k,
					    bc_class_name(sdets[i].cls),
					    sdets[i].w, sdets[i].h,
					    sdets[i].x, sdets[i].y,
					    sdets[i].score);
			} else {
				/* Shortest way round, half-turn period: a
				 * bar at +89 degrees and one at -89 are two
				 * degrees apart, not 178. */
				float da = wrap_bar(smeta[i].angle -
				                    lv->meta[k].angle);
				lv->meta[k].angle = wrap_bar(lv->meta[k].angle +
				                         0.35f * da);
			}
			lv->meta[k].eyebar = smeta[i].eyebar;
			lv->meta[k].lm_bar = smeta[i].lm_bar;
			lv->meta[k].has_bar  = smeta[i].has_bar;
			if (smeta[i].has_bar) {
				lv->meta[k].bar_cx_f = smeta[i].bar_cx_f;
				lv->meta[k].bar_cy_f = smeta[i].bar_cy_f;
				lv->meta[k].bar_w_f  = smeta[i].bar_w_f;
				lv->meta[k].bar_h_f  = smeta[i].bar_h_f;
			}
			if (smeta[i].eyebar) {
				lv->meta[k].fx = smeta[i].fx;
				lv->meta[k].fy = smeta[i].fy;
				lv->meta[k].fw = smeta[i].fw;
				lv->meta[k].fh = smeta[i].fh;
				/*
				 * Carried only when this detection
				 * measured one. A frame with no torso
				 * must leave the remembered axis alone,
				 * or the memory would be erased by
				 * exactly the case it exists for.
				 */
				if (smeta[i].axis_at > 0.0) {
					lv->meta[k].ux = smeta[i].ux;
					lv->meta[k].uy = smeta[i].uy;
					lv->meta[k].have_axis = 1;
					lv->meta[k].axis_at = smeta[i].axis_at;
				}
			}
			lv->meta[k].cls = sdets[i].cls;
			lv->meta[k].n_detect++;
			lv->meta[k].last_detect = lv->tnow;
		}

		lv->last_infer = lv->tnow;
		stat_add(&lv->s_infer, res.ms);
	}

	/*
	 * Errors do not themselves blank the output -- the deadline
	 * below does that, and it catches a hang as well as an error.
	 * They are logged because a detector erroring every run and a
	 * detector merely running slowly look identical from outside.
	 */
	/* At most once a second: a failing detector is retried every
	 * frame (see det_failing), which is ~146 lines a second. */
	if (res.status != BC_DET_PENDING)
		lv->det_failing = res.status == BC_DET_FAILED;
	if (res.status == BC_DET_FAILED) {
		static double warned_at;
		lv->det_errors++;
		if (lv->tnow - warned_at >= 1000.0) {
			warned_at = lv->tnow;
			WARN("detector errors: %lu", lv->det_errors);
		}
	}
}

/* Turn the tracks into covers. */
static void step_covers(struct bc_live *lv)
{
	/*
	 * 4. Expire stale tracks. Note the hold above has already
	 *    refreshed any track whose region is unchanged, so this only
	 *    retires boxes over content that has actually moved.
	 */
	/*
	 * The global effect, for the covers that are not about one
	 * class: the takeover fill, the reverse mask, the test box.
	 * Per-track boxes take theirs from the class instead, below.
	 */
	lv->effect = overlay_effect(lv->cfg.effect);
	lv->strength = lv->cfg.strength;

	/*
	 * How capture pixels map onto the overlay's buffer this frame.
	 * Recomputed per frame rather than cached: the output can be
	 * rescaled or the capture renegotiated while running, and a
	 * stale factor here misplaces every cover on screen.
	 *
	 * 1.0 until the first frame has been imported, when the capture
	 * size is not known yet -- there is nothing to draw then
	 * either, so the identity is both harmless and the honest
	 * answer.
	 */
	const float draw_sx = draw_scale(bc_gpu_width(lv->gpu),
	                                 bc_overlay_width(lv->ov));
	const float draw_sy = draw_scale(bc_gpu_height(lv->gpu),
	                                 bc_overlay_height(lv->ov));
	/*
	 * Did this frame come from a screen with nothing of ours on
	 * it? Only frames composited after the peek began qualify, and
	 * only those may later justify dropping a cover.
	 */
	if (lv->peeking) {
		const int64_t pts = bc_capture_frame_pts(lv->cap);
		/*
		 * No header, no timestamp. Only ever verified on KWin,
		 * and a portal that omits it made every frame read as
		 * "not clean" -- which, with the gate below, meant no
		 * inference ever ran and nothing at all was drawn.
		 *
		 * Without a composite time the next best evidence is
		 * arrival: a frame that turns up at least one refresh
		 * after the screen was cleared is very probably a
		 * picture of the cleared screen. Weaker, and enough.
		 */
		const int have_pts = pts > 0;
		const double waited = now_ms() - lv->peek_started;
		if (have_pts ? (pts > lv->peek_start_ns) : (waited > 16.0)) {
			lv->clean_pending = 1;
			lv->clean_frames++;
			lv->last_clean_ms = now_ms();
			lv->peeking = 0;
		} else if (now_ms() - lv->peek_started > PEEK_MAX_MS) {
			/* No clean frame arrived in time. Cover the
			 * screen again and try later: an uncovered
			 * screen is never the thing to wait on. */
			lv->peeking = 0;
		}
	}

	if (!lv->logged_draw_scale && bc_gpu_width(lv->gpu) > 0) {
		lv->logged_draw_scale = 1;
		LOG("drawing: capture %dx%d -> overlay %dx%d "
		    "(scale %.4f x %.4f)%s",
		    bc_gpu_width(lv->gpu), bc_gpu_height(lv->gpu),
		    bc_overlay_width(lv->ov), bc_overlay_height(lv->ov),
		    draw_sx, draw_sy,
		    (draw_sx == 1.0f && draw_sy == 1.0f)
		        ? "" : "  <- fractional/HiDPI output");

		/*
		 * Does the stream even describe this screen?
		 *
		 * The portal picks the monitor -- the user chooses it
		 * in the consent dialog -- and the overlay picks its
		 * own. Nothing connects the two, so they can be
		 * different monitors, and then this covers a screen it
		 * cannot see while the one it is watching stays bare.
		 * A size that does not match the output's mode is the
		 * only evidence available from in here.
		 *
		 * A warning and not a blank: a compositor is allowed to
		 * hand out a scaled-down stream of the right monitor,
		 * which would look identical to this test, and blanking
		 * the screen over a legal negotiation is worse than
		 * saying so loudly. The refusal to guess at startup is
		 * what actually prevents the mismatch.
		 */
		int om_w = 0, om_h = 0;
		bc_overlay_output_mode(lv->ov, &om_w, &om_h);
		if (om_w > 0 && om_h > 0 &&
		    (bc_gpu_width(lv->gpu) != om_w ||
		     bc_gpu_height(lv->gpu) != om_h)) {
			WARN("the capture is %dx%d but output '%s' is "
			     "%dx%d", bc_gpu_width(lv->gpu),
			     bc_gpu_height(lv->gpu),
			     bc_overlay_output_name(lv->ov), om_w, om_h);
			WARN("if those are different monitors, this is "
			     "drawing covers on a screen it is not "
			     "watching -- pick the same one in the "
			     "portal dialog, or name it: "
			     "betacensor live %s",
			     bc_overlay_output_name(lv->ov));
		}
	}

	lv->n_boxes = 0;
	lv->frame_cover = 0.0f;
	for (int k = 0; k < BC_TRK_MAX; k++) {
		float bx, by, bw, bh;
		if (!bc_trk_box(&lv->tracker.t[k], &bx, &by, &bw, &bh))
			continue;

		/*
		 * The filter's estimate, eased asymmetrically per edge:
		 * an edge that needs to move outward jumps there
		 * immediately, an edge moving inward eases.
		 *
		 * The Kalman filter already removes the jitter this used
		 * to fight, and it does so symmetrically -- which on its
		 * own would let the box lag *behind* content moving out
		 * from under it. That is the one failure this program
		 * cannot have. Easing only inward means the box is at
		 * worst momentarily too large, never too small, so the
		 * filter smooths and this keeps it safe.
		 */
		if (!lv->meta[k].used || lv->no_smooth) {
			lv->meta[k].x = bx; lv->meta[k].y = by;
			lv->meta[k].w = bw; lv->meta[k].h = bh;
			lv->meta[k].used = 1;
		} else {
			float el[4] = { lv->meta[k].x, lv->meta[k].y,
			                lv->meta[k].x + lv->meta[k].w,
			                lv->meta[k].y + lv->meta[k].h };
			const float nl[4] = { bx, by, bx + bw, by + bh };
			for (int e = 0; e < 4; e++) {
				const int outward = (e < 2) ? (nl[e] < el[e])
				                            : (nl[e] > el[e]);
				const float pa = lv->alpha_env ? lv->alpha_env
				                           : lv->cfg.smoothing;
				el[e] = outward ? nl[e]
				                : el[e] + pa * (nl[e] - el[e]);
			}
			const float ox = lv->meta[k].x, oy = lv->meta[k].y;
			const float ow = lv->meta[k].w, oh = lv->meta[k].h;
			lv->meta[k].x = el[0];
			lv->meta[k].y = el[1];
			lv->meta[k].w = el[2] - el[0];
			lv->meta[k].h = el[3] - el[1];
			stat_add(&lv->s_jitter,
			         fabsf(lv->meta[k].x - ox) + fabsf(lv->meta[k].y - oy) +
			         fabsf(lv->meta[k].w - ow) + fabsf(lv->meta[k].h - oh));
		}
		bx = lv->meta[k].x; by = lv->meta[k].y;
		bw = lv->meta[k].w; bh = lv->meta[k].h;

		/*
		 * "growing": a cover creeps outward the longer the thing
		 * under it has been on screen. Applied after smoothing
		 * so the ease cannot undo it, and about the centre so it
		 * grows evenly rather than drifting one way.
		 *
		 * Only ever enlarges, so it cannot uncover -- which is
		 * also why it is applied here rather than to the track:
		 * the tracker's boxes stay honest measurements and this
		 * is purely what gets drawn over them.
		 */
		if (lv->cfg.growth > 0.0f && lv->tracker.t[k].born > 0.0) {
			float secs = (float)((now_ms() - lv->tracker.t[k].born)
			                     / 1000.0);
			if (secs < 0.0f)
				secs = 0.0f;
			float pct = lv->cfg.growth * secs;
			if (pct > lv->cfg.growth_max)
				pct = lv->cfg.growth_max;
			const float gx = bw * pct / 200.0f;
			const float gy = bh * pct / 200.0f;
			bx -= gx; by -= gy;
			bw += gx * 2.0f; bh += gy * 2.0f;
		}

		if (lv->tracker.t[k].state == BC_TRK_LOST)
			lv->pred_frames++;

		/*
		 * Ramp away over the tail of the sticky window. Full
		 * strength until the last FADE_OUT_MS, so the covered
		 * area and duration are unchanged -- only the final
		 * step is smoothed.
		 */
		float alpha = 1.0f;
		if (!lv->no_fade) {
			/*
			 * The fade must fit inside the sticky window
			 * with room to spare. sticky_ms defaults to
			 * 100 ms, so a flat 120 ms fade would put every
			 * box in permanent partial transparency from
			 * the moment it appeared -- content visible
			 * through every bar, all the time. Cap it at
			 * half the window so a box is always at full
			 * strength for at least as long as it fades.
			 */
			/*
			 * Against tracker.memory_ms, not cfg.sticky_ms.
			 * Those are not the same number: the detection
			 * gap floor raises the memory above a short
			 * configured window, and the tracker expires on
			 * the floored value. Fading on the config value
			 * meant the two read different clocks, and the
			 * fade always lost -- at sticky_ms 20 with a
			 * floor of ~227, a box hit alpha 0 twenty ms
			 * after its last match and stayed invisible for
			 * the remaining ~200 ms the track was alive,
			 * blinking back only when a detection landed.
			 * With detections ~100 ms apart while the mouse
			 * moves, that is a box that is transparent more
			 * often than not: the "crazy flickering" at low
			 * sticky values, which no box-count or
			 * transition counter could see, because the
			 * track never died and the count never moved.
			 */
			const double life = lv->tracker.memory_ms;
			double fade_ms = FADE_OUT_MS;
			if (fade_ms > life * 0.5)
				fade_ms = life * 0.5;
			double left = life -
			              (lv->tnow - lv->tracker.t[k].last_update);
			if (fade_ms > 0.0 && left < fade_ms) {
				alpha = (float)(left / fade_ms);
				if (alpha < 0.0f) alpha = 0.0f;
				if (alpha > 1.0f) alpha = 1.0f;
				if (alpha < 1.0f)
					lv->fade_frames++;
			}
		}

		/*
		 * Hard bound: an eye bar may never leave its face. This
		 * is applied last, after dilate, prediction and easing,
		 * because each of those grows the box independently and
		 * only the final geometry can be checked. Shrink first,
		 * then re-centre -- clamping the centre alone would slide
		 * an oversized bar around inside the face instead of
		 * making it fit.
		 */
		if (lv->meta[k].eyebar && !lv->meta[k].lm_bar && lv->meta[k].fw > 0.0f) {
			float hw = bw * 0.5f, hh = bh * 0.5f;
			float ca = fabsf(cosf(lv->meta[k].angle));
			float sa = fabsf(sinf(lv->meta[k].angle));
			float cx = bx + hw, cy = by + hh;
			for (int pass = 0; pass < 2; pass++) {
				float ex = hw * ca + hh * sa;
				float ey = hw * sa + hh * ca;
				float mx = lv->meta[k].fw * 0.5f;
				float my = lv->meta[k].fh * 0.5f;
				float sc = 1.0f;
				if (ex > mx) sc = mx / ex;
				if (ey > my && my / ey < sc) sc = my / ey;
				if (sc < 1.0f) { hw *= sc; hh *= sc; }
			}
			float ex = hw * ca + hh * sa;
			float ey = hw * sa + hh * ca;
			if (cx < lv->meta[k].fx + ex) cx = lv->meta[k].fx + ex;
			if (cx > lv->meta[k].fx + lv->meta[k].fw - ex)
				cx = lv->meta[k].fx + lv->meta[k].fw - ex;
			if (cy < lv->meta[k].fy + ey) cy = lv->meta[k].fy + ey;
			if (cy > lv->meta[k].fy + lv->meta[k].fh - ey)
				cy = lv->meta[k].fy + lv->meta[k].fh - ey;
			bx = cx - hw; by = cy - hh;
			bw = hw * 2.0f; bh = hh * 2.0f;
		}

		/*
		 * This box's own effect, from the class that opened the
		 * track. Strength comes from the per-effect table rather
		 * than cfg.strength, which is only the *global* effect's
		 * figure -- a face pinned to blur wants the blur radius,
		 * not whatever number the pixelate slider is showing.
		 */
		/*
		 * Room for this track's cover and the bar that may
		 * follow it. BC_MAX_BOXES is sized so this cannot fire
		 * -- it is here because the bar push below has always
		 * been bounds-checked and this one was not, which is
		 * the wrong way round: the cover is the box that always
		 * gets written, so it was the one that ran off the end.
		 */
		if (lv->n_boxes >= BC_MAX_BOXES)
			break;

		/*
		 * Capture pixels are not overlay pixels.
		 *
		 * Everything above this line -- the tracker, the
		 * easing, growth, the clamp -- works in the coordinates
		 * of the captured frame, because that is what the
		 * detector saw. The overlay draws in its *buffer*,
		 * which is the surface's logical size times its buffer
		 * scale, and the two are only the same number by
		 * coincidence: they are equal on an unscaled display
		 * whose capture matches its mode, which is the machine
		 * this was written on and is why nothing converted.
		 *
		 * They come apart the moment a compositor uses
		 * fractional scaling. Reported from niri at 2560x1600
		 * with a 1.2 scale: logical 2133x1333, buffer 4266x2666
		 * against a 2560x1600 capture, so every box was drawn
		 * at 0.6x of where it belonged -- up and to the left,
		 * and further out the further from the origin.
		 *
		 * meta[] deliberately keeps the capture-space values:
		 * it is the smoothing memory, compared against the
		 * tracker's next output, and converting it would apply
		 * the scale again on every frame.
		 */
		/* Before the draw scale, so the ratio is in one space. */
		{
			const float area = bw * bh;
			if (lv->first_area[k] <= 0.0f && area > 0.0f)
				lv->first_area[k] = area;
			if (lv->first_area[k] > 0.0f) {
				const float g = area / lv->first_area[k];
				if (g > lv->max_box_growth)
					lv->max_box_growth = g;
			}
			lv->frame_cover += area;
		}

		bx *= draw_sx; bw *= draw_sx;
		by *= draw_sy; bh *= draw_sy;

		int   b_cfg_eff = bc_cfg_class_effect(&lv->cfg, lv->tracker.t[k].cls);
		/*
		 * With the face left alone, this track *is* the eye
		 * band -- so eye_bar_effect names the effect for this
		 * box rather than for a second one drawn on top. That
		 * is what decouples the bar from the face's class
		 * effect: previously the only box in this mode was the
		 * band and it had no way to be told apart from the
		 * class it came from.
		 */
		if (lv->meta[k].eyebar) {
			lv->eyebar_solo++;
			if (lv->cfg.eye_bar_effect >= 0)
				b_cfg_eff = lv->cfg.eye_bar_effect;
		}
		int   b_effect  = overlay_effect(b_cfg_eff);
		float b_strength = lv->cfg.strength_by[b_cfg_eff];

		/*
		 * Reverse mode, per class. A class that names its own
		 * effect is covered with it; one that just follows the
		 * global effect is revealed, as every detection was
		 * before this existed.
		 *
		 * Reusing the pin rather than adding a second setting
		 * was the user's call, made knowing it changes what an
		 * existing pinned class does in reverse -- it used to
		 * be a hole and is now a cover. Outside reverse mode
		 * the flag means nothing and nothing changes: an `over`
		 * box is already in draw order there.
		 *
		 * This is the same mechanism the eye bar uses, which is
		 * why a covered face and a barred one compose: the face
		 * is drawn from here and the bar lands on top of it.
		 */
		const int b_over = lv->cfg.reverse &&
		    bc_cfg_class_pinned(&lv->cfg, lv->tracker.t[k].cls);

		lv->boxes[lv->n_boxes++] = (struct bc_box){
			.over = b_over,
			/*
			 * Clip to the silhouette, once one exists. Only
			 * the class covers -- the eye bar below stays a
			 * full rectangle, because it is placed from
			 * landmarks inside a face and a person mask has
			 * nothing useful to say about where an eye ends.
			 */
			/* Never while outlines own the mask -- that
			 * texture is a shape the user drew, and
			 * clipping a detection to it would hide the
			 * cover somewhere else on the screen. */
			.use_mask = lv->seg_have && !lv->lasso_have,
			.x = (int)bx, .y = (int)by,
			.w = (int)bw, .h = (int)bh,
			/* fill colour is the fail-closed fallback, used
			 * only until the first frame has been captured */
			.r = lv->cfg.censor_rgb[0], .g = lv->cfg.censor_rgb[1],
			.b = lv->cfg.censor_rgb[2], .a = alpha,
			.effect = b_effect, .strength = b_strength,
			.outline = lv->cfg.outline,
			.outline_rgb = { lv->cfg.outline_rgb[0],
			                 lv->cfg.outline_rgb[1],
			                 lv->cfg.outline_rgb[2] },
			.shadow = lv->cfg.shadow,
			.shadow_a = lv->cfg.shadow_opacity,
			.auto_color = lv->cfg.censor_auto,
			.angle = lv->meta[k].angle,
			.text_mode = lv->cfg.text_mode,
			.text_rgb = { lv->cfg.text_rgb[0], lv->cfg.text_rgb[1],
			              lv->cfg.text_rgb[2] }
		};
		/*
		 * Caption text. The configured string wins; with none
		 * set, each box is labelled with the class that opened
		 * it, which is more useful than a generic word and
		 * doubles as a live read-out of what the detector
		 * thinks it is looking at.
		 */
		{
			const char *txt = lv->cfg.caption[0]
			    ? lv->cfg.caption
			    : bc_class_name(lv->tracker.t[k].cls);
			snprintf(lv->boxes[lv->n_boxes - 1].text,
			         sizeof lv->boxes[lv->n_boxes - 1].text,
			         "%s", txt ? txt : "");
		}

		/*
		 * The eye bar, on top of the face that was just
		 * emitted. Immediately after it, because the overlay
		 * draws in array order and a bar drawn before its face
		 * would be painted over by it.
		 *
		 * Geometry comes back from fractions of the *current*
		 * tracked box, so the bar follows the face through
		 * prediction and easing without a filter of its own.
		 */
		if (lv->meta[k].has_bar && lv->n_boxes < BC_MAX_BOXES) {
			const float cx = bx + lv->meta[k].bar_cx_f * bw;
			const float cy = by + lv->meta[k].bar_cy_f * bh;
			const float w2 = lv->meta[k].bar_w_f * bw;
			const float h2 = lv->meta[k].bar_h_f * bh;
			/* The bar's own effect if it was given one,
			 * otherwise the face's -- which draws a bar
			 * indistinguishable from the face cover, and is
			 * the honest reading of "no effect of its own". */
			const int   be = lv->cfg.eye_bar_effect >= 0
			               ? lv->cfg.eye_bar_effect : b_cfg_eff;
			lv->boxes[lv->n_boxes++] = (struct bc_box){
				.x = (int)(cx - w2 * 0.5f),
				.y = (int)(cy - h2 * 0.5f),
				.w = (int)w2, .h = (int)h2,
				.r = lv->cfg.censor_rgb[0],
				.g = lv->cfg.censor_rgb[1],
				.b = lv->cfg.censor_rgb[2],
				.a = alpha,
				.effect = overlay_effect(be),
				.strength = lv->cfg.strength_by[be],
				.auto_color = lv->cfg.censor_auto,
				/* A cover in both modes. In reverse the
				 * face is revealed and this still has to
				 * hide the eyes, so it must never become
				 * one of the holes. */
				.over = 1,
				.angle = lv->meta[k].angle,
				.text_mode = lv->cfg.text_mode,
				.text_rgb = { lv->cfg.text_rgb[0],
				              lv->cfg.text_rgb[1],
				              lv->cfg.text_rgb[2] }
			};
			/* No outline or shadow: those are drawn
			 * outward, and outward from a bar sitting on a
			 * face is over the face's own cover. */
			const char *btxt = lv->cfg.caption[0]
			    ? lv->cfg.caption : "";
			snprintf(lv->boxes[lv->n_boxes - 1].text,
			         sizeof lv->boxes[lv->n_boxes - 1].text,
			         "%s", btxt);
			lv->eyebar_over++;
		}
	}

	/*
	 * BC_TEST_BOX="x,y,w,h" pins a box at a fixed spot regardless of
	 * detections, so the effect can be inspected at a known location.
	 */
	if (lv->n_boxes != lv->last_n_boxes) {
		if (lv->last_n_boxes >= 0)
			lv->boxcount_changes++;
		lv->last_n_boxes = lv->n_boxes;
	}
}

/* Add the fixed regions, the drawing cursor and the test box. */
static void step_regions(struct bc_live *lv)
{
	/*
	 * Hand-drawn regions.
	 *
	 * Unconditional: they are not detections and there is nothing to
	 * be confident about, which is the point of them. Detection has
	 * a floor, and a subject scoring under it is uncovered with
	 * nothing in the pipeline able to know -- these are the backstop
	 * for exactly that, so they do not consult the tracker, the
	 * confidence, or any model.
	 *
	 * Added after the detections so they draw on top, and before the
	 * panic and pause paths below, which clear every box: a manual
	 * region is still a censor bar, and "show me the raw screen"
	 * has to mean all of it.
	 *
	 * Never clipped to a silhouette either -- the user drew the
	 * shape, so there is nothing for a model to improve on.
	 */
	for (int i = 0; i < lv->cfg.n_regions && lv->n_boxes < BC_MAX_BOXES; i++) {
		const struct bc_region *r = &lv->cfg.regions[i];
		const int ow = bc_overlay_width(lv->ov);
		const int oh = bc_overlay_height(lv->ov);
		const int rx = (int)(r->x * ow + 0.5f);
		const int ry = (int)(r->y * oh + 0.5f);
		const int rw = (int)(r->w * ow + 0.5f);
		const int rh = (int)(r->h * oh + 0.5f);
		if (rw <= 0 || rh <= 0)
			continue;
		const int re = (r->effect >= 0 && r->effect < BC_CFG_LAST_)
		             ? r->effect : lv->cfg.effect;
		lv->boxes[lv->n_boxes++] = (struct bc_box){
			.x = rx, .y = ry, .w = rw, .h = rh,
			.r = lv->cfg.censor_rgb[0], .g = lv->cfg.censor_rgb[1],
			.b = lv->cfg.censor_rgb[2], .a = 1.0f,
			.effect = overlay_effect(re),
			.strength = lv->cfg.strength_by[re],
			.outline = lv->cfg.outline,
			.outline_rgb = { lv->cfg.outline_rgb[0],
			                 lv->cfg.outline_rgb[1],
			                 lv->cfg.outline_rgb[2] },
			.shadow = lv->cfg.shadow,
			.shadow_a = lv->cfg.shadow_opacity,
			.auto_color = lv->cfg.censor_auto,
			/* A cover in reverse mode too. Revealing a region
			 * the user drew a bar over would be the opposite
			 * of what drawing it meant. */
			.over = lv->cfg.reverse,
			/*
			 * An outline covers its bounding box clipped to
			 * the mask; a rectangle covers the box itself.
			 * If the mask never got built the clip is off
			 * and this is the whole box either way, which is
			 * the direction to fail in.
			 */
			.use_mask = r->n_pts >= 3 && lv->lasso_have,
			.text_mode = lv->cfg.text_mode,
			.text_rgb = { lv->cfg.text_rgb[0], lv->cfg.text_rgb[1],
			              lv->cfg.text_rgb[2] }
		};
		snprintf(lv->boxes[lv->n_boxes - 1].text,
		         sizeof lv->boxes[lv->n_boxes - 1].text, "%s",
		         lv->cfg.caption[0] ? lv->cfg.caption : "CENSORED");
		lv->regions_drawn++;
	}

	/*
	 * Drawing-mode cursor and rubber band.
	 *
	 * Drawn by this pass, into the same surface as everything else,
	 * for the reason given in overlay.h: a separate window would
	 * have to stack somewhere relative to the censors, and above
	 * them it would show the screen it is aimed at uncensored for as
	 * long as the user spent aiming.
	 *
	 * After the regions so the band sits on top of what is already
	 * covered, and still before the panic and pause paths -- those
	 * clear every box, and the mode is left on the same frame.
	 */
	if (lv->drawing) {
		/*
		 * The stroke itself is not a box -- it is hundreds of
		 * points, and a box apiece would exhaust the array
		 * several times over. It is drawn instead as one
		 * full-screen box clipped to the mask, which already
		 * holds the line (and its fill, once the loop closes).
		 */
		if (lv->n_path > 0 && lv->lasso_have && lv->n_boxes < BC_MAX_BOXES) {
			/*
			 * Bounded to the stroke, not the screen.
			 *
			 * This box only shows the line because the mask
			 * clips it -- and the overlay drops the clip
			 * when no mask texture is bound, on the rule
			 * that an absent mask means cover everything.
			 * That rule is right for a detection cover and
			 * catastrophic here: a full-screen box taking
			 * it covers the whole display with the current
			 * effect. Sizing the box to the stroke makes
			 * the failure proportionate -- a smear around
			 * what was being drawn, not a blackout.
			 */
			float sx0 = lv->path_x[0], sy0 = lv->path_y[0];
			float sx1 = lv->path_x[0], sy1 = lv->path_y[0];
			for (int i = 1; i < lv->n_path; i++) {
				if (lv->path_x[i] < sx0) sx0 = lv->path_x[i];
				if (lv->path_y[i] < sy0) sy0 = lv->path_y[i];
				if (lv->path_x[i] > sx1) sx1 = lv->path_x[i];
				if (lv->path_y[i] > sy1) sy1 = lv->path_y[i];
			}
			const int pad = 8;   /* room for the line's width */
			int bx0 = (int)sx0 - pad, by0 = (int)sy0 - pad;
			int bx1 = (int)sx1 + pad, by1 = (int)sy1 + pad;
			const int ow3 = bc_overlay_width(lv->ov);
			const int oh3 = bc_overlay_height(lv->ov);
			if (bx0 < 0) bx0 = 0;
			if (by0 < 0) by0 = 0;
			if (bx1 > ow3) bx1 = ow3;
			if (by1 > oh3) by1 = oh3;
			lv->boxes[lv->n_boxes++] = (struct bc_box){
				.x = bx0, .y = by0,
				.w = bx1 - bx0,
				.h = by1 - by0,
				.r = lv->cfg.censor_rgb[0],
				.g = lv->cfg.censor_rgb[1],
				.b = lv->cfg.censor_rgb[2],
				.a = 1.0f,
				.effect = lv->effect,
				.strength = lv->strength,
				.auto_color = lv->cfg.censor_auto,
				.over = lv->cfg.reverse,
				.use_mask = 1,
				/* Only for the mask's outline band; the
				 * box-relative outline is off for masked
				 * boxes. */
				.outline_rgb = { lv->cfg.outline_rgb[0],
				                 lv->cfg.outline_rgb[1],
				                 lv->cfg.outline_rgb[2] },
				.text_mode = lv->cfg.text_mode,
				.text_rgb = { lv->cfg.text_rgb[0],
				              lv->cfg.text_rgb[1],
				              lv->cfg.text_rgb[2] }
			};
		}

		int px, py;
		if (bc_overlay_pointer(lv->ov, &px, &py)) {
			const int ow = bc_overlay_width(lv->ov);
			const int oh = bc_overlay_height(lv->ov);
			/*
			 * Full-span crosshair rather than a small
			 * reticle. The corner being placed is often on
			 * the edge of something with no landmark near
			 * it, and a line running the width of the
			 * screen can be lined up against content
			 * anywhere along it.
			 */
			const struct bc_box hair = {
				.r = 1.0f, .g = 1.0f, .b = 1.0f, .a = 1.0f,
				.effect = BC_EFFECT_FILL,
				/* A cover in reverse mode too: a
				 * crosshair that cut holes in the mask
				 * would uncover two full lines of
				 * screen. */
				.over = lv->cfg.reverse,
			};
			if (lv->n_boxes < BC_MAX_BOXES) {
				lv->boxes[lv->n_boxes] = hair;
				lv->boxes[lv->n_boxes].x = 0;
				lv->boxes[lv->n_boxes].y = py - 1;
				lv->boxes[lv->n_boxes].w = ow;
				lv->boxes[lv->n_boxes].h = 2;
				lv->n_boxes++;
			}
			if (lv->n_boxes < BC_MAX_BOXES) {
				lv->boxes[lv->n_boxes] = hair;
				lv->boxes[lv->n_boxes].x = px - 1;
				lv->boxes[lv->n_boxes].y = 0;
				lv->boxes[lv->n_boxes].w = 2;
				lv->boxes[lv->n_boxes].h = oh;
				lv->n_boxes++;
			}

			/*
			 * The pending rectangle, drawn with the effect
			 * it will be saved with -- so what is previewed
			 * is the cover itself, not an outline standing
			 * in for one. Sizing it by eye is the whole
			 * task, and an outline hides whether the thing
			 * underneath is actually covered.
			 */
			if (lv->have_anchor && lv->n_boxes < BC_MAX_BOXES) {
				const int x0 = lv->anchor_x < px ? lv->anchor_x : px;
				const int y0 = lv->anchor_y < py ? lv->anchor_y : py;
				const int x1 = lv->anchor_x < px ? px : lv->anchor_x;
				const int y1 = lv->anchor_y < py ? py : lv->anchor_y;
				if (x1 > x0 && y1 > y0)
					lv->boxes[lv->n_boxes++] = (struct bc_box){
						.x = x0, .y = y0,
						.w = x1 - x0, .h = y1 - y0,
						.r = lv->cfg.censor_rgb[0],
						.g = lv->cfg.censor_rgb[1],
						.b = lv->cfg.censor_rgb[2],
						.a = 1.0f,
						.effect = lv->effect,
						.strength = lv->strength,
						.auto_color = lv->cfg.censor_auto,
						.over = lv->cfg.reverse,
						.text_mode = lv->cfg.text_mode,
						.text_rgb = { lv->cfg.text_rgb[0],
						              lv->cfg.text_rgb[1],
						              lv->cfg.text_rgb[2] }
					};
			}
		}
	}

	if (lv->test_box[2] > 0 && lv->n_boxes < BC_MAX_BOXES)
		lv->boxes[lv->n_boxes++] = (struct bc_box){
			.x = lv->test_box[0], .y = lv->test_box[1],
			.w = lv->test_box[2], .h = lv->test_box[3],
			.r = lv->cfg.censor_rgb[0], .g = lv->cfg.censor_rgb[1],
			.b = lv->cfg.censor_rgb[2], .a = 1.0f,
			.effect = lv->effect, .strength = lv->strength,
			.outline = lv->cfg.outline,
			.outline_rgb = { lv->cfg.outline_rgb[0],
			                 lv->cfg.outline_rgb[1],
			                 lv->cfg.outline_rgb[2] },
			.shadow = lv->cfg.shadow,
			.shadow_a = lv->cfg.shadow_opacity,
			.auto_color = lv->cfg.censor_auto,
			.text_mode = lv->cfg.text_mode,
			.text_rgb = { lv->cfg.text_rgb[0], lv->cfg.text_rgb[1],
			              lv->cfg.text_rgb[2] }
		};
	if (lv->test_box[2] > 0 && lv->n_boxes > 0)
		snprintf(lv->boxes[lv->n_boxes - 1].text,
		         sizeof lv->boxes[lv->n_boxes - 1].text, "%s",
		         lv->cfg.caption[0] ? lv->cfg.caption : "CENSORED");
}

/* Decide what may be shown, then takeover and joining. */
/* Do two eye bars overlap each other? */
static int has_doubled_bar(const struct bc_box *bx, int n)
{
	for (int i = 0; i < n; i++) {
		if (!bx[i].over)
			continue;
		for (int j = i + 1; j < n; j++) {
			if (!bx[j].over)
				continue;
			const struct bc_box *a = &bx[i], *b = &bx[j];
			const int x1 = a->x > b->x ? a->x : b->x;
			const int y1 = a->y > b->y ? a->y : b->y;
			const int x2 = a->x + a->w < b->x + b->w ? a->x + a->w : b->x + b->w;
			const int y2 = a->y + a->h < b->y + b->h ? a->y + a->h : b->y + b->h;
			if (x2 <= x1 || y2 <= y1)
				continue;
			const double inter = (double)(x2 - x1) * (y2 - y1);
			const double uni = (double)a->w * a->h + (double)b->w * b->h - inter;
			if (uni > 0.0 && inter / uni > 0.2)
				return 1;
		}
	}
	return 0;
}

static void step_decide(struct bc_live *lv)
{
	/* §6 and the user's switches, ranked -- see failsafe.h. */
	const struct bc_fs_verdict verdict = bc_failsafe_decide(
		&(struct bc_fs_facts){
			.now = lv->tnow, .last_frame = lv->last_frame,
			.last_infer = lv->last_infer,
			.capture_dead = lv->capture_dead,
			.panicked = lv->panicked, .paused = lv->paused,
			.enabled = lv->cfg.enabled, .reverse = lv->cfg.reverse,
		});
	bc_overlay_set_reverse(lv->ov, verdict.reverse);
	lv->show = verdict.show;

	switch (verdict.show) {
	case BC_SHOW_PANIC:
		blank_all(lv->ov, lv->boxes, &lv->n_boxes);
		lv->panic_frames++;
		break;
	case BC_SHOW_FAILSAFE:
		blank_all(lv->ov, lv->boxes, &lv->n_boxes);
		lv->failsafe_frames++;
		if (!lv->failsafe) {
			lv->failsafe = 1;
			lv->failsafe_trips++;
			ERR("FAIL-CLOSED: %s -- output blanked", verdict.why);
		}
		break;
	case BC_SHOW_RAW:
		lv->n_boxes = 0;
		if (lv->paused)
			lv->paused_frames++;
		break;
	case BC_SHOW_CENSOR:
		break;
	}

	if (lv->failsafe && verdict.show != BC_SHOW_FAILSAFE) {
		lv->failsafe = 0;
		LOG("fail-closed cleared -- pipeline live again");
	}

	/*
	 * Area joining, last of all: after the panic and pause paths
	 * have had their say, so it can never resurrect a box one of
	 * them cleared, and after the fail-closed cover is in place,
	 * which is a single full-screen box that nothing can usefully
	 * join to anyway.
	 */
	/*
	 * "takeover": once something has stayed on screen long enough,
	 * stop covering it and cover everything.
	 *
	 * Placed with the join, after panic and pause have had their
	 * say, so it cannot resurrect a frame either of them cleared.
	 * The whole screen replaces the boxes rather than joining them:
	 * one full-screen cover is what takeover means, and leaving the
	 * individual boxes underneath would only cost overdraw.
	 *
	 * Never in reverse mode. There the boxes are the parts left
	 * *visible*, so a full-screen box would reveal the entire
	 * display -- the same reason panic forces reverse off.
	 *
	 * And never over panic or a fail-closed blank. Those are also
	 * full-screen covers, but deliberately a plain opaque fill:
	 * replacing one with an effect that samples the screen would
	 * put the picture back on a display that was blanked precisely
	 * because it must not be shown.
	 */
	if (lv->cfg.takeover_ms > 0.0f && !lv->cfg.reverse &&
	    verdict.show == BC_SHOW_CENSOR) {
		const double t = now_ms();
		int over = 0;
		for (int k = 0; k < BC_TRK_MAX && !over; k++)
			if (lv->tracker.t[k].state != BC_TRK_FREE &&
			    lv->tracker.t[k].born > 0.0 &&
			    t - lv->tracker.t[k].born >= lv->cfg.takeover_ms)
				over = 1;
		if (over) {
			lv->n_boxes = 0;
			lv->boxes[lv->n_boxes++] = (struct bc_box){
				.x = 0, .y = 0,
				.w = bc_overlay_width(lv->ov),
				.h = bc_overlay_height(lv->ov),
				.r = lv->cfg.censor_rgb[0],
				.g = lv->cfg.censor_rgb[1],
				.b = lv->cfg.censor_rgb[2], .a = 1.0f,
				.effect = lv->effect, .strength = lv->strength,
				.auto_color = lv->cfg.censor_auto,
				.text_mode = lv->cfg.text_mode,
				.text_rgb = { lv->cfg.text_rgb[0],
				              lv->cfg.text_rgb[1],
				              lv->cfg.text_rgb[2] }
			};
			lv->takeovers++;
		}
	}

	if (lv->cfg.join && lv->n_boxes > 1) {
		const int before = lv->n_boxes;
		lv->n_boxes = bc_join_boxes(lv->boxes, lv->n_boxes,
		                        BC_JOIN_GAP, lv->cfg.join_waste);
		lv->joins_made += (unsigned long)(before - lv->n_boxes);
	}

	/*
	 * Two eye bars overlapping each other is one face with two covers on it:
	 * a track that was never taken over. Counted per frame, so a doubled bar
	 * is a number and not an impression from a recording.
	 */
	lv->doubled_bar_frames += (unsigned long)has_doubled_bar(lv->boxes, lv->n_boxes);

	if (lv->n_boxes > 0)
		lv->frames_with_boxes++;
	if ((lv->n_boxes > 0) != lv->prev_had_boxes) {
		lv->transitions++;
		lv->prev_had_boxes = lv->n_boxes > 0;
	}
}

/* Reverse-mask style and the clip bookkeeping. */
static void step_mask(struct bc_live *lv)
{
	/*
	 * Style for the reverse mask. Set every frame from the config,
	 * never from the boxes: it must not change because a detection
	 * came or went, or the whole screen repaints on every gap.
	 */
	struct bc_box mask_style = {
		.r = lv->cfg.censor_rgb[0], .g = lv->cfg.censor_rgb[1],
			.b = lv->cfg.censor_rgb[2], .a = 1.0f,
		.effect = lv->effect, .strength = lv->strength,
			.outline = lv->cfg.outline,
			.outline_rgb = { lv->cfg.outline_rgb[0],
			                 lv->cfg.outline_rgb[1],
			                 lv->cfg.outline_rgb[2] },
			.shadow = lv->cfg.shadow,
			.shadow_a = lv->cfg.shadow_opacity,
			.auto_color = lv->cfg.censor_auto,
		.text_mode = lv->cfg.text_mode,
		.text_rgb = { lv->cfg.text_rgb[0], lv->cfg.text_rgb[1],
		              lv->cfg.text_rgb[2] }
	};
	snprintf(mask_style.text, sizeof mask_style.text, "%s",
	         lv->cfg.caption[0] ? lv->cfg.caption : "CENSORED");
	bc_overlay_set_mask(lv->ov, &mask_style);
	if (lv->cfg.reverse && lv->n_boxes == 0)
		lv->mask_no_cutouts++;

	/*
	 * What the clip is worth, sampled on a coarse grid per box --
	 * the same question the shader answers per fragment, asked
	 * cheaply enough to run every frame.
	 */
	if (lv->seg_det && lv->seg_have) {
		const int mw = bc_detect_mask_w(lv->seg_det);
		const int mh = bc_detect_mask_h(lv->seg_det);
		const float nw = (float)bc_preproc_net_w(lv->seg_pp);
		const float nh = (float)bc_preproc_net_h(lv->seg_pp);
		for (int i = 0; i < lv->n_boxes; i++) {
			if (!lv->boxes[i].use_mask || lv->boxes[i].w <= 0 ||
			    lv->boxes[i].h <= 0)
				continue;
			int kept = 0, total = 0;
			for (int gy = 0; gy < 16; gy++) {
				for (int gx = 0; gx < 16; gx++) {
					const float sx = lv->boxes[i].x +
					    (gx + 0.5f) * lv->boxes[i].w / 16.0f;
					const float sy = lv->boxes[i].y +
					    (gy + 0.5f) * lv->boxes[i].h / 16.0f;
					float nx, ny;
					bc_preproc_to_net(lv->seg_pp, sx, sy,
					                  &nx, &ny);
					int mx = (int)(nx / nw * mw);
					int my = (int)(ny / nh * mh);
					total++;
					if (mx < 0 || my < 0 ||
					    mx >= mw || my >= mh)
						continue;
					if (lv->seg_dil[(size_t)my * mw + mx])
						kept++;
				}
			}
			if (total) {
				lv->seg_keep_sum += 100.0 * kept / total;
				lv->seg_keep_n++;
			}
		}
	}
}

/* Present. Blocks until vsync, which paces the whole loop. */
static enum bc_next step_present(struct bc_live *lv)
{
	lv->t = now_ms();
	delay_prepare(lv);
	/*
	 * Is any cover overdue for verification? A track the detector
	 * has not measured for GATE_HOLD_MS is one we have been
	 * covering blind, and under a captured overlay it can never be
	 * measured while the cover is up. Ask for a peek; the presenter
	 * decides whether one is allowed this often.
	 */
	/*
	 * Ask on every frame; the presenter's limiter is what makes it
	 * 2 Hz. Not conditional on having covers up any more: with the
	 * clean-frame gate below, a peeked frame is the *only* thing
	 * that can produce a detection, so waiting for a track to go
	 * overdue before asking for one would mean never detecting
	 * anything in the first place.
	 */
	/*
	 * Declared ahead of the peek's goto, which used to jump over
	 * its initialiser: a granted peek then tested an indeterminate
	 * value below, and any garbage that was not 0 ended the run.
	 */
	int prc = 0;
	if (lv->peek_enabled && !lv->no_hold && !lv->peeking) {
		{
			const double tp = now_ms();
			if (bc_overlay_peek(lv->ov, tp)) {
				struct timespec pts_now;
				clock_gettime(CLOCK_MONOTONIC, &pts_now);
				lv->peek_start_ns = (int64_t)pts_now.tv_sec
				              * 1000000000 + pts_now.tv_nsec;
				lv->peek_started = tp;
				lv->peeking = 1;
				lv->peeks_granted++;
				/* The empty frame is already on screen;
				 * do not immediately cover it again. */
				lv->presents++;
				goto after_present;
			}
			lv->peeks_refused++;
		}
	}

	if (bc_gpu_width(lv->gpu) > 0 && bc_gpu_height(lv->gpu) > 0)
		stat_add(&lv->s_cover_frac, 100.0f * lv->frame_cover /
		         ((float)bc_gpu_width(lv->gpu) * bc_gpu_height(lv->gpu)));

	prc = bc_overlay_present(lv->ov, lv->boxes, lv->n_boxes);
	stat_add(&lv->s_present, now_ms() - lv->t);
	lv->presents++;
after_present:
	if (prc != 0)
		return BC_STOP;
	return BC_GO_ON;
}

int main(int argc, char **argv)
{
	static struct bc_live live;
	struct bc_live *const lv = &live;

	/*
	 * An empty or "auto" argv[1] reads as "not given", the same way argv[2]
	 * does for the model: the launcher must fill the position to reach the
	 * duration in argv[3], and it has no more idea than we do what the
	 * monitor is called.
	 */
	const char *output = (argc > 1 && argv[1][0] &&
	                      strcasecmp(argv[1], "auto") != 0) ? argv[1]
	                                                        : DEFAULT_OUTPUT;
	/* An empty argv[2] means "not given" -- an unquoted shell variable that
	 * expanded to nothing must fall back to the default, not load "". */
	const int   model_argv = (argc > 2 && argv[2][0]);
	const char *model  = model_argv ? argv[2] : DEFAULT_MODEL;
	lv->secs = (argc > 3) ? atof(argv[3]) : DEFAULT_SECS;

	if (!(lv->secs > 0) || lv->secs > HARD_LIMIT)
		lv->secs = DEFAULT_SECS;

	/*
	 * Input size follows the model. 16:9-ish rather than square: the models
	 * have dynamic H/W, and a square letterbox of a 2560x1440 screen would
	 * spend ~44% of the compute on grey padding.
	 */
	lv->net_w = 320;
	lv->net_h = 192;

	LOG("phase 3 joined pipeline");

	signal(SIGINT, on_signal);
	signal(SIGTERM, on_signal);
	signal(SIGALRM, on_alarm);
	/* Install before anything else can send them: the default action for
	 * both is to terminate the process. */
	signal(SIGUSR1, on_user);
	signal(SIGUSR2, on_user);
	alarm((unsigned)lv->secs + 30);

	/*
	 * A pose model used purely as an eye locator (BC_POSE_MODEL), alongside
	 * the primary detector rather than instead of it: puryfi keeps finding
	 * and censoring everything, and this only ever answers "where are the
	 * eyes inside this face, and at what angle".
	 *
	 * That division is what makes it safe to run a face model whose taxonomy
	 * has nothing to do with the censor policy -- no class index from here
	 * ever reaches cfg.censor[], which is keyed to the primary model's
	 * table.
	 *
	 * Kept opt-in. The 5-point convention is WIDERFACE's, trained on
	 * photographic faces, and on drawn ones the box lands but the landmarks
	 * can regress somewhere else on the face -- confidently, so the model's
	 * own score cannot catch it. The geometric sanity gate below rejects the
	 * worst of that; it does not make the model right about anime.
	 */
	/*
	 * Anime landmarks (BC_LANDMARK_MODEL), the one orientation source that
	 * works on drawn faces. Top-down: it runs on a crop per face, so it has
	 * its own preprocess at the model's input size with ImageNet
	 * normalisation, and the face boxes go in as crop rectangles.
	 *
	 * Capped per frame because the cost is linear in faces and inference is
	 * already the scarcest thing in the loop. Faces past the cap keep the
	 * band, which is a worse bar rather than no bar.
	 */



	struct bc_portal    portal = { .pipewire_fd = -1 };
	int                 have_portal = 0;
	int                 rc = 1;

	const char *cfg_path = getenv("BC_CONFIG");
	bc_config_load(&lv->cfg, cfg_path ? cfg_path : DEFAULT_CONFIG);
	/*
	 * The config remembers the model, so the GUI can change it and a plain
	 * run comes back on the same one. An explicit argv[2] still wins for
	 * that run -- naming a model on the command line and being given a
	 * different one would be indefensible -- and is then written back as
	 * the new current model.
	 */
	if (!model_argv && lv->cfg.model[0])
		model = lv->cfg.model;

	/*
	 * Declared up here rather than beside the fopen below: the model check
	 * that follows can jump to the cleanup label, and a flag the cleanup
	 * reads must not be skipped over on the way there.
	 */
	int wrote_pid = 0;

	/*
	 * Whatever we settled on, it has to exist. It very often will not on a
	 * machine that is not the one this was developed on: the config carries
	 * a path, the built-in default names a specific file, and a source tree
	 * unpacked from a tarball ships whichever models fitted in it. Failing
	 * at that point would be a first run that dies on a filename.
	 *
	 * So fall back to any model present. Auxiliary heads are skipped by
	 * name -- a segmentation, landmark, pose or eye model loaded as *the*
	 * detector would run, find its own classes, and censor nothing that
	 * matters, which looks far more like a broken program than a missing
	 * file does.
	 */
	static char model_found[1024];
	if (access(model, R_OK) != 0) {
		const char *picked = pick_any_model(model_found,
		                                    sizeof model_found);
		if (picked) {
			WARN("model '%s' not found -- falling back to %s",
			     model, picked);
			model = picked;
		} else {
			ERR("model '%s' not found, and no .onnx in vendor/", model);
			ERR("run ./tools/setup.sh, or name one: betacensor live "
			    "auto path/to/model.onnx");
			goto out;
		}
	}

	/*
	 * Report the model *after* the config has had its say, not the
	 * command-line default: run.sh passes an empty argv[2] so the config
	 * decides, and announcing the fallback here named a model that was
	 * never loaded.
	 */
	want_net(model, lv->cfg.section_split, &lv->net_w, &lv->net_h);
	lv->net_req_h = lv->net_h;
	LOG("output=%s model=%s net=%dx%d duration=%.1fs",
	    output ? output : "auto (first listed)", model, lv->net_w, lv->net_h, lv->secs);
	cfg_env_overrides(&lv->cfg);

	/*
	 * A PID file, so the hotkey can name this exact process. Deliberately
	 * not `pkill -USR1 bc`: a pattern kill matches anything whose command
	 * line happens to contain the pattern, and the failure mode is signalling
	 * something unrelated.
	 */
	FILE *pf = fopen(DEFAULT_PIDFILE, "w");
	if (pf) {
		fprintf(pf, "%ld\n", (long)getpid());
		fclose(pf);
		wrote_pid = 1;
	} else {
		WARN("could not write %s; panic/pause must be signalled by hand",
		     DEFAULT_PIDFILE);
	}

	/*
	 * Which overlay surface to use is a *compositor* question, not a
	 * preference, and the right answer differs by machine -- so it is
	 * detected rather than fixed.
	 *
	 * On KWin: toplevel. It appears in the task manager, which is the only
	 * place KWin offers "Hide from Screencast"; with that set the detector
	 * stops seeing our own censor bars and the feedback loop disappears
	 * entirely (DESIGN.md §3[7]). The cost is that it does not draw over
	 * another fullscreen window.
	 *
	 * Anywhere else: layer-shell. The toplevel mode buys exactly one thing,
	 * and that thing is a KWin window-rule property no other compositor
	 * has; without it a toplevel is strictly worse than a layer surface,
	 * which at least stacks reliably. wlroots compositors (sway, Hyprland,
	 * river) all carry zwlr_layer_shell_v1. The feedback loop is then real,
	 * and the machinery built for it -- the occlusion hold, kept as the
	 * layer-shell fallback -- switches itself on with the mode below.
	 *
	 * BC_LAYER=1 and BC_TOPLEVEL=1 force either way, because a detected
	 * default is a guess and the person testing this needs to be able to
	 * try the other one without editing code.
	 */
	enum bc_overlay_mode omode;
	const char *desktop = getenv("XDG_CURRENT_DESKTOP");
	const int   is_kde  = (desktop && strcasestr(desktop, "KDE")) ||
	                      getenv("KDE_FULL_SESSION") != NULL;
	if (getenv("BC_LAYER"))
		omode = BC_OVERLAY_LAYER;
	else if (getenv("BC_TOPLEVEL"))
		omode = BC_OVERLAY_TOPLEVEL;
	else
		omode = is_kde ? BC_OVERLAY_TOPLEVEL : BC_OVERLAY_LAYER;

	LOG("overlay mode: %s (desktop=%s)%s",
	    omode == BC_OVERLAY_LAYER ? "layer-shell" : "toplevel",
	    desktop ? desktop : "unset",
	    (getenv("BC_LAYER") || getenv("BC_TOPLEVEL")) ? " [forced]" : "");

	/* Overlay first: it owns the EGLDisplay everything else shares. */
	lv->ov = bc_overlay_create_ex(output, omode);
	if (!lv->ov)
		goto out;

	lv->gpu = bc_gpu_create_shared(bc_overlay_egl_display(lv->ov));
	if (!lv->gpu)
		goto out;

	/*
	 * Detector before preprocessor, because only the model knows the tensor
	 * shape. net_w/net_h above are a filename guess and are merely the
	 * request; a model with a fixed input overrides them, and sizing the
	 * preprocessor from the guess instead would feed every inference the
	 * wrong dimensions and produce nothing at all.
	 */
	lv->no_hyst = getenv("BC_NO_HYST") != NULL;
	lv->no_smooth = getenv("BC_NO_SMOOTH") != NULL;
	/*
	 * Easing is inward-only, so a smaller alpha only ever leaves the box
	 * momentarily *larger* -- the fail-closed direction. That is what makes
	 * this safe to turn down as far as the jitter needs.
	 */
	lv->alpha_env = (float)env_num("BC_POS_ALPHA", 0.0);
	if (lv->alpha_env <= 0.0f || lv->alpha_env > 1.0f)
		lv->alpha_env = 0.0f;   /* unset or nonsense: follow the config */

	lv->det = bc_detect_create(model, lv->net_w, lv->net_h, 4);
	if (!lv->det)
		goto out;
	lv->net_w = bc_detect_net_w(lv->det);
	lv->net_h = bc_detect_net_h(lv->det);

	lv->pp = bc_preproc_create(lv->net_w, lv->net_h);
	if (!lv->pp)
		goto out;
	apply_sections(lv->pp, lv->det, &lv->cfg, bc_overlay_width(lv->ov),
					               bc_overlay_height(lv->ov));

	/*
	 * Second detector, for eye position only.
	 *
	 * The main models carry an EYE class they never predict, and the
	 * dedicated model's boxes are too small and jittery to track on their
	 * own -- smoothing and prediction ratchet them out over the whole face.
	 * Used together they work: the face box supplies a stable region and the
	 * eye boxes say where inside it to put the bar, which is the one thing a
	 * face box cannot tell us at an arbitrary rotation.
	 *
	 * Its taxonomy is deliberately not adopted: the config, the GUI and every
	 * class index in flight are keyed to the primary model's table.
	 */
	/*
	 * Config supplies the path; the env var still wins for a single run,
	 * exactly as `model` behaves for the primary detector. Naming one on the
	 * command line and being given a different one would be indefensible.
	 */
	const char *lm_path = getenv("BC_LANDMARK_MODEL");
	if (!lm_path && lv->cfg.landmark_model[0])
		lm_path = lv->cfg.landmark_model;
	if (lv->cfg.eye_bar && lm_path) {
		const char *lp = lm_path;
		lv->lm_det = bc_landmark_create(lp, 2);
		if (lv->lm_det) {
			const int side = bc_landmark_input(lv->lm_det);
			lv->lm_pp = bc_preproc_create(side, side);
			if (lv->lm_pp) {
				/* ImageNet, as the model was trained. Getting
				 * this wrong does not fail -- it just moves
				 * every point. */
				static const float mean[3] = { 0.485f, 0.456f, 0.406f };
				static const float std[3]  = { 0.229f, 0.224f, 0.225f };
				bc_preproc_set_norm(lv->lm_pp, mean, std);
				lv->lm_tensor = malloc((size_t)BC_LM_MAX_FACES * 3 *
				                   (size_t)side * (size_t)side *
				                   sizeof *lv->lm_tensor);
			}
			if (!lv->lm_pp || !lv->lm_tensor) {
				WARN("landmark: preprocess setup failed -- "
				     "falling back to the band");
				bc_landmark_destroy(lv->lm_det);
				lv->lm_det = NULL;
			}
		} else {
			WARN("landmark: could not load '%s' -- eye bars will "
			     "use the band", lp);
		}
	}

	/*
	 * The segmentation model. Its taxonomy is COCO and must never reach the
	 * global class table, hence create_ex with adopt_taxonomy = 0 -- the same
	 * reason the eye model uses it. If anything here fails the censor runs on
	 * without it, covering whole rectangles, which is the working behaviour
	 * and not a degraded one.
	 */
	if (lv->cfg.seg_model[0]) {
		lv->seg_det = bc_detect_create_ex(lv->cfg.seg_model, 640, 640, 2, 0);
		if (lv->seg_det && bc_detect_has_mask(lv->seg_det)) {
			lv->seg_pp = bc_preproc_create(bc_detect_net_w(lv->seg_det),
			                           bc_detect_net_h(lv->seg_det));
			const size_t px = (size_t)bc_detect_mask_w(lv->seg_det) *
			                  (size_t)bc_detect_mask_h(lv->seg_det);
			if (lv->seg_pp) {
				lv->seg_tensor = malloc(bc_preproc_total_floats(lv->seg_pp) *
				                    sizeof *lv->seg_tensor);
				lv->seg_mask   = malloc(px);
				lv->seg_dil    = malloc(px);
			}
			if (!lv->seg_pp || !lv->seg_tensor || !lv->seg_mask || !lv->seg_dil) {
				WARN("segmentation: setup failed -- covers stay "
				     "rectangular");
				bc_detect_destroy(lv->seg_det);
				lv->seg_det = NULL;
			} else {
				LOG("segmentation: %s, %dx%d masks, margin %.2f%%",
				    lv->cfg.seg_model, bc_detect_mask_w(lv->seg_det),
				    bc_detect_mask_h(lv->seg_det), lv->cfg.seg_margin);
			}
		} else {
			WARN("segmentation: '%s' has no mask output -- covers stay "
			     "rectangular", lv->cfg.seg_model);
			if (lv->seg_det) {
				bc_detect_destroy(lv->seg_det);
				lv->seg_det = NULL;
			}
		}
	}

	if (lv->cfg.eye_bar && (getenv("BC_EYE_MODEL") || getenv("BC_POSE_MODEL"))) {
		/*
		 * A pose model wins if both are named: it answers the same
		 * question with keypoints, which are predicted jointly with
		 * their own box and so cannot disagree with it -- the failure
		 * that makes the box-based eye model ratchet outward.
		 *
		 * 640 rather than the eye model's 320: the landmarks are the
		 * output here, not the box, and they need the resolution.
		 */
		lv->pose_mode = getenv("BC_POSE_MODEL") != NULL;
		const char *em = lv->pose_mode ? getenv("BC_POSE_MODEL")
		                           : getenv("BC_EYE_MODEL");
		const int   side = lv->pose_mode ? 640 : 320;
		lv->eye_det = bc_detect_create_ex(em, side, side, 2, 0);
		if (lv->eye_det) {
			lv->eye_pp = bc_preproc_create(bc_detect_net_w(lv->eye_det),
			                           bc_detect_net_h(lv->eye_det));
			if (lv->eye_pp)
				lv->eye_tensor = malloc(bc_preproc_floats(lv->eye_pp) *
				                    sizeof *lv->eye_tensor);
			if (!lv->eye_pp || !lv->eye_tensor) {
				WARN("eye model: preprocess setup failed -- "
				     "falling back to the centred band");
				bc_detect_destroy(lv->eye_det);
				lv->eye_det = NULL;
			} else {
				/*
				 * Deliberately below the main threshold. A weak
				 * eye detection is only ever used to place a bar
				 * *inside* a face the primary model already
				 * found, and anything outside one is discarded --
				 * so containment does the filtering that a
				 * confidence floor would normally do, and a
				 * higher floor only costs us bars.
				 */
				bc_detect_set_conf(lv->eye_det, lv->pose_mode ? 0.35f
				                                      : 0.15f);
				if (lv->pose_mode && bc_detect_n_kpt(lv->eye_det) < 2) {
					WARN("pose model: '%s' has no pose head "
					     "(%d keypoints) -- it cannot say "
					     "where eyes are; ignoring it", em,
					     bc_detect_n_kpt(lv->eye_det));
					bc_detect_destroy(lv->eye_det);
					lv->eye_det = NULL;
					lv->pose_mode = 0;
				} else {
					LOG("%s: %s (%dx%d) -- bars placed from "
					    "%s", lv->pose_mode ? "pose model"
					                    : "eye model", em,
					    bc_detect_net_w(lv->eye_det),
					    bc_detect_net_h(lv->eye_det),
					    lv->pose_mode ? "eye keypoints, matched "
					                "to the primary model's "
					                "faces"
					              : "detected eyes");
				}
			}
		} else {
			WARN("eye model: could not load '%s' -- eye bars will "
			     "use the centred band", em);
		}
	}

	lv->tensor = malloc(bc_preproc_total_floats(lv->pp) * sizeof *lv->tensor);
	if (!lv->tensor)
		goto out;

	/*
	 * The config was parsed above, before the model had told us its
	 * taxonomy, so its per-class flags were resolved against the default
	 * table. If the model turned out to disagree, those flags are now
	 * indexed wrong -- silently censoring the wrong body parts -- so re-read
	 * the file now that bc_class_name() means what the model means.
	 *
	 * The count is not the thing that changed -- the names are. Two
	 * taxonomies of the same size read as no change at all, and the reload
	 * is what rewrites the file into the running model's class list, so
	 * skipping it leaves the GUI offering the previous model's checkboxes.
	 * The reload is one open() against correctness here, so it is
	 * unconditional.
	 */
	LOG("config: re-reading against the model's %d-class taxonomy",
	    bc_class_count());
	bc_config_load(&lv->cfg, cfg_path ? cfg_path : DEFAULT_CONFIG);
	cfg_env_overrides(&lv->cfg);

	/*
	 * Record the model that actually loaded, so the GUI has something to
	 * show and a later plain run comes back on the same one. bc_config_save
	 * does not touch the remembered mtime, so poll once to resync it --
	 * otherwise our own write looks like an external edit.
	 */
	if (strcmp(lv->cfg.model, model)) {
		snprintf(lv->cfg.model, sizeof lv->cfg.model, "%s", model);
		bc_config_save(&lv->cfg);
		bc_config_poll(&lv->cfg);
	}

	apply_confidence(lv->det, NULL, &lv->cfg, lv->no_hyst);

	/*
	 * Map the overlay *before* the screencast exists.
	 *
	 * A Wayland surface is not mapped until it commits a buffer, and KWin
	 * applies window rules to a window when it maps. Presenting for the
	 * first time inside the main loop -- which is what this used to do --
	 * meant the stream was already running before the window existed as far
	 * as KWin was concerned, so the "Hide from screenshots and screen
	 * recordings" rule had nothing to apply to and the compositor happily
	 * captured our own censor bars.
	 *
	 * That is a feedback loop, not a cosmetic problem: the bars land in the
	 * captured frame, change what the detector sees, which moves the bars,
	 * which changes the frame again. Measured on a *frozen* video frame,
	 * 6.8% of captured frames differed from the previous one with bars drawn
	 * versus 0.2% with none, and tracks were dying and respawning 3.8 times
	 * a second over content that was not moving at all.
	 *
	 * Two empty frames: the first commits a buffer and maps the surface, the
	 * second gives the compositor a round trip to have acted on it.
	 */
	{
		/*
		 * Empty frames until the compositor has had time to act, not
		 * just one or two. Measured: with two presents the loop is gone
		 * in steady state but ~1 s of startup still leaks -- a fixed
		 * ~150 changed frames whether the run is 8 s or 20 s, which is
		 * the rule landing late rather than never. Presenting for a set
		 * wall-clock period closes that window.
		 *
		 * These frames are empty, so nothing is uncensored that would
		 * otherwise be covered: the capture has not started yet and
		 * there is nothing to censor.
		 */
		double t0 = now_ms();
		int    n  = 0;
		while (now_ms() - t0 < OVERLAY_SETTLE_MS) {
			bc_overlay_present(lv->ov, NULL, 0);
			n++;
		}
		LOG("overlay settled: %d empty frames over %.0f ms before "
		    "starting the screencast", n, now_ms() - t0);
	}

	if (bc_portal_acquire(&portal, DEFAULT_RESTORE) < 0)
		goto out;
	have_portal = 1;

	lv->cap = bc_capture_start(portal.pipewire_fd, portal.node_id, lv->gpu);
	portal.pipewire_fd = -1;   /* pipewire owns it now */
	if (!lv->cap)
		goto out;

	bc_tracker_init(&lv->tracker, lv->cfg.confidence, track_iou_thresh(),
	                lv->cfg.sticky_ms);
	lv->tracker.same_class = classes_similar;
	apply_confidence(lv->det, &lv->tracker, &lv->cfg, lv->no_hyst);

	/*
	 * Keypoint bars are only right when the model was trained on the kind of
	 * face on screen; a WIDERFACE model on anime is confidently wrong. Off
	 * switch for running such a model anyway and keeping the geometric band.
	 */
	lv->no_kpt = getenv("BC_NO_KPT") != NULL;
	lv->debug_lm = getenv("BC_DEBUG_LM") != NULL;
	/*
	 * How much bigger than the measured eye extents the bar is drawn, and
	 * whether the old separation-scaled size still acts as a floor.
	 *
	 * Tunable because the right value is a judgement about the content, not
	 * a fact about the model: the landmarks say where the eyes are, and how
	 * much margin around them reads as "covered" is taste. BC_LM_NOFLOOR
	 * lets the measured extents shrink the bar below the separation-scaled
	 * size, which is the only way to get a *smaller* bar than the old
	 * heuristic gave.
	 */
	lv->lm_bar_pad = lv->cfg.lm_bar_pad > 0.0f ? lv->cfg.lm_bar_pad : LM_BAR_PAD;
	lv->lm_bar_pad = (float)env_num("BC_LM_BAR_PAD", lv->lm_bar_pad);
	if (!(lv->lm_bar_pad > 0.0f))
		lv->lm_bar_pad = LM_BAR_PAD;
	lv->lm_nofloor = getenv("BC_LM_NOFLOOR") != NULL;
	/* Size the landmark crop by the box's longest side again -- the arm to
	 * measure against. */
	lv->lm_wide_crop = getenv("BC_LM_WIDE_CROP") != NULL;
	/*
	 * Rotating the bar to a body axis is opt-in, and off by default.
	 *
	 * It was on, and it placed every bar wrongly on the content it was
	 * tried against: mean tilt 35 degrees over ~100 bars whose subjects
	 * were plainly upright, the bar lying diagonally across the face
	 * instead of along the eye line.
	 *
	 * Two independent faults, which is why this is a default change and not
	 * a tuning one:
	 *
	 * - body_axis() assumes the torso lies straight below the head. Portrait
	 *   framing breaks that: a face above and to one side of the chest gives
	 *   an axis tens of degrees off vertical for an upright subject.
	 * - Worse, the angle feeds the extent recovery, which solves for the
	 *   head's width through it. A wrong angle returns a width a third
	 *   short, so the bar stopped spanning the face and left an eye
	 *   uncovered -- the failure direction §2 and §6 both forbid.
	 *
	 * The straight band cannot do either. It is wrong for a subject lying
	 * down, which is the known limitation the file already documents, but it
	 * is wrong in the direction of covering a band of the face rather than
	 * missing the eyes entirely.
	 *
	 * An aspect-ratio gate was tried first and abandoned: it assumes a head
	 * is longer than it is wide, and these face boxes measure w/h 1.08 --
	 * wider than tall -- so it permitted 84 degrees and rejected nothing.
	 * That premise is also what the extent recovery rests on.
	 */
	lv->axis_bar = getenv("BC_AXIS_BAR") != NULL;
	/*
	 * Where the band sits in the face box, as fractions of its height from
	 * the top. Tunable because the right answer depends on how the model
	 * draws the box -- hairline-to-chin on a photographic face puts the eyes
	 * near 40%, and a model that includes hair or ears puts them lower.
	 */
	lv->eyebar_top = EYEBAR_TOP;
	lv->eyebar_bot = EYEBAR_BOT;
	lv->eyebar_top = (float)env_num("BC_EYEBAR_TOP", lv->eyebar_top);
	lv->eyebar_bot = (float)env_num("BC_EYEBAR_BOT", lv->eyebar_bot);
	if (!(lv->eyebar_bot > lv->eyebar_top)) {
		lv->eyebar_top = EYEBAR_TOP;
		lv->eyebar_bot = EYEBAR_BOT;
	}
	LOG("eye bar: band %.0f%%-%.0f%% of the face box%s", lv->eyebar_top * 100.0f,
	    lv->eyebar_bot * 100.0f,
	    lv->axis_bar ? ", body-axis rotation ON (BC_AXIS_BAR)" : "");

	/*
	 * Debug switches. BC_DEBUG_TENSOR dumps the first preprocessed tensor so
	 * the GPU path can be checked against the screen it came from;
	 * BC_DEBUG_ALL logs every detection regardless of the censor policy, as a
	 * positive control that the model is actually firing.
	 */
	lv->dump_path = getenv("BC_DEBUG_TENSOR");
	lv->debug_all = getenv("BC_DEBUG_ALL") != NULL;
	/*
	 * Stability comes from the *region*, not from re-detection.
	 *
	 * Re-detection is the wrong thing to key on: the model fires only
	 * intermittently on our own pixelated output, so tracks die and respawn
	 * and the censor visibly oscillates. The pixelated interior, by
	 * contrast, is byte-identical frame to frame while the content behind it
	 * is static (measured: 0.00/255 over 6 s), and changes the instant the
	 * content does. So hold a box while its region is unchanged and release
	 * it the moment it is not -- stable on static content, responsive on
	 * change. BC_NO_HOLD reverts to purely detection-driven.
	 */
	/*
	 * With the overlay excluded from capture the detector sees the real
	 * screen every frame, so tracks refresh from genuine detections and the
	 * hold is not merely unnecessary but harmful -- it would smear boxes past
	 * the content. Only the layer-shell path, which cannot be excluded, needs
	 * it. BC_HOLD / BC_NO_HOLD force either way.
	 */
	lv->no_hold = (omode == BC_OVERLAY_TOPLEVEL);
	if (getenv("BC_HOLD"))    lv->no_hold = 0;
	if (getenv("BC_NO_HOLD")) lv->no_hold = 1;

	/*
	 * `prev_tensor` (declared at the top, with the rest of what `out:`
	 * frees) is the last tensor we inferred, kept so the occlusion hold can
	 * ask what changed under each cover. Allocated only when the hold is
	 * live -- on the KWin path it is off and this is 1.2 MB of nothing.
	 */
	/*
	 * Off by default, and staying off until the growth defect below is
	 * fixed. Holding a track for lack of evidence advances its prediction
	 * clock, and prediction grows a box to the union of its old and
	 * predicted extents -- correct once, unbounded when repeated every
	 * frame with nothing ever re-measuring it. Observed as covers growing
	 * without limit in every direction. The ceiling this bypassed was what
	 * had been preventing that.
	 */
	lv->peek_enabled = getenv("BC_PEEK") != NULL;
	lv->max_box_growth = 1.0f;
	if (!lv->no_hold && lv->tensor) {
		lv->prev_tensor = malloc(bc_preproc_total_floats(lv->pp) * sizeof *lv->prev_tensor);
		if (!lv->prev_tensor)
			WARN("no memory for the occlusion hold -- censoring will "
			     "be detection-driven, and may oscillate under a "
			     "captured overlay");
		else
			lv->want_host = 1;
	}
	if (lv->want_host && bc_detect_zero_copy(lv->det) && bc_preproc_zero_copy(lv->pp))
		LOG("occlusion hold: forcing the tensor readback the CUDA path "
		    "would skip -- the hold reads it on the host");
	lv->trace = getenv("BC_TRACE") != NULL;
	lv->no_gate = getenv("BC_NO_GATE") != NULL;
	int         no_predict = getenv("BC_NO_PREDICT") != NULL;
	lv->no_fade = getenv("BC_NO_FADE") != NULL;
	lv->last_n_boxes = -1;
	snprintf(lv->cur_model, sizeof lv->cur_model, "%s", model);

	/* Lag vs flicker, tunable without a rebuild. */
	double hold_max_ms = env_num("BC_HOLD_MAX", HOLD_MAX_DEFAULT);
	if (!(hold_max_ms >= 0.0))
		hold_max_ms = HOLD_MAX_DEFAULT;

	lv->next_cfg_poll = now_ms() + CONFIG_POLL_MS;



	/* Minimum travel between samples; doubles whenever the buffer fills. */
	lv->path_min_d = 1.0f;


	/* Was a two-way ternary from when there were two effects, so all
	 * thirteen of the others reported themselves as "solid fill". */
	LOG("effect: %s (strength %.0f%% of box short side)",
	    bc_effect_name(lv->cfg.effect), lv->cfg.strength);
	/*
	 * Classes pinned to an effect of their own. Named rather than counted:
	 * the whole point of the feature is that different parts are covered
	 * differently, and "3 classes pinned" would not tell you whether the one
	 * you set is among them.
	 */
	for (int i = 0; i < bc_class_count(); i++)
		if (lv->cfg.censor[i] && lv->cfg.effect_for[i] >= 0)
			LOG("  %-24s -> %s (strength %.0f%%)",
			    bc_class_name(i), bc_effect_name(lv->cfg.effect_for[i]),
			    lv->cfg.strength_by[lv->cfg.effect_for[i]]);

	/*
	 * Fault injection, debug only. The §6 paths are the ones that must work
	 * on the day everything else does not, so they need to be exercisable on
	 * demand -- the alternative is killing the compositor's screencast and
	 * hoping the failure looks like the real one.
	 * BC_FAULT_STALL=ms  stops consuming frames  (capture stalled)
	 * BC_FAULT_DEAD=ms   reports a lost stream   (capture stream lost)
	 */
	lv->fault_stall = env_num("BC_FAULT_STALL", 0.0);
	lv->fault_dead = env_num("BC_FAULT_DEAD", 0.0);
	/*
	 * BC_FAULT_NODETECT=ms  stop submitting work to the detector.
	 *
	 * Tracking is now stable enough that tracks essentially never expire on
	 * live content, so the fade-out path cannot be reached by tuning
	 * thresholds -- and an untestable path is one that rots. This stops
	 * detections cleanly so existing tracks age out and fade.
	 */
	lv->fault_nodet = env_num("BC_FAULT_NODETECT", 0.0);
	if (lv->fault_stall > 0.0 || lv->fault_dead > 0.0)
		WARN("fault injection active: stall@%.0fms dead@%.0fms",
		     lv->fault_stall, lv->fault_dead);

	const char *test_box_env = getenv("BC_TEST_BOX");
	if (test_box_env) {
		if (sscanf(test_box_env, "%d,%d,%d,%d", &lv->test_box[0],
		           &lv->test_box[1], &lv->test_box[2], &lv->test_box[3]) == 4)
			LOG("debug: pinned test box at %d,%d %dx%d",
			    lv->test_box[0], lv->test_box[1], lv->test_box[2], lv->test_box[3]);
		else
			lv->test_box[2] = 0;
	}


	lv->start = now_ms();
	lv->last_frame = lv->last_infer = lv->start;
	LOG("tracking: sticky %.0f ms, hold %s, ceiling %.0f s",
	    lv->cfg.sticky_ms,
	    lv->no_hold ? "off (detection-driven)" : "on (region-stability)",
	    hold_max_ms / 1000.0);

	char censored[512] = { 0 };
	for (int i = 0; i < bc_class_count(); i++)
		if (lv->cfg.censor[i]) {
			if (censored[0])
				strncat(censored, " ",
				        sizeof censored - strlen(censored) - 1);
			strncat(censored, bc_class_name(i),
			        sizeof censored - strlen(censored) - 1);
		}
	LOG("running -- censoring: %s", censored[0] ? censored : "(nothing)");
	LOG("edit %s to change policy while running", lv->cfg.path);
	if (wrote_pid) {
		LOG("panic (blank screen): kill -USR1 $(cat %s)", DEFAULT_PIDFILE);
		LOG("pause  (no censoring): kill -USR2 $(cat %s)", DEFAULT_PIDFILE);
	}

	while (running && (now_ms() - lv->start) < lv->secs * 1000.0) {
		step_signals(lv);
		step_config(lv);
		step_drawing(lv);
		step_outlines(lv);
		step_capture(lv);
		if (step_submit(lv) == BC_NEXT_FRAME)
			continue;
		step_eyes(lv);
		step_detections(lv);
		step_covers(lv);
		step_regions(lv);
		step_decide(lv);
		step_mask(lv);
		if (step_present(lv) == BC_STOP)
			break;
	}

	double elapsed = (now_ms() - lv->start) / 1000.0;
	alarm(0);

	LOG("");
	LOG("=== measured (DESIGN.md §2) ===");
	LOG("  capture frames imported : %lu (%.1f/s)",
	    (unsigned long)bc_capture_frames(lv->cap),
	    bc_capture_frames(lv->cap) / elapsed);
	LOG("  preprocess + readback   : mean %.2f ms  max %.2f ms  (n=%lu)",
	    stat_mean(&lv->s_pre), lv->s_pre.max, lv->s_pre.n);
	LOG("  inference (%dx%d, %s) : mean %.1f ms  max %.1f ms  (n=%lu)",
	    lv->net_w, lv->net_h, bc_detect_on_gpu(lv->det) ? "CUDA" : "CPU",
	    stat_mean(&lv->s_infer), lv->s_infer.max, lv->s_infer.n);
	LOG("  overlay present (vsync) : mean %.2f ms  max %.2f ms  (n=%lu)",
	    stat_mean(&lv->s_present), lv->s_present.max, lv->s_present.n);
	LOG("  detections submitted    : %lu, skipped (worker busy): %lu",
	    lv->submitted, lv->skipped);
	LOG("  change gate             : %lu frames identical, skipped (%.1f%% "
	    "of readbacks)%s", lv->gated,
	    lv->s_pre.n ? 100.0 * lv->gated / lv->s_pre.n : 0.0,
	    lv->no_gate ? "  [DISABLED]" : "");
	LOG("  takeovers               : %lu (%.1f/s) <- whole screen covered",
	    lv->takeovers, lv->takeovers / elapsed);
	LOG("  boxes joined            : %lu (%.1f/s) <- overlapping covers merged%s",
	    lv->joins_made, lv->joins_made / elapsed, lv->cfg.join ? "" : "  [DISABLED]");
	if (lv->cfg.reverse)
		LOG("  reverse mask, no holes  : %lu (%.1f/s) <- everything covered, "
		    "nothing revealed", lv->mask_no_cutouts, lv->mask_no_cutouts / elapsed);
	if (bc_preproc_tiles(lv->pp) > 1)
		LOG("  seam merges             : %lu (%.1f/s) <- halves rejoined across "
		    "a tile boundary", lv->seam_merges, lv->seam_merges / elapsed);
	if (lv->eyes_under_bar)
		LOG("  eyes left to the bar    : %lu (%.1f/s) <- EYE detections inside "
		    "a barred face, not drawn twice", lv->eyes_under_bar,
		    lv->eyes_under_bar / elapsed);
	LOG("  readbacks skipped       : %lu (%.1f/s) <- worker busy, tensor not built",
	    lv->readbacks_skipped, lv->readbacks_skipped / elapsed);
	LOG("  scene cuts              : %lu (%.1f/s) <- picture replaced, ghosts dropped",
	    lv->scene_cuts, lv->scene_cuts / elapsed);
	LOG("  still-screen holds      : %lu (%.1f/s) <- no frame sent at all",
	    lv->quiet_holds, lv->quiet_holds / elapsed);
	LOG("  gate overridden by cap  : %lu (%.1f/s) <- re-verify on a still screen",
	    lv->gate_forced, lv->gate_forced / elapsed);
	LOG("  inferences completed    : %lu (%.1f/s)",
	    bc_detect_count(lv->det), bc_detect_count(lv->det) / elapsed);
	if (lv->delay_applied > 0.0f && lv->ring)
		LOG("  screen delay            : %.0f ms asked; shown %.0f ms behind on average (max %.0f), "
		    "records bracketed %.0f ms apart; %lu frames held, %lu blanked, of %lu; "
		    "%.1f distinct frames/s shown, clock held at the checked horizon %lu times; "
		    "%lu shown frames with two eye bars on one spot",
		    lv->delay_applied, stat_mean(&lv->s_delay_lag), lv->s_delay_lag.max,
		    stat_mean(&lv->s_delay_gap), lv->delay_held, lv->delay_blank,
		    lv->delay_shown + lv->delay_held + lv->delay_blank,
		    lv->delay_distinct / elapsed, lv->delay_stalled, lv->delay_doubled);
	LOG("  present rate            : %.1f fps over %.2fs",
	    lv->presents / elapsed, elapsed);
	LOG("  frames showing a box    : %lu / %lu (%.1f%%)",
	    lv->frames_with_boxes, lv->presents,
	    lv->presents ? 100.0 * lv->frames_with_boxes / lv->presents : 0.0);
	LOG("  box on/off transitions  : %lu  (hold %s) <- flicker metric",
	    lv->transitions, lv->no_hold ? "DISABLED" : "enabled");
	if (lv->s_mad.n)
		LOG("  change MAD (hold thresh %.3f): mean %.5f max %.5f", HOLD_MAD,
		    stat_mean(&lv->s_mad), lv->s_mad.max);
	LOG("  hold: %lu checks, %lu released by change, %lu refreshes hit the "
	    "%.0f s gate cap", lv->s_mad.n, lv->held_released, lv->held_capped,
	    GATE_HOLD_MS / 1000.0);
	/*
	 * The flash-rate line. Peeks are the only thing that uncovers the
	 * screen on its own, and the rate is a safety property, not a
	 * performance one -- so it is reported as a rate against the limit
	 * rather than as a bare count.
	 */
	LOG("  covered area            : mean %.1f%% of the screen, max %.1f%%",
	    stat_mean(&lv->s_cover_frac), lv->s_cover_frac.max);
	LOG("  largest cover growth    : %.2fx its first size%s", lv->max_box_growth,
	    lv->max_box_growth > 3.0f ? "   <- COVERS ARE GROWING, see DESIGN.md §11"
	                          : "");
	LOG("  peeks: %lu granted (%.2f/s, limit %.1f/s), %lu refused as too "
	    "soon, %lu clean frames, %lu covers retired on two clean samples",
	    lv->peeks_granted, lv->peeks_granted / elapsed, 1000.0 / BC_PEEK_MIN_MS,
	    lv->peeks_refused, lv->clean_frames, lv->clean_drops);
	if (lv->peeks_granted / elapsed > 1000.0 / BC_PEEK_MIN_MS + 0.01)
		WARN("  peek rate exceeded its limit -- this is a flash-safety "
		     "bug, not a tuning problem");
	LOG("  tracks                  : %lu taken over by a nearby detection; "
	    "%lu frames (%.1f%%) with two eye bars on one spot",
	    lv->tracker.revived, lv->doubled_bar_frames,
	    lv->presents ? 100.0 * lv->doubled_bar_frames / lv->presents : 0.0);
	LOG("  track churn             : %lu created, %lu expired (%.1f/s) "
	    "<- perceived flicker", lv->tracker.created, lv->tracker.removed,
	    lv->tracker.removed / elapsed);
	LOG("  association             : %lu matched confident, %lu matched weak, "
	    "%lu went lost and were predicted",
	    lv->tracker.matched_high, lv->tracker.matched_low, lv->tracker.lost);
	LOG("  box-count changes       : %lu (%.1f/s) <- boxes appearing/vanishing",
	    lv->boxcount_changes, lv->boxcount_changes / elapsed);
	LOG("  per-detection box jump  : mean %.1f px  max %.1f px (sum of 4 edges)",
	    stat_mean(&lv->s_jitter), lv->s_jitter.max);
	LOG("  fade-out                : %lu box-frames drawn mid-fade over "
	    "%.0f ms%s", lv->fade_frames, FADE_OUT_MS,
	    lv->no_fade ? "  [DISABLED]" : "");
	/*
	 * Score hysteresis and the similar-class counter used to be reported
	 * here. Both are now properties of the association -- the weak band is
	 * the second pass, and the class pairing is tracker.same_class -- so
	 * they are counted above rather than as separate mechanisms.
	 */
	LOG("  edge smoothing          : alpha %.2f, inward only%s",
	    lv->alpha_env ? lv->alpha_env : lv->cfg.smoothing,
	    lv->no_smooth ? "  [DISABLED]" : "");
	/*
	 * The *effective* memory, not cfg.sticky_ms. Those differ whenever the
	 * detection-gap floor is doing its job, and printing the config value
	 * hid that: a run reported "memory 20 ms" while actually holding tracks
	 * for the floor's several hundred, which reads as the floor being absent.
	 */
	LOG("  kalman prediction       : %lu box-frames drawn from prediction "
	    "(memory %.0f ms, config asked %.0f)%s", lv->pred_frames,
	    lv->tracker.memory_ms, lv->cfg.sticky_ms,
	    no_predict ? "  [DISABLED]" : "");
	if (lv->cfg.eye_bar)
		LOG("  eye bars                : effect %s, face %s "
		    "(%lu box-frames as the band alone, %lu drawn over a "
		    "covered face)",
		    lv->cfg.eye_bar_effect >= 0 ? bc_effect_name(lv->cfg.eye_bar_effect)
		                            : "(the face class's own)",
		    lv->cfg.eye_bar_face ? "covered underneath" : "left alone",
		    lv->eyebar_solo, lv->eyebar_over);
	if (lv->lm_det)
		LOG("  landmarks: %lu bar(s) placed from anime eye points, "
		    "%lu rejected as low confidence, %lu run(s) at %.1f ms mean "
		    "[%s]", lv->lm_used, lv->lm_weak, lv->lm_runs,
		    lv->lm_runs ? lv->lm_ms_sum / (double)lv->lm_runs : 0.0,
		    bc_landmark_on_gpu(lv->lm_det) ? "CUDA" : "CPU");
	if (lv->pose_matched || lv->pose_unmatched)
		LOG("  pose model: %lu face(s) matched to a primary face and "
		    "used for the bar, %lu seen but not matched",
		    lv->pose_matched, lv->pose_unmatched);
	if (lv->eyebar_kpt || lv->kpt_reject)
		LOG("  eye bars: %lu placed from model keypoints, "
		    "%lu rejected as implausible for their face box",
		    lv->eyebar_kpt, lv->kpt_reject);
	LOG("  eye bars: %lu from the face band (%lu turned to a body axis, "
	    "%lu of those on a remembered one), %lu placed from detected eyes%s",
	    lv->eyebar_fallback, lv->eyebar_axis, lv->eyebar_axis_mem, lv->eyebars,
	    lv->cfg.eye_bar ? "" : "   [eye_bar off]");
	/*
	 * The ratio is the point, not the total: a remembered axis is the bar
	 * *not* snapping back to upright for a frame. A run where most turned
	 * bars are on remembered axes is one where body_axis is intermittent,
	 * which is the condition this exists for.
	 */
	if (lv->eyebar_axis)
		LOG("    axis tilt: mean %.1f deg, max %.1f deg, %lu bars past 30 deg",
		    lv->axis_deg_sum / lv->eyebar_axis, lv->axis_deg_max, lv->axis_over_30);
	LOG("  detections by class (raw, before the censor policy):");
	{
		int any = 0;
		for (int i = 0; i < bc_class_count(); i++)
			if (lv->det_by_class[i]) {
				LOG("      %-26s %lu%s", bc_class_name(i),
				    lv->det_by_class[i],
				    lv->cfg.censor[i] ? "" : "   (not censored)");
				any = 1;
			}
		for (int i = 0; i < bc_class_count(); i++)
			if (!lv->det_by_class[i] && lv->cfg.censor[i])
				LOG("      %-26s 0   <- censored but never detected",
				    bc_class_name(i));
		if (!any)
			LOG("      (none)");
	}
	if (lv->swaps)
		LOG("  model swaps             : %lu", lv->swaps);
	if (lv->seg_det)
		LOG("  silhouettes             : %lu (%.1f/s), mean %.1f%% of the "
		    "frame <- covers clipped to the subject",
		    lv->seg_runs, lv->seg_runs / (elapsed > 0 ? elapsed : 1.0),
		    lv->seg_runs ? lv->seg_cov_sum / (double)lv->seg_runs : 0.0);
	if (lv->seg_det)
		LOG("  cover kept by the outline: %.1f%% of box area over %lu boxes "
		    "<- 100%% means the clip changed nothing",
		    lv->seg_keep_n ? lv->seg_keep_sum / (double)lv->seg_keep_n : 0.0,
		    lv->seg_keep_n);
	if (lv->cfg.n_regions)
		LOG("  hand-drawn regions      : %d, drawn %lu times <- covered "
		    "regardless of what the detector found",
		    lv->cfg.n_regions, lv->regions_drawn);
	if (lv->lasso_measured)
		LOG("  outline clip keeps      : %.1f%% of the traced shapes' "
		    "bounding boxes <- 100%% would mean the outline is a rectangle",
		    lv->lasso_kept_pct);
	LOG("  config reloads          : %lu", lv->reloads);
	LOG("  frames blanked (panic)  : %lu, paused: %lu",
	    lv->panic_frames, lv->paused_frames);
	LOG("  max gap between frames  : %.1f ms (stall deadline %.0f ms)",
	    lv->max_frame_gap, STALL_MS);
	LOG("  fail-closed             : %lu trips, %lu frames blanked, "
	    "%lu detector errors", lv->failsafe_trips, lv->failsafe_frames, lv->det_errors);

	/*
	 * A run that had to fail closed is not a success, however cleanly it
	 * exited: reporting it as one would hide exactly the events worth
	 * knowing about.
	 */
	rc = (lv->capture_dead || lv->failsafe_trips || lv->det_errors) ? 1 : 0;
	if (rc)
		WARN("run completed DEGRADED -- see fail-closed counters above");

out:
	if (wrote_pid)
		unlink(DEFAULT_PIDFILE);
	if (lv->cap) bc_capture_stop(lv->cap);
	if (lv->det) bc_detect_destroy(lv->det);
	free(lv->tensor);
	free(lv->prev_tensor);
	free(lv->lasso_mask);
	free(lv->lasso_tmp);
	free(lv->lasso_dil);
	free(lv->lasso_band);
	free(lv->lasso_base);
	if (lv->ring) bc_framering_destroy(lv->ring);
	if (lv->dh)   bc_dh_destroy(lv->dh);
	if (lv->pp)  bc_preproc_destroy(lv->pp);
	if (lv->gpu) bc_gpu_destroy(lv->gpu);
	if (have_portal) bc_portal_release(&portal);
	if (lv->ov)  bc_overlay_destroy(lv->ov);

	if (rc == 0)
		LOG("SUCCESS -- end-to-end pipeline ran.");
	return rc;
}
