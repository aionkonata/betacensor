#ifndef BC_CONFIG_H
#define BC_CONFIG_H

#include "detect.h"

/*
 * Runtime configuration (DESIGN.md §5), reloaded from disk when the file
 * changes so the GUI can edit it while the censor is running. There is no IPC:
 * the file *is* the interface between the two processes, which keeps the C core
 * free of any GUI dependency.
 */

enum bc_cfg_effect {
	BC_CFG_PIXELATE = 0,
	BC_CFG_FILL,
	BC_CFG_BLUR,
	BC_CFG_NOISE,
	BC_CFG_GLITCH,
	BC_CFG_RGBSHIFT,
	BC_CFG_TAPE,
	BC_CFG_HALFTONE,
	BC_CFG_SOBEL,
	BC_CFG_TRIANGLE,
	BC_CFG_CLUSTER,
	BC_CFG_SPLATTER,
	BC_CFG_SHAPE,
	BC_CFG_CAPTION,
	BC_CFG_WALL,
	BC_CFG_LAST_         /* keep last: bounds the name lookup */
};

const char *bc_effect_name(int effect);

/*
 * A region the user drew, rather than one a model found.
 *
 * Stored normalised 0..1 so it means the same thing at any resolution, and so
 * a config written on one monitor is not silently wrong on another.
 *
 * These exist because detection has a floor: a subject the model scores below
 * the threshold is uncovered and nothing in the pipeline can know it. A drawn
 * region is the backstop -- it never asks a model anything, so it cannot be
 * missed, and it is the one kind of cover here that only ever *adds* coverage.
 */
#define BC_MAX_REGIONS 16

/*
 * Points in a hand-drawn outline, after simplification.
 *
 * A freehand stroke arrives with one sample per frame -- hundreds of them --
 * and almost all of it is the hand shaking. Sixty-four is far more than enough
 * to describe any shape drawn by dragging a mouse once, and it keeps the
 * config line short enough to stay a line.
 */
#define BC_MAX_POLY_PTS 64

struct bc_region {
	float x, y, w, h;   /* normalised to the output, top-left origin */
	int   effect;       /* enum bc_cfg_effect, or -1 to follow the global one */

	/*
	 * An outline, or 0 points for a plain rectangle.
	 *
	 * x,y,w,h stays the bounding box either way, and stays what everything
	 * downstream reads: a polygon is drawn as its bounding box *clipped* to
	 * a mask rasterised from these points. So the failure mode of anything
	 * going wrong with the mask -- an allocation, an upload -- is the whole
	 * bounding box covered, which is the §2 direction.
	 */
	int   n_pts;
	float px[BC_MAX_POLY_PTS];
	float py[BC_MAX_POLY_PTS];
};

struct bc_config {
	int   enabled;                    /* master switch */
	/*
	 * Drawing mode: the overlay takes pointer and keyboard input so the
	 * user can place a region by hand. Set by the GUI button, cleared by
	 * the core when drawing ends -- so it is a *request*, not a setting,
	 * and the core's answer is what the file ends up holding.
	 *
	 * Deliberately forced off by bc_config_load(), so it can only ever be
	 * entered by a live change to a running core. A file that already said
	 * true -- because the process died mid-draw -- would otherwise start
	 * the next run with the pointer grabbed and nothing on screen yet to
	 * explain why.
	 */
	int   draw;
	int   reverse;                    /* mask the screen, reveal detections */
	int   eye_bar;                    /* reduce FACE boxes to an eye band */
	/*
	 * Effect for the eye bar itself, or -1 for "no separate bar".
	 *
	 * -1 keeps the behaviour eye_bar has always had: the face box is
	 * *replaced* by the band, so one detection makes one cover. Setting an
	 * effect makes it additive instead -- the face is covered with whatever
	 * its class asks for, and the bar is drawn on top with this. That is the
	 * only way to say "wall-of-text over the face, solid bar over the eyes",
	 * which the either/or model could not express.
	 *
	 * Separate from effect_for[] because it is not about a class: every
	 * eye bar is drawn this way regardless of which head class it came
	 * from, and a face's own class effect is still what covers the face.
	 */
	int   eye_bar_effect;
	/*
	 * Whether the face is covered as well, underneath the bar.
	 *
	 * Split from eye_bar_effect because the two questions are independent
	 * and used to be answered by one number: naming an effect for the bar
	 * was also the only way to say "cover the face", and leaving it off was
	 * the only way to say "bar alone". So a bar with an effect of its own
	 * over an *uncovered* face -- an eye bar in the ordinary sense, on a
	 * face that is otherwise left alone -- could not be asked for at all.
	 *
	 * A file that predates this key has it derived from eye_bar_effect at
	 * the tail of parse(), which is exactly the behaviour it had before, so
	 * nothing changes under an existing config until it is asked to.
	 */
	int   eye_bar_face;
	int   effect;                     /* enum bc_cfg_effect */
	float strength;                   /* % of box short side, for c->effect */

	/*
	 * The same figure, remembered per effect.
	 *
	 * One shared `strength` meant tuning the blur moved the pixelate blocks
	 * and the halftone dots with it, because every effect reads the same
	 * number for a different purpose -- block size, blur radius, dot pitch,
	 * facet size. They want different values, so they get one each, and
	 * `strength` above is just whichever belongs to the effect in use.
	 */
	float strength_by[BC_CFG_LAST_];
	float confidence;                 /* detection floor */
	float dilate;                     /* % box growth, §2 */
	float sticky_ms;                  /* box lifetime past last detection */
	float delay_ms;                   /* show the screen this far behind; 0 = live */
	int   join;                       /* merge overlapping covers into one */
	/*
	 * Detection grid. 1x1 is the whole screen in one inference, which is
	 * the default: a grid costs one inference per tile, and inference is
	 * the scarcest thing in the loop. Raising these makes the detector see
	 * more detail, which is what finds small body parts on a wide screen.
	 */
	int   section_cols, section_rows;
	float section_overlap;            /* fraction of a tile shared, 0..0.9 */
	/*
	 * One further inference over the whole screen alongside the grid. The
	 * grid buys small objects at the cost of large ones -- something filling
	 * the screen is split across every tile and no fragment still looks like
	 * the object -- and this is what catches those. Ignored at 1x1, where it
	 * would only duplicate the one tile there already is.
	 */
	int   section_full_frame;
	/*
	 * Derive the tiles from the screen's shape instead of from cols/rows: as
	 * few full-height square tiles as cover the width (two on 16:9), sized so
	 * the network input is square and nothing is letterboxed away. This is
	 * what Hotscreen's "double detection" is. section_cols/rows/overlap are
	 * ignored while it is on.
	 */
	int   section_split;
	float join_waste;                 /* max fraction of new area a join adds */
	float smoothing;                  /* inward edge ease, 0..1; lower = steadier */

	/*
	 * How the censor *shape* is drawn, as opposed to what fills it
	 * (PUREVISION.md: CensorStrokeConfig / CensorShadowConfig /
	 * SolidOrAutomaticColor). All three are decoration on a cover that
	 * already works, and all three are constrained by §6: the outline is
	 * drawn outward so it can only add covered area, and the shadow lies
	 * beyond it, over screen that was never censored in the first place.
	 */
	/*
	 * Two of Hotscreen's filters, by name only -- its list survives in
	 * config_reader.gdc with no semantics attached, so these are the obvious
	 * reading rather than a port. Both only ever cover *more*, which is what
	 * makes guessing at them safe: the worst a wrong guess does is cover too
	 * much, which is the direction §2 already prefers.
	 */
	float growth;                     /* % per second a cover grows; 0 = off */
	float growth_max;                 /* ceiling, % */
	float takeover_ms;                /* ms on screen before the lot is covered */

	float outline;                    /* px, outward from the box; 0 = off */
	char  outline_color[32];
	float outline_rgb[3];
	float shadow;                     /* px of blur radius; 0 = off */
	float shadow_opacity;             /* 0..1 */
	char  censor_color[32];           /* "auto", a name, or #RRGGBB */
	int   censor_auto;                /* colour sampled from the surroundings */
	float censor_rgb[3];
	int   censor[BC_MAX_CLASSES];     /* per-class policy, indexed by model class */

	/*
	 * Which effect covers each class, or -1 for "whatever `effect` says".
	 *
	 * MMCensor's settings file is a list of effect instances each owning a
	 * class set (DESIGN.md §9), which is strictly more expressive than one
	 * global effect: a bar over faces and a pixelate over everything else is
	 * the obvious thing to want and was not sayable before.
	 *
	 * Stored per class rather than as a list of instances because the class
	 * table is already the unit everything else here is keyed by -- censor[]
	 * above, the GUI's grid, the retained block. A list would need its own
	 * identity and ordering rules to say the same thing.
	 *
	 * -1 rather than a copy of `effect` so that changing the global effect
	 * still moves every class that was never given one of its own. A class
	 * pinned to an effect is opting out; a class left alone is not opting
	 * into the value that happened to be current when the file was written.
	 */
	int   effect_for[BC_MAX_CLASSES];

	/*
	 * The score a detection of this class must reach to open a track, or -1
	 * for "whatever `confidence` says".
	 *
	 * One global floor is a single compromise across classes whose cost of
	 * being missed is wildly different. The floor that keeps a noisy class
	 * from flickering is far too high for the class this program exists to
	 * cover, and picking one number means accepting the worse half of that
	 * trade everywhere. Per class, each can sit where its own miss cost puts
	 * it -- low where a miss is the failure, higher where a false positive
	 * is merely annoying.
	 *
	 * -1 rather than a copy of `confidence`, for the same reason effect_for
	 * above is: a class left alone must keep following the global knob, not
	 * freeze at whatever it happened to be when the file was written.
	 *
	 * Written into the class line as a `@` suffix -- `FACE = fill@0.30`,
	 * `BELLYEXPOSED = true@0.45` -- so it rides along with the retained
	 * block and an old config keeps parsing unchanged.
	 */
	float conf_for[BC_MAX_CLASSES];

	/* Hand-drawn regions; see struct bc_region. */
	struct bc_region regions[BC_MAX_REGIONS];
	int   n_regions;

	/*
	 * Detector to load, as a path. Empty means "whatever was named on the
	 * command line", so an existing config keeps working and the flag still
	 * wins on the run that sets it. Changing this while running swaps the
	 * detector -- see the swap in bc_main.c, which is not just a reload:
	 * the taxonomy, the class table and the tensor geometry all change with
	 * it.
	 */
	char  model[512];

	/*
	 * A second-stage landmark model, used only to locate eyes inside faces
	 * the primary detector already found (see landmark.h). Empty means the
	 * eye bar falls back to a band across the face box.
	 *
	 * Separate from `model` because the two answer different questions and
	 * are chosen independently: the detector decides *what* is covered, and
	 * this only decides where the bar sits within a face. Nothing here ever
	 * reaches the censor policy.
	 */
	char  landmark_model[512];
	/*
	 * Optional segmentation model. Its taxonomy is COCO and is never adopted:
	 * it is a *shape* source, not a detector. Empty means off, and off means
	 * every cover is the full rectangle it has always been.
	 */
	char  seg_model[512];

	/*
	 * How far the silhouette is grown before it clips anything, in percent of
	 * the screen's short side. A mask is the one thing here that removes
	 * covered area, and a silhouette that hugs the subject exactly leaves a
	 * rim of them outside it, so this errs outward on purpose.
	 */
	float seg_margin;
	/*
	 * How much bigger than the measured eye extents the bar is drawn. Taste
	 * rather than fact -- the landmarks say where the eyes are, and how much
	 * margin reads as "covered" differs by art style more than the eye
	 * positions do.
	 */
	float lm_bar_pad;

	/*
	 * Text drawn by the caption and wall-of-text effects. Empty means
	 * "use the detected class's own name" for caption, and random letters
	 * for the wall -- neither of which affects coverage, since both effects
	 * draw an opaque base before any glyph.
	 */
	char  caption[64];

	/*
	 * Colour of that text, as written in the file: "auto", "rainbow", a
	 * #RRGGBB hex triplet, or a colour name. Kept verbatim so saving does
	 * not rewrite the user's spelling of it, and resolved once at parse
	 * time into the two fields below -- see bc_text_color().
	 *
	 * text_mode is 0 for "auto", which is what an unset key gives, so the
	 * caption and the wall keep the colours they have always had.
	 */
	char  text_color[32];
	int   text_mode;                  /* 0 auto, 1 solid, 2 moving hue */
	float text_rgb[3];                /* used when text_mode == 1 */

	/*
	 * Class settings from the file that no class of the *current* model
	 * matched. Kept verbatim and written back out, so switching models does
	 * not discard the other taxonomy's toggles -- see bc_config_save().
	 */
	char  retained[BC_MAX_CLASSES][80];
	int   n_retained;

	/* Not user-facing: was the key present, or are we running on its default? */
	int   saw_smoothing;
	int   saw_reverse;
	int   saw_join;
	int   saw_eye_bar;
	int   saw_model;
	int   saw_landmark;      /* landmark_model */
	int   saw_seg;           /* seg_model + seg_margin */
	int   saw_regions;       /* the [regions] section existed */
	int   saw_draw;          /* draw */
	int   saw_bar_face;      /* eye_bar_face */
	int   saw_lm_pad;        /* eye_bar_pad */
	int   saw_bar_effect;    /* eye_bar_effect */
	int   saw_caption;
	int   saw_text_color;
	int   saw_sections;
	int   saw_full_frame;
	int   saw_split;                 /* section_split */
	int   saw_strengths;
	int   saw_shape;
	int   saw_grow;                  /* outline / shadow / censor colour */
	int   saw_delay;                 /* delay_ms */
	/*
	 * A class line is in the wrong section: one of this model's own filed
	 * under [classes.inactive], or another model's loose in [classes]. The
	 * core does not care -- it looks keys up by name wherever they are -- but
	 * the GUI reads the sections to know which classes the loaded model
	 * actually has, so a misfiled line becomes a checkbox that is missing or
	 * one that does nothing. Triggers a rewrite, which sorts them.
	 */
	int   misfiled;

	/* Not user-facing; used to detect that the file changed. */
	long  mtime;
	char  path[512];
};

/*
 * Load from `path`. A missing file is not an error -- defaults are used and the
 * file is written out, so first run produces something for the GUI to edit.
 */
int bc_config_load(struct bc_config *c, const char *path);

/* Re-read only if the file's mtime moved. Returns 1 if reloaded. */
int bc_config_poll(struct bc_config *c);

int bc_config_save(const struct bc_config *c);

/*
 * The effect index that should cover class `cls` -- its own if it has one, the
 * global `effect` otherwise. An out-of-range class gets the global effect, so a
 * caller holding a class index from before a model swap draws something rather
 * than reading off the end of the table.
 */
int bc_cfg_class_effect(const struct bc_config *c, int cls);

/*
 * Does this class name an effect of its own? In reverse mode that is what
 * marks it as covered rather than revealed -- see bc_main.c where the box is
 * built, and DESIGN.md "Reverse censoring".
 */
int bc_cfg_class_pinned(const struct bc_config *c, int cls);

/*
 * The detection floor for class `cls` -- its own if it has one, the global
 * `confidence` otherwise. Always in (0,1), whatever the file said.
 */
float bc_cfg_class_conf(const struct bc_config *c, int cls);

void bc_config_defaults(struct bc_config *c);

#endif /* BC_CONFIG_H */
