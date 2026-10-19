#include "config.h"
#include "preprocess.h"   /* BC_MAX_GRID */
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int parse_text_color(const char *s, int *mode, float rgb[3]);

/*
 * Defaults mirror §2's bias: cover what is exposed, and prefer a false positive
 * to a false negative. Faces and eyes are on with them -- anonymity is a goal of
 * this project in its own right, so a face left uncovered is the failure, not
 * the caution. Feet, bellies and armpits stay off; those are neither.
 */
void bc_config_defaults(struct bc_config *c)
{
	memset(c->censor, 0, sizeof c->censor);
	/* -1 = "use the global effect". Not zero: zero is a real effect. */
	for (int i = 0; i < BC_MAX_CLASSES; i++)
		c->effect_for[i] = -1;
	/* -1 = "use the global confidence". Zero would mean "cover anything". */
	for (int i = 0; i < BC_MAX_CLASSES; i++)
		c->conf_for[i] = -1.0f;
	c->n_retained = 0;
	c->saw_smoothing = 0;
	c->saw_reverse = 0;
	c->saw_join = 0;
	c->saw_eye_bar = 0;
	c->saw_model   = 0;
	c->saw_landmark = 0;
	c->saw_lm_pad = 0;
	c->saw_bar_effect = 0;
	c->saw_caption = 0;
	c->saw_text_color = 0;
	c->saw_sections = 0;
	c->saw_full_frame = 0;
	c->saw_shape = 0;
	c->saw_strengths = 0;
	c->saw_grow = 0;
	c->misfiled = 0;
	c->enabled    = 1;
	c->draw       = 0;
	c->eye_bar_face = 0;
	c->reverse    = 0;
	c->eye_bar    = 0;
	c->eye_bar_effect = -1;   /* bar replaces the face, as it always has */
	c->model[0]   = 0;
	c->landmark_model[0] = 0;
	c->seg_model[0] = 0;
	c->n_regions    = 0;
	c->saw_regions  = 0;
	c->saw_draw     = 0;
	c->saw_bar_face = 0;
	c->seg_margin   = 2.0f;
	c->lm_bar_pad = 1.45f;
	snprintf(c->caption, sizeof c->caption, "CENSORED");
	snprintf(c->text_color, sizeof c->text_color, "auto");
	parse_text_color(c->text_color, &c->text_mode, c->text_rgb);
	c->effect     = BC_CFG_PIXELATE;
	c->strength   = 8.0f;
	c->confidence = 0.35f;
	c->dilate     = 12.0f;
	/*
	 * Track memory: how long a track is predicted through a gap before it is
	 * given up on.
	 *
	 * This used to be a sticky timer -- the only thing standing between the
	 * user and flicker -- and it could not do that job. The detector's score
	 * for content it is already looking at oscillates across the threshold,
	 * so a box died on every dip and sticky was papering over it. Tuning
	 * only traded flicker for boxes outliving their content: transitions per
	 * 10 s went 500ms:1, 300:5, 200:5, 150:27, 100:43, 80:75, and no value
	 * was right because sticky could not tell "the detector blinked" from
	 * "the content left". 100 ms was the answer once score hysteresis
	 * removed that trade at the source.
	 *
	 * ByteTrack changed what this means (see src/track.c). It is no longer a
	 * grace period on a frozen box: the Kalman filter *predicts* the track
	 * through the gap, so the question is now how long a prediction stays
	 * credible. The old tuning does not carry over, and 100 ms was far too
	 * short -- a track died mid-gap and the next detection opened a new one
	 * rather than resuming it, which is precisely what churn counts.
	 *
	 * Re-measured on the same content, 12 s per arm: 100 ms gave 1.0
	 * expiries/s and 0.8 px of box jump, 400 gave 0.2/s and 0.4 px, 800 gave
	 * 0.0/s and 1.0 px. Box-count changes were identical (11) at all three,
	 * so the improvement is track continuity, not boxes lingering.
	 *
	 * 400 ms is the knee. 800 reaches zero churn only by never expiring
	 * anything, and its 1597 px worst-case box movement is tracks being
	 * carried far from where they were last actually seen -- which is
	 * covering content that has gone, the failure in the other direction.
	 */
	c->sticky_ms  = 400.0f;
	c->join       = 1;
	c->join_waste = 0.35f;
	c->section_cols = 1;
	c->section_rows = 1;
	/*
	 * 10% of a tile, as Hotscreen uses. The overlap is what gives a
	 * detection that straddles a seam enough of itself in each tile to
	 * still be recognised; too little and objects on the boundary are
	 * simply missed, which is a hole in the coverage rather than a
	 * cosmetic flaw.
	 */
	c->section_overlap = 0.10f;
	/*
	 * On, because it costs nothing at the 1x1 default -- there it is
	 * suppressed as a duplicate of the only tile -- and because without it
	 * turning a grid *on* silently loses everything too large to fit in one
	 * tile. Trading big detections for small ones is not a trade anyone
	 * asked for, and §2 says prefer covering too much.
	 */
	c->section_full_frame = 1;

	/*
	 * Off by default. An outline is the sort of thing that has to be asked
	 * for -- switched on unasked it would look like the censor had started
	 * drawing frames around things for a reason.
	 */
	c->growth = 0.0f;
	c->growth_max = 60.0f;
	c->takeover_ms = 0.0f;
	c->outline = 0.0f;
	snprintf(c->outline_color, sizeof c->outline_color, "%s", "black");
	c->outline_rgb[0] = c->outline_rgb[1] = c->outline_rgb[2] = 0.0f;
	c->shadow = 0.0f;
	c->shadow_opacity = 0.55f;
	snprintf(c->censor_color, sizeof c->censor_color, "%s", "black");
	c->censor_auto = 0;
	c->censor_rgb[0] = c->censor_rgb[1] = c->censor_rgb[2] = 0.0f;

	/*
	 * Per-effect defaults. Not one number copied fifteen times: a blur wants
	 * a smaller radius than a pixelate wants a block, and a triangle facet
	 * that matches a pixel block is far too fine to read as facets.
	 */
	for (int e = 0; e < BC_CFG_LAST_; e++)
		c->strength_by[e] = 8.0f;
	c->strength_by[BC_CFG_PIXELATE] = 11.0f;
	c->strength_by[BC_CFG_BLUR]     = 6.0f;
	c->strength_by[BC_CFG_HALFTONE] = 9.0f;
	c->strength_by[BC_CFG_SOBEL]    = 7.0f;
	c->strength_by[BC_CFG_TRIANGLE] = 16.0f;
	c->strength_by[BC_CFG_SHAPE]    = 12.0f;
	/*
	 * Inward edge ease. Measured on anime y11s-640, 20 s per arm on the same
	 * content: alpha 0.35 gave a drawn-box jump of 29.2 px, 0.20 gave 21.9,
	 * 0.10 gave 16.7, with coverage flat at 94.9% throughout -- so by the
	 * counters it is nearly free. The cost is not in the counters: easing is
	 * inward-only, so a lower value only delays a box *shrinking*, and at
	 * 0.10 contraction takes about 0.7 s, which reads as covers lingering
	 * over content that has gone. 0.15 sits at the knee; it is a taste knob,
	 * which is why it is in the config rather than a constant.
	 */
	c->smoothing  = 0.15f;

	/*
	 * Faces and eyes are on by default alongside the exposed classes:
	 * anonymity is a first-class goal here, not an extra. Only one spelling
	 * of each is needed -- bc_class_index() matches a reordered name, so
	 * FACE_FEMALE also selects a model's FEMALE_FACE. EYE exists only in
	 * the y11-derived taxonomies; on models without it the lookup simply
	 * misses, which is why this list is applied by name rather than index.
	 */
	static const char *on[] = {
		"BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
		"FEMALE_GENITALIA_EXPOSED", "ANUS_EXPOSED",
		"MALE_GENITALIA_EXPOSED",
		"FACE_FEMALE", "FACE_MALE", "EYE",
		/* puryfi YOLO26m (Pure Vision 2026-07) splits nipples out of
		 * the breast classes; the exposed one is as explicit as the
		 * breast it sits in. Its covered twin, MOUTH and the HAND pair
		 * stay off with the other clothed and incidental classes. */
		"NIPPLE_EXPOSED",
		/* Dedicated face models have a single ungendered class. Without
		 * this they would load and censor nothing at all, since an
		 * unlisted class defaults to off. */
		"FACE",
		/*
		 * booru_yolo's 26-class taxonomy shares no names with the
		 * others, so without these it loaded and censored *nothing* --
		 * the same silent failure as an unknown class name, just across
		 * a whole model. Semantics are the author's (models/README.md),
		 * not guessed:
		 *
		 *   BOOB   uncovered breasts        SIDEB  partial exposure
		 *   NOPAN  visible intimate area    ASS    uncovered buttocks
		 *   HIP    full hip visibility
		 *   SPRD VSPLT VSPRD  spread-leg poses, the explicit end
		 *   JACKO JACKX       presenting pose
		 *
		 * Deliberately left off, matching how the covered/exposed split
		 * is handled above: BUST and BUTT (clothed counterparts of BOOB
		 * and ASS), SHLD, BELLY, WING, FERAL, and SPLIT, the mildest of
		 * the four spread poses.
		 */
		"BOOB", "SIDEB", "NOPAN", "ASS", "HIP",
		"SPRD", "VSPLT", "VSPRD", "JACKO", "JACKX",
		/* Heads, human and anthro. Faces are first-class here, and an
		 * anthro head is a face for this purpose. */
		"HEAD", "HDRAGO", "HPONY", "HFOX", "HRABB",
		"HCAT", "HBEAR", "HHORSE", "HBIRD",
	};
	for (unsigned i = 0; i < sizeof on / sizeof on[0]; i++) {
		int idx = bc_class_index(on[i]);
		if (idx >= 0)
			c->censor[idx] = 1;
	}
}

/*
 * Resolve a text colour into the mode/rgb pair the shader wants.
 *
 * Anything unrecognised resolves to "auto" rather than to black. A typo must
 * not paint the glyphs the same colour as the bar they sit on: that reads as
 * the text having vanished, and the user's next move would be to go looking
 * for a bug in the effect. Returns 0 if the spelling was understood.
 */
static int parse_text_color(const char *s, int *mode, float rgb[3])
{
	static const struct { const char *name; unsigned hex; } named[] = {
		{ "white",   0xffffff }, { "black",   0x000000 },
		{ "red",     0xff2020 }, { "green",   0x30ff50 },
		{ "blue",    0x4080ff }, { "yellow",  0xffe020 },
		{ "cyan",    0x30ffe0 }, { "magenta", 0xff40c0 },
		{ "orange",  0xff8020 }, { "pink",    0xff90c0 },
		{ "grey",    0x909090 }, { "gray",    0x909090 },
	};

	*mode = 0;
	rgb[0] = rgb[1] = rgb[2] = 1.0f;

	if (!s || !*s || !strcasecmp(s, "auto") || !strcasecmp(s, "default"))
		return 0;
	if (!strcasecmp(s, "rainbow") || !strcasecmp(s, "rgb") ||
	    !strcasecmp(s, "wave")) {
		*mode = 2;
		return 0;
	}

	unsigned hex = 0;
	int matched = 0;
	if (*s == '#' || !strncasecmp(s, "0x", 2)) {
		const char *h = (*s == '#') ? s + 1 : s + 2;
		char *end;
		unsigned long v = strtoul(h, &end, 16);
		/* Exactly six digits and nothing after them. Three-digit CSS
		 * shorthand is not accepted rather than guessed at, because
		 * "#abc" read as 0x000abc is a plausible-looking near-black. */
		if (end == h + 6 && !*end) {
			hex = (unsigned)v;
			matched = 1;
		}
	} else {
		for (unsigned i = 0; i < sizeof named / sizeof named[0]; i++)
			if (!strcasecmp(s, named[i].name)) {
				hex = named[i].hex;
				matched = 1;
				break;
			}
	}

	if (!matched)
		return -1;

	*mode  = 1;
	rgb[0] = (float)((hex >> 16) & 0xff) / 255.0f;
	rgb[1] = (float)((hex >>  8) & 0xff) / 255.0f;
	rgb[2] = (float)( hex        & 0xff) / 255.0f;
	return 0;
}

const char *bc_effect_name(int e)
{
	switch (e) {
	case BC_CFG_FILL:     return "fill";
	case BC_CFG_BLUR:     return "blur";
	case BC_CFG_NOISE:    return "noise";
	case BC_CFG_GLITCH:   return "glitch";
	case BC_CFG_RGBSHIFT: return "rgbshift";
	case BC_CFG_TAPE:     return "tape";
	case BC_CFG_HALFTONE: return "halftone";
	case BC_CFG_SOBEL:    return "sobel";
	case BC_CFG_TRIANGLE: return "triangle";
	case BC_CFG_CLUSTER:  return "cluster";
	case BC_CFG_SPLATTER: return "splatter";
	case BC_CFG_SHAPE:    return "shape";
	case BC_CFG_CAPTION:  return "caption";
	case BC_CFG_WALL:     return "wall";
	default:           return "pixelate";
	}
}

/*
 * Was this class given an effect of its own, rather than following the global
 * one? Reverse mode reads this as "cover it instead of revealing it", so it is
 * a policy question and not merely a lookup -- hence its own name.
 */
int bc_cfg_class_pinned(const struct bc_config *c, int cls)
{
	if (!c || cls < 0 || cls >= BC_MAX_CLASSES)
		return 0;
	return c->effect_for[cls] >= 0 && c->effect_for[cls] < BC_CFG_LAST_;
}

/*
 * The score a detection of `cls` must reach, falling back to the global floor.
 *
 * Never returns zero or worse however the config got into that state: a floor
 * of zero admits every channel the model emitted at any score, which is a
 * screen covered corner to corner rather than a censor.
 */
float bc_cfg_class_conf(const struct bc_config *c, int cls)
{
	float s = -1.0f;
	if (c && cls >= 0 && cls < BC_MAX_CLASSES)
		s = c->conf_for[cls];
	if (!(s > 0.0f) || s >= 1.0f)
		s = c ? c->confidence : 0.05f;
	return (s > 0.0f && s < 1.0f) ? s : 0.05f;
}

int bc_cfg_class_effect(const struct bc_config *c, int cls)
{
	int e = -1;
	if (cls >= 0 && cls < BC_MAX_CLASSES)
		e = c->effect_for[cls];
	if (e < 0 || e >= BC_CFG_LAST_)
		e = c->effect;
	/* The result indexes strength_by[]; never hand back an out-of-range
	 * effect, however the config got into that state. */
	return (e >= 0 && e < BC_CFG_LAST_) ? e : BC_CFG_PIXELATE;
}

static char *trim(char *s)
{
	while (*s == ' ' || *s == '\t')
		s++;
	char *e = s + strlen(s);
	while (e > s && (e[-1] == ' ' || e[-1] == '\t' ||
	                 e[-1] == '\n' || e[-1] == '\r'))
		*--e = 0;
	return s;
}

static int truthy(const char *v)
{
	return !strcasecmp(v, "true") || !strcasecmp(v, "yes") ||
	       !strcasecmp(v, "on")   || !strcmp(v, "1");
}

static long file_mtime(const char *path)
{
	struct stat st;
	return stat(path, &st) == 0 ? (long)st.st_mtime : 0;
}

static int parse(struct bc_config *c, FILE *f, int *seen)
{
	/*
	 * Long enough for the longest line the writer can produce, which is an
	 * outline: BC_MAX_POLY_PTS pairs at "0.1234,0.5678 " apiece. fgets()
	 * does not report truncation -- it just returns the head and hands the
	 * tail back on the next call, where it parses as a nameless line and is
	 * silently dropped. For an outline that would mean a shape quietly
	 * losing its far half, so the buffer is sized off the format rather
	 * than off a round number.
	 */
	char line[BC_MAX_POLY_PTS * 16 + 512];
	int inactive_section = 0;
	int region_section = 0;
	while (fgets(line, sizeof line, f)) {
		char *s = trim(line);
		if (*s == '[') {
			/* Sections carry no meaning for looking a key up -- every
			 * key is unique across the file -- but which one a class
			 * landed in has to be right, for the GUI. */
			inactive_section = !strncasecmp(s, "[classes.inactive]", 18);
			region_section   = !strncasecmp(s, "[regions]", 9);
			if (region_section)
				c->saw_regions = 1;
			continue;
		}
		if (!*s || *s == '#' || *s == ';')
			continue;

		char *eq = strchr(s, '=');
		if (!eq)
			continue;
		*eq = 0;
		char *k = trim(s), *v = trim(eq + 1);

		/*
		 * Strip a trailing comment. Numeric values happened to survive
		 * without this because strtof() stops at the '#', but booleans
		 * did not: truthy("true   # explanation") is false, so a line
		 * like `enabled = true  # master switch` silently turned
		 * censoring *off*. Fail-open, and invisible -- the key parsed
		 * fine, so nothing warned.
		 *
		 * The marker has to be *preceded by whitespace* to count, which
		 * is the usual INI rule and here a necessary one: a colour is
		 * written #RRGGBB, so a '#' at the head of a value is data. Cut
		 * on it and `text_color = #40c0ff` becomes the empty string,
		 * which resolves to "auto" -- the setting silently ignored, with
		 * the file still plainly showing the colour that was asked for.
		 */
		for (char *c = v; *c; c++)
			if ((*c == '#' || *c == ';') && c > v &&
			    (c[-1] == ' ' || c[-1] == '\t')) {
				*c = 0;
				break;
			}
		v = trim(v);

		/*
		 * A value that is *nothing but* a comment is an empty setting.
		 *
		 * The rule above needs whitespace before the marker so that
		 * `text_color = #40c0ff` keeps its colour -- but that also means
		 * a comment sitting at the head of the value survives, and after
		 * trim() a comment is at the head exactly when the value was
		 * empty. So `seg_model =    # optional: ...` read back as the
		 * *comment text*, which was then written out as the value, which
		 * doubled the comment and turned an unset optional model into a
		 * path made of prose. Every key here written with a comment and
		 * commonly left blank had it -- seg_model showed it first only
		 * because it is the first one usually empty.
		 *
		 * A colour is still safe: #RRGGBB has no space after the hash
		 * and a comment written by anything here does.
		 */
		if ((*v == '#' || *v == ';') &&
		    (v[1] == ' ' || v[1] == '\t' || v[1] == 0))
			*v = 0;
		if (!*k)
			continue;

		/*
		 * A drawn region: "x,y,w,h" normalised, with an optional effect
		 * name. The key is only a label -- these have no identity beyond
		 * their order, and naming them r1, r2 keeps the file legible and
		 * hand-editable, which is the whole interface until there is a
		 * picker.
		 *
		 * Anything unparseable is dropped with a warning rather than
		 * clamped into existence: a region at the wrong place covers the
		 * wrong thing, and a silent half-parse is how you get a bar
		 * sitting in the corner for no visible reason.
		 */
		if (region_section) {
			/*
			 * An outline: "poly x,y x,y ..." with an optional
			 * effect written as "poly:blur". The points are the
			 * shape; the bounding box everything downstream reads
			 * is derived from them here, so the two can never
			 * disagree in the file.
			 *
			 * A closed ring is implied -- the last point joins the
			 * first, and is not written twice. Drawing mode only
			 * ever produces closed shapes, so an explicit repeat
			 * would be a duplicate point in every outline for the
			 * sake of a case that cannot occur.
			 */
			if (!strncasecmp(v, "poly", 4) &&
			    (v[4] == 0 || v[4] == ':' || v[4] == ' ' || v[4] == '\t')) {
				char *p = v + 4;
				int   eff_id = -1;
				if (*p == ':') {
					char name[32] = { 0 };
					p++;
					int i = 0;
					while (*p && *p != ' ' && *p != '\t' &&
					       i < (int)sizeof name - 1)
						name[i++] = *p++;
					for (int e = 0; e < BC_CFG_LAST_; e++)
						if (!strcasecmp(name, bc_effect_name(e))) {
							eff_id = e;
							break;
						}
					if (eff_id < 0)
						WARN("config: outline '%s' names effect "
						     "'%s', which is not one -- using "
						     "the global effect", k, name);
				}

				if (c->n_regions >= BC_MAX_REGIONS) {
					WARN("config: more than %d regions -- '%s' "
					     "ignored", BC_MAX_REGIONS, k);
					continue;
				}

				float xs[BC_MAX_POLY_PTS], ys[BC_MAX_POLY_PTS];
				int n = 0, over = 0;
				while (*p) {
					float ax, ay;
					int used = 0;
					if (sscanf(p, " %f , %f%n", &ax, &ay, &used) < 2)
						break;
					p += used;
					if (n < BC_MAX_POLY_PTS) {
						xs[n] = ax;
						ys[n] = ay;
						n++;
					} else {
						over = 1;
					}
				}
				if (over)
					WARN("config: outline '%s' has more than %d "
					     "points -- the rest are dropped",
					     k, BC_MAX_POLY_PTS);
				/*
				 * Three points is the minimum that encloses
				 * anything. Fewer is a line, which has no
				 * inside to cover, and rasterising it would
				 * produce an empty mask -- a region that exists
				 * in the file and covers nothing on screen,
				 * which is the silent half-parse this section
				 * refuses to do.
				 */
				if (n < 3) {
					WARN("config: outline '%s' has %d point(s) "
					     "-- at least 3 enclose an area, so it "
					     "is ignored", k, n);
					continue;
				}

				float x0 = xs[0], y0 = ys[0], x1 = xs[0], y1 = ys[0];
				for (int i = 1; i < n; i++) {
					if (xs[i] < x0) x0 = xs[i];
					if (ys[i] < y0) y0 = ys[i];
					if (xs[i] > x1) x1 = xs[i];
					if (ys[i] > y1) y1 = ys[i];
				}
				if (!(x1 - x0 > 0.0f) || !(y1 - y0 > 0.0f)) {
					WARN("config: outline '%s' has no area "
					     "-- ignored", k);
					continue;
				}

				struct bc_region *r = &c->regions[c->n_regions++];
				r->x = x0; r->y = y0;
				r->w = x1 - x0; r->h = y1 - y0;
				r->effect = eff_id;
				r->n_pts = n;
				for (int i = 0; i < n; i++) {
					r->px[i] = xs[i];
					r->py[i] = ys[i];
				}
				continue;
			}

			float rx, ry, rw, rh;
			char  eff[32] = { 0 };
			const int got = sscanf(v, "%f , %f , %f , %f , %31s",
			                       &rx, &ry, &rw, &rh, eff);
			if (got < 4) {
				WARN("config: region '%s' is not x,y,w,h -- ignored", k);
				continue;
			}
			if (c->n_regions >= BC_MAX_REGIONS) {
				WARN("config: more than %d regions -- '%s' ignored",
				     BC_MAX_REGIONS, k);
				continue;
			}
			if (!(rw > 0.0f) || !(rh > 0.0f)) {
				WARN("config: region '%s' has no area -- ignored", k);
				continue;
			}
			struct bc_region *r = &c->regions[c->n_regions++];
			r->x = rx; r->y = ry; r->w = rw; r->h = rh;
			r->effect = -1;
			r->n_pts = 0;   /* a plain rectangle */
			if (got >= 5 && eff[0]) {
				for (int e = 0; e < BC_CFG_LAST_; e++)
					if (!strcasecmp(eff, bc_effect_name(e))) {
						r->effect = e;
						break;
					}
				if (r->effect < 0)
					WARN("config: region '%s' names effect '%s', "
					     "which is not one -- using the global "
					     "effect", k, eff);
			}
			continue;
		}

		if (!strcasecmp(k, "enabled"))         c->enabled = truthy(v);
		else if (!strcasecmp(k, "draw")) {
			c->draw = truthy(v);
			c->saw_draw = 1;
		}
		else if (!strcasecmp(k, "eye_bar_face")) {
			c->eye_bar_face = truthy(v);
			c->saw_bar_face = 1;
		}
		else if (!strcasecmp(k, "join")) {
			c->join = truthy(v);
			c->saw_join = 1;
		}
		else if (!strcasecmp(k, "reverse")) {
			c->reverse = truthy(v);
			c->saw_reverse = 1;
		}
		else if (!strcasecmp(k, "model")) {
			snprintf(c->model, sizeof c->model, "%s", v);
			c->saw_model = 1;
		}
		else if (!strcasecmp(k, "eye_bar_effect")) {
			/* An effect name makes the bar additive; anything else
			 * -- "off", "none", a typo -- leaves it replacing the
			 * face, which is the behaviour that has always been
			 * there and cannot surprise anyone. */
			c->eye_bar_effect = -1;
			for (int e = 0; e < BC_CFG_LAST_; e++)
				if (!strcasecmp(v, bc_effect_name(e))) {
					c->eye_bar_effect = e;
					break;
				}
			c->saw_bar_effect = 1;
		}
		else if (!strcasecmp(k, "seg_model")) {
			snprintf(c->seg_model, sizeof c->seg_model, "%s", v);
			c->saw_seg = 1;
		}
		else if (!strcasecmp(k, "seg_margin")) {
			c->seg_margin = (float)atof(v);
			if (c->seg_margin < 0.0f)  c->seg_margin = 0.0f;
			if (c->seg_margin > 25.0f) c->seg_margin = 25.0f;
		}
		else if (!strcasecmp(k, "landmark_model")) {
			snprintf(c->landmark_model, sizeof c->landmark_model,
			         "%s", v);
			c->saw_landmark = 1;
		}
		else if (!strcasecmp(k, "eye_bar_pad")) {
			c->lm_bar_pad = strtof(v, NULL);
			c->saw_lm_pad = 1;
		}
		else if (!strcasecmp(k, "caption")) {
			snprintf(c->caption, sizeof c->caption, "%s", v);
			c->saw_caption = 1;
		}
		else if (!strcasecmp(k, "text_color") ||
		         !strcasecmp(k, "text_colour")) {
			if (parse_text_color(v, &c->text_mode, c->text_rgb) == 0)
				snprintf(c->text_color, sizeof c->text_color,
				         "%s", v);
			else
				WARN("config: text_color '%s' not understood -- "
				     "using the effect's own colour. Try auto, "
				     "rainbow, a name like cyan, or #RRGGBB", v);
			c->saw_text_color = 1;
		}
		else if (!strcasecmp(k, "eye_bar")) {
			c->eye_bar = truthy(v);
			c->saw_eye_bar = 1;
		}
		else if (!strcasecmp(k, "effect")) {
			/* Unknown names fall back to pixelate rather than
			 * failing: a typo must still cover something. */
			c->effect = BC_CFG_PIXELATE;
			for (int e = 0; e < BC_CFG_LAST_; e++)
				if (!strcasecmp(v, bc_effect_name(e))) {
					c->effect = e;
					break;
				}
		}
		else if (!strcasecmp(k, "strength")) {
			/*
			 * The old single key. Kept so an existing file still
			 * says something: it seeds every effect that has no key
			 * of its own yet, and the per-effect lines below
			 * override it wherever they appear.
			 */
			c->strength = strtof(v, NULL);
			for (int e = 0; e < BC_CFG_LAST_; e++)
				c->strength_by[e] = c->strength;
		}
		else if (!strncasecmp(k, "strength_", 9)) {
			for (int e = 0; e < BC_CFG_LAST_; e++)
				if (!strcasecmp(k + 9, bc_effect_name(e))) {
					c->strength_by[e] = strtof(v, NULL);
					c->saw_strengths = 1;
					break;
				}
		}
		else if (!strcasecmp(k, "confidence")) c->confidence = strtof(v, NULL);
		else if (!strcasecmp(k, "dilate"))     c->dilate     = strtof(v, NULL);
		else if (!strcasecmp(k, "sticky_ms"))  c->sticky_ms  = strtof(v, NULL);
		else if (!strcasecmp(k, "join_waste")) c->join_waste = strtof(v, NULL);
		else if (!strcasecmp(k, "section_cols")) {
			c->section_cols = atoi(v);
			c->saw_sections = 1;
		}
		else if (!strcasecmp(k, "section_rows")) {
			c->section_rows = atoi(v);
			c->saw_sections = 1;
		}
		else if (!strcasecmp(k, "section_overlap"))
			c->section_overlap = strtof(v, NULL);
		else if (!strcasecmp(k, "section_full_frame")) {
			c->section_full_frame = truthy(v);
			c->saw_full_frame = 1;
		}
		else if (!strcasecmp(k, "growth")) {
			c->growth = strtof(v, NULL);
			c->saw_grow = 1;
		}
		else if (!strcasecmp(k, "growth_max"))
			c->growth_max = strtof(v, NULL);
		else if (!strcasecmp(k, "takeover_ms"))
			c->takeover_ms = strtof(v, NULL);
		else if (!strcasecmp(k, "outline")) {
			c->outline = strtof(v, NULL);
			c->saw_shape = 1;
		}
		else if (!strcasecmp(k, "outline_color")) {
			int m = 0;
			/* Reuses the glyph-colour parser, so the spellings are
			 * the same everywhere. "auto"/"rainbow" have no meaning
			 * for an outline, so anything that does not resolve to a
			 * solid colour keeps the default rather than inventing
			 * one. */
			if (parse_text_color(v, &m, c->outline_rgb) == 0 && m == 1)
				snprintf(c->outline_color, sizeof c->outline_color,
				         "%s", v);
			else
				parse_text_color(c->outline_color, &m, c->outline_rgb);
		}
		else if (!strcasecmp(k, "shadow"))
			c->shadow = strtof(v, NULL);
		else if (!strcasecmp(k, "shadow_opacity"))
			c->shadow_opacity = strtof(v, NULL);
		else if (!strcasecmp(k, "censor_color")) {
			/*
			 * "auto" is a real value here, unlike for the outline:
			 * it means sample the colour from around the box.
			 */
			snprintf(c->censor_color, sizeof c->censor_color, "%s", v);
			if (!strcasecmp(v, "auto")) {
				c->censor_auto = 1;
			} else {
				int m = 0;
				c->censor_auto = 0;
				if (parse_text_color(v, &m, c->censor_rgb) != 0 || m != 1) {
					/* Unreadable: black, which is the one
					 * answer that certainly covers. */
					c->censor_rgb[0] = c->censor_rgb[1] =
					        c->censor_rgb[2] = 0.0f;
					snprintf(c->censor_color,
					         sizeof c->censor_color, "%s", "black");
				}
			}
		}
		else if (!strcasecmp(k, "smoothing")) {
			c->smoothing = strtof(v, NULL);
			c->saw_smoothing = 1;
		}
		else {
			/* Anything else is treated as a class name. */
			int idx = bc_class_index(k);
			if (idx >= 0) {
				/*
				 * A class line is either a boolean, as it has
				 * always been, or the name of the effect that
				 * should cover it -- which also means "on", so
				 * `FACE_FEMALE = fill` needs no second key.
				 *
				 * Naming an effect is checked first because the
				 * two vocabularies do not overlap: no effect is
				 * spelled "true" or "false". An unreadable value
				 * is neither, and falls through to truthy() ==
				 * off, which is the pre-existing behaviour for a
				 * line nobody can parse.
				 */
				/*
				 * An optional `@score` suffix pins this class's
				 * own detection floor. Split off first, so what
				 * is left in front is exactly the vocabulary the
				 * code below already knows; a line without one
				 * is untouched, which is what keeps every
				 * existing config parsing as it did.
				 */
				char vbuf[80];
				snprintf(vbuf, sizeof vbuf, "%s", v);
				char *at = strchr(vbuf, '@');
				if (at) {
					*at++ = 0;
					char *end = NULL;
					const float s = strtof(at, &end);
					/*
					 * Only a value that parsed and landed in
					 * range. A typo has to leave the class on
					 * the global floor rather than on zero,
					 * which would match everything the model
					 * emitted and cover the whole screen.
					 */
					c->conf_for[idx] = (end && end != at &&
					                    s > 0.0f && s < 1.0f)
					                 ? s : -1.0f;
					size_t n = strlen(vbuf);
					while (n && (vbuf[n - 1] == ' ' ||
					             vbuf[n - 1] == '\t'))
						vbuf[--n] = 0;
					v = vbuf;
				}

				int eff = -1;
				for (int e = 0; e < BC_CFG_LAST_; e++)
					if (!strcasecmp(v, bc_effect_name(e))) {
						eff = e;
						break;
					}
				c->effect_for[idx] = eff;
				c->censor[idx] = eff >= 0 ? 1 : truthy(v);
				if (seen)
					seen[idx] = 1;
				/*
				 * One of ours, filed under the other model's
				 * heading. Happens because the file is written
				 * once before the detector exists, against the
				 * built-in taxonomy, and any class the real
				 * model adds to that lands in the retained
				 * block on that first pass.
				 */
				if (inactive_section)
					c->misfiled = 1;
			}
			else {
				/* Not ours, but not filed as such -- an older
				 * file, from before the sections existed. */
				if (!inactive_section)
					c->misfiled = 1;
				/*
				 * Not a class of the model running now. It may
				 * well be one of another model's, so keep it
				 * rather than warn and drop it: dropping reset
				 * the user's toggles every time they switched
				 * models, and reset means back to the default,
				 * which for most classes is off.
				 */
				if (c->n_retained < BC_MAX_CLASSES)
					snprintf(c->retained[c->n_retained++],
					         sizeof c->retained[0],
					         "%-26s = %s", k, v);
			}
		}
	}

	/* Clamp, so a hand-edited file cannot produce nonsense. */
	if (c->strength   < 1.0f)  c->strength   = 1.0f;
	if (c->strength   > 100.f) c->strength   = 100.f;
	if (c->growth      < 0.f)    c->growth      = 0.f;
	if (c->growth      > 200.f)  c->growth      = 200.f;
	if (c->growth_max  < 0.f)    c->growth_max  = 0.f;
	if (c->growth_max  > 400.f)  c->growth_max  = 400.f;
	if (c->takeover_ms < 0.f)    c->takeover_ms = 0.f;
	for (int e = 0; e < BC_CFG_LAST_; e++) {
		if (c->strength_by[e] < 1.0f)   c->strength_by[e] = 1.0f;
		if (c->strength_by[e] > 100.0f) c->strength_by[e] = 100.0f;
	}
	if (c->confidence <= 0.f)  c->confidence = 0.05f;
	if (c->confidence >= 1.f)  c->confidence = 0.95f;
	if (c->dilate     < 0.f)   c->dilate     = 0.f;
	if (c->dilate     > 100.f) c->dilate     = 100.f;
	if (c->sticky_ms  < 0.f)   c->sticky_ms  = 0.f;
	if (c->sticky_ms  > 5000.f) c->sticky_ms = 5000.f;
	if (c->join_waste < 0.f)   c->join_waste = 0.f;
	if (c->join_waste > 4.f)   c->join_waste = 4.f;
	if (c->section_cols < 1) c->section_cols = 1;
	if (c->section_rows < 1) c->section_rows = 1;
	if (c->section_cols > BC_MAX_GRID) c->section_cols = BC_MAX_GRID;
	if (c->section_rows > BC_MAX_GRID) c->section_rows = BC_MAX_GRID;
	if (c->section_overlap < 0.f)  c->section_overlap = 0.f;
	if (c->section_overlap > 0.9f) c->section_overlap = 0.9f;
	if (c->smoothing  <= 0.f)  c->smoothing  = 0.01f;
	if (c->smoothing  > 1.f)   c->smoothing  = 1.f;

	/*
	 * `strength` is whichever per-effect value belongs to the effect in use.
	 * Resolved here, at the end of parsing, because the effect and its value
	 * can arrive in either order in the file -- and because *every* path
	 * that reads the file has to do it. It used to live in bc_config_load()
	 * alone, which left bc_config_poll() -- the live reload the GUI relies
	 * on -- handing back `c->strength` as parsed from the legacy `strength =`
	 * key in [global]. The GUI only ever writes `strength_<effect>`, so
	 * editing coarseness changed the file, reloaded, and had no effect on
	 * anything: the number the renderer read never moved off the legacy one.
	 */
	if (c->effect >= 0 && c->effect < BC_CFG_LAST_)
		c->strength = c->strength_by[c->effect];

	/*
	 * A file written before eye_bar_face existed gets the meaning that key
	 * used to be folded into: naming an effect for the bar was the only way
	 * to ask for the face to be covered too, and leaving it off was the only
	 * way to ask for the bar alone.
	 *
	 * Derived here rather than defaulted, because a plain default cannot
	 * depend on another key -- and a fixed one would silently change what
	 * every existing config does the first time it was read by this build.
	 * Resolved at the tail of parse() so the live reload gets it as well;
	 * the two keys can arrive in either order in the file.
	 */
	if (!c->saw_bar_face)
		c->eye_bar_face = c->eye_bar_effect >= 0;
	return 0;
}

int bc_config_save(const struct bc_config *c)
{
	FILE *f = fopen(c->path, "w");
	if (!f) {
		ERR("cannot write config %s", c->path);
		return -1;
	}

	char strengths[512];
	{
		int off = 0;
		for (int e = 0; e < BC_CFG_LAST_ && off < (int)sizeof strengths; e++)
			off += snprintf(strengths + off,
			                sizeof strengths - (size_t)off,
			                "strength_%-9s = %.1f\n",
			                bc_effect_name(e), c->strength_by[e]);
	}

	/*
	 * Drawn regions. The header is written even with none, so the section is
	 * discoverable in a file that has never had one -- this is the only part
	 * of the config with no GUI behind it yet, so the file has to explain
	 * itself.
	 */
	char regions[BC_MAX_REGIONS * (BC_MAX_POLY_PTS * 16 + 64) + 640];
	{
		int off = snprintf(regions, sizeof regions,
			"\n[regions]\n"
			"# Areas covered no matter what the detector says. Two shapes:\n"
			"#   <name> = x,y,w,h[,effect]        a rectangle\n"
			"#   <name> = poly[:effect] x,y x,y ...   an outline\n"
			"# All coordinates are a fraction of the screen (0..1) from the top\n"
			"# left, so they survive a resolution change. An outline's last point\n"
			"# joins its first; three points is the minimum that encloses\n"
			"# anything. The effect is optional and defaults to the global one.\n"
			"# These ask no model anything, so nothing can score them below a\n"
			"# threshold and miss them.\n"
			"# Example:  r1 = 0.10,0.20,0.30,0.25,fill\n"
			"#           p1 = poly:blur 0.10,0.20 0.30,0.22 0.25,0.40\n");
		for (int i = 0; i < c->n_regions && off < (int)sizeof regions; i++) {
			const struct bc_region *r = &c->regions[i];
			/*
			 * The two shapes spell the effect differently -- a
			 * rectangle appends it as a fifth comma-separated
			 * field, an outline hangs it off "poly:" -- because a
			 * comma already separates an outline's coordinates and
			 * a trailing one there could not be told from a point.
			 */
			const char *ename =
				(r->effect >= 0 && r->effect < BC_CFG_LAST_)
				? bc_effect_name(r->effect) : NULL;

			if (r->n_pts >= 3) {
				off += snprintf(regions + off,
					sizeof regions - (size_t)off,
					"p%-3d = poly%s%s", i + 1,
					ename ? ":" : "", ename ? ename : "");
				for (int j = 0; j < r->n_pts &&
				                off < (int)sizeof regions; j++)
					off += snprintf(regions + off,
						sizeof regions - (size_t)off,
						" %.4f,%.4f", r->px[j], r->py[j]);
				if (off < (int)sizeof regions)
					off += snprintf(regions + off,
						sizeof regions - (size_t)off, "\n");
			} else if (ename) {
				off += snprintf(regions + off,
					sizeof regions - (size_t)off,
					"r%-3d = %.4f,%.4f,%.4f,%.4f,%s\n",
					i + 1, r->x, r->y, r->w, r->h, ename);
			} else {
				off += snprintf(regions + off,
					sizeof regions - (size_t)off,
					"r%-3d = %.4f,%.4f,%.4f,%.4f\n",
					i + 1, r->x, r->y, r->w, r->h);
			}
		}
	}

	fprintf(f,
		"# BetaCensor configuration.\n"
		"# Edited by betacensor-gui; re-read automatically while running.\n"
		"\n[global]\n"
		"enabled    = %s\n"
		"draw       = %s   # true puts the overlay in drawing mode; it clears this when done\n"
		"join       = %s   # merge overlapping covers into one region\n"
		"join_waste = %.2f   # how much new area a join may add (0 = only exact overlaps)\n"
		"reverse    = %s   # mask the screen and reveal detections instead\n"
		"           #   (a class below with an effect of its own stays covered with it)\n"
		"eye_bar    = %s   # cover only the eye band of each detected face\n"
		"model      = %s   # detector to load; swapped live when this changes\n"
		"landmark_model = %s   # optional: locates eyes inside faces, for the eye bar\n"
		"seg_model  = %s   # optional: clips covers to the subject's outline instead of a box\n"
		"seg_margin = %.2f   # %% of the short side the outline is grown by before it clips\n"
		"eye_bar_pad = %.2f   # how far the eye bar extends past the eyes themselves\n"
		"eye_bar_effect = %s   # effect for the bar itself; off = it takes the face's own effect\n"
		"eye_bar_face = %s   # also cover the face under the bar; false leaves the face alone\n"
		"effect     = %s\n"
		"caption    = %s   # text for the caption and wall-of-text effects\n"
		"text_color = %s   # glyph colour: auto, rainbow, a name, or #RRGGBB\n"
		"strength   = %.1f   # pixelate block size, %% of box short side\n"
		"confidence = %.2f   # detection floor\n"
		"dilate     = %.1f   # box growth %%, covers motion since capture\n"
		"sticky_ms  = %.0f   # how long a box outlives its last detection\n"
		"smoothing  = %.2f   # inward edge ease; lower is steadier, slower to shrink\n"
		"section_cols = %d   # detection grid across; >1 costs one inference per tile\n"
		"section_rows = %d   # detection grid down\n"
		"section_overlap = %.2f   # fraction of a tile shared with its neighbour\n"
		"section_full_frame = %s   # also infer the whole screen, to catch what a tile cannot hold\n"
		"censor_color = %s   # bar colour: auto (sampled from around it), a name, or #RRGGBB\n"
		"outline    = %.1f   # px of outline around each cover, drawn outward; 0 = off\n"
		"outline_color = %s   # outline colour: a name or #RRGGBB\n"
		"shadow     = %.1f   # px of drop shadow beyond the outline; 0 = off\n"
		"shadow_opacity = %.2f   # 0..1\n"
		"growth     = %.1f   # %% per second each cover grows while on screen; 0 = off\n"
		"growth_max = %.1f   # ceiling for growth, %%\n"
		"takeover_ms = %.0f   # ms a detection must persist before the whole screen is covered; 0 = off\n"
		"\n[strength]\n"
		"# Block size, blur radius, dot pitch -- one per effect, because they\n"
		"# are different quantities and want different values. The active\n"
		"# effect's line is the one that has any bearing.\n"
		"%s"
		"%s"
		"\n[classes]\n",
		c->enabled ? "true" : "false",
		c->draw ? "true" : "false",
		c->join ? "true" : "false", c->join_waste,
		c->reverse ? "true" : "false",
		c->eye_bar ? "true" : "false",
		c->model, c->landmark_model, c->seg_model, c->seg_margin,
		c->lm_bar_pad,
		c->eye_bar_effect >= 0 && c->eye_bar_effect < BC_CFG_LAST_
		        ? bc_effect_name(c->eye_bar_effect) : "off",
		c->eye_bar_face ? "true" : "false",
		bc_effect_name(c->effect), c->caption, c->text_color,
		c->strength, c->confidence, c->dilate, c->sticky_ms, c->smoothing,
		c->section_cols, c->section_rows, c->section_overlap,
		c->section_full_frame ? "true" : "false",
		c->censor_color, c->outline, c->outline_color,
		c->shadow, c->shadow_opacity,
		c->growth, c->growth_max, c->takeover_ms, strengths, regions);

	/*
	 * A class pinned to an effect is written as that effect's name; every
	 * other line keeps the boolean spelling it has always had. Writing
	 * "true" for a pinned class would silently discard the pin on the next
	 * save, and writing an effect name for every class would bake today's
	 * global effect into all of them -- after which changing the global one
	 * would move nothing.
	 */
	for (int i = 0; i < bc_class_count(); i++) {
		const char *v = c->censor[i] ? "true" : "false";
		if (c->censor[i] && c->effect_for[i] >= 0 &&
		    c->effect_for[i] < BC_CFG_LAST_)
			v = bc_effect_name(c->effect_for[i]);
		/*
		 * The pinned floor rides on the same line, and only when there
		 * is one -- writing the global value onto every class would
		 * freeze them all at today's number, the same trap the effect
		 * pin above avoids.
		 */
		if (c->conf_for[i] > 0.0f && c->conf_for[i] < 1.0f)
			fprintf(f, "%-26s = %s@%.2f\n", bc_class_name(i), v,
			        (double)c->conf_for[i]);
		else
			fprintf(f, "%-26s = %s\n", bc_class_name(i), v);
	}

	/*
	 * Settings for classes the current model does not have. Written back so
	 * that loading a model which *does* have them picks the values up again.
	 *
	 * These go in their own section rather than trailing the live ones behind
	 * a comment. parse() skips '[' lines, so it reads them back exactly as
	 * before -- the section exists for the GUI, which has no ONNX and no way
	 * to tell a loaded model's class from a leftover otherwise. It used to
	 * guess by elimination and drew a checkbox for every one of them, so the
	 * list was three times longer than the model and most of it did nothing.
	 */
	if (c->n_retained) {
		fprintf(f, "\n[classes.inactive]\n"
		           "# Kept for other models -- not classes of the one "
		           "currently loaded.\n");
		for (int i = 0; i < c->n_retained; i++)
			fprintf(f, "%s\n", c->retained[i]);
	}

	fclose(f);
	return 0;
}

int bc_config_load(struct bc_config *c, const char *path)
{
	bc_config_defaults(c);
	snprintf(c->path, sizeof c->path, "%s", path);

	FILE *f = fopen(path, "r");
	if (!f) {
		LOG("config: %s not found, writing defaults", path);
		bc_config_save(c);
		c->mtime = file_mtime(path);
		return 0;
	}

	int seen[BC_MAX_CLASSES] = { 0 };
	parse(c, f, seen);
	fclose(f);
	c->mtime = file_mtime(path);

	/*
	 * Drawing mode is never inherited from the file. It is a request made of
	 * a running core, and the core clears it when the drawing ends -- so a
	 * file still holding `true` means the last run died mid-draw, and
	 * honouring it would start this one with the pointer and keyboard
	 * grabbed before there is anything on screen to say why. Only the live
	 * reload in bc_config_poll() can turn it on.
	 */
	c->draw = 0;

	/* `strength` is resolved from `strength_by` at the tail of parse(), so
	 * that the poll path gets it too. */

	/*
	 * Rewrite the file if it does not mention every class this model has.
	 *
	 * Models carry different taxonomies -- 16 classes for the y11 family, 18
	 * for 320n/640m, and EYE exists only in the former. A class the file
	 * never names keeps its built-in default, which the user cannot see and
	 * the GUI cannot offer, so switching models left settings that silently
	 * did not exist. Writing the current model's list back makes the file a
	 * true picture of what is running.
	 *
	 * Values already in the file win, because parse() ran first; only the
	 * missing names are added. The cost is that keys belonging to the *other*
	 * taxonomy are dropped, so switching back restores that model's defaults
	 * rather than the previous toggles.
	 */
	int missing = 0;
	for (int i = 0; i < bc_class_count(); i++)
		if (!seen[i])
			missing++;
	/*
	 * A *global* key the file never mentions is invisible in the same way a
	 * missing class is: the GUI only offers keys that are present, so a
	 * setting added after the file was written could never be reached.
	 *
	 * Counted in a loop rather than one `if` per key. Written out longhand
	 * this block grew a `return` where a `missing++` belonged, which turned
	 * the entire rewrite off -- the class list with it -- for every file
	 * written before that key existed, which is every file. The censor still
	 * worked, because keys are looked up by name wherever they sit, but the
	 * file stopped being a picture of the running model: the GUI reads the
	 * sections to know which classes exist, so it went on offering the
	 * previous model's checkboxes and no model swap could ever change them.
	 * A list cannot be exited early by accident.
	 */
	const int saw[] = { c->saw_smoothing, c->saw_join, c->saw_reverse,
	                    c->saw_model, c->saw_eye_bar, c->saw_caption,
	                    c->saw_text_color, c->saw_sections,
	                    c->saw_full_frame, c->saw_shape,
	                    c->saw_strengths, c->saw_grow,
	                    c->saw_landmark, c->saw_lm_pad, c->saw_seg,
	                    c->saw_bar_effect, c->saw_draw, c->saw_regions,
	                    c->saw_bar_face };
	for (unsigned i = 0; i < sizeof saw / sizeof saw[0]; i++)
		if (!saw[i])
			missing++;
	/*
	 * Class lines under the wrong heading. Rewriting sorts them, and happens
	 * once: the file it writes is correctly sorted, so the next load is quiet.
	 *
	 * Only judged once a model has said what its classes are. This runs a
	 * first time before any detector exists, against the built-in table, and
	 * back then a correctly sorted file looked wrong from both ends at once
	 * -- the real model's extra classes read as foreign, the built-in's
	 * absent ones as missing -- so every startup rewrote the file twice,
	 * ping-ponging it between the two taxonomies.
	 */
	if (c->misfiled && bc_class_from_model())
		missing++;
	if (missing) {
		LOG("config: %d setting(s) missing from %s -- rewriting", missing, path);
		bc_config_save(c);
		c->mtime = file_mtime(path);
	}

	int n = 0;
	for (int i = 0; i < bc_class_count(); i++)
		n += c->censor[i];
	LOG("config: %s (%d classes censored, effect=%s, conf=%.2f)",
	    path, n, bc_effect_name(c->effect),
	    c->confidence);
	/*
	 * Censoring nothing is a valid state to ask for and a disastrous one to
	 * arrive at by accident, and the accident is easy: the first run of a
	 * model with an unfamiliar taxonomy writes every one of its classes to
	 * the file as false, and an explicit false in the file beats the
	 * built-in default forever after. The program then runs, reports
	 * success, and covers nothing. Say so loudly rather than let a silent
	 * zero look like a clean screen.
	 */
	if (n == 0)
		WARN("config: NOTHING will be censored -- every class of this "
		     "model is off in %s. If this is a model you have not used "
		     "before, delete its class lines from the file to get the "
		     "defaults back", path);
	return 0;
}

int bc_config_poll(struct bc_config *c)
{
	long m = file_mtime(c->path);
	if (m == 0 || m == c->mtime)
		return 0;

	FILE *f = fopen(c->path, "r");
	if (!f)
		return 0;

	/*
	 * Reload onto defaults rather than onto the live values, so removing a
	 * line in the file actually reverts that setting instead of leaving the
	 * old value stuck.
	 */
	char keep[512];
	snprintf(keep, sizeof keep, "%s", c->path);
	bc_config_defaults(c);
	snprintf(c->path, sizeof c->path, "%s", keep);

	parse(c, f, NULL);
	fclose(f);
	c->mtime = m;
	LOG("config: reloaded");
	return 1;
}
