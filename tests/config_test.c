/*
 * The config file is the interface between the censor and the GUI. Nothing
 * else connects them: the core writes the running model's class list into
 * [classes], and the GUI draws a checkbox for each line it finds there.
 *
 * So the property that matters is not "does the file parse" but "does loading
 * it leave the file describing the model that is actually running". When that
 * stops being true the censor keeps working -- keys are looked up by name
 * wherever they sit -- and the only symptom is a GUI offering the previous
 * model's checkboxes, which is exactly the failure this exists to catch.
 *
 * The class table is stubbed rather than linked from detect.c. That is not a
 * shortcut: the scenario under test *is* a taxonomy change, and a stub is the
 * only way to swap one mid-process without an ONNX model to hand.
 */
#include <stdio.h>
#include <string.h>

/* --- stand-in for the model's taxonomy ------------------------------------ */

static const char *BOORU[] = { "HEAD", "BUST", "BOOB", "SHLD", "SIDEB" };
static const char *NUDE[]  = { "FACE_FEMALE", "FACE_MALE", "BUTTOCKS_EXPOSED",
                               "FEMALE_BREAST_EXPOSED", "ANUS_EXPOSED" };

static const char **g_classes = BOORU;
static int g_n_classes = 5;
static int g_from_model = 1;

int bc_class_count(void) { return g_n_classes; }
int bc_class_from_model(void) { return g_from_model; }

const char *bc_class_name(int i)
{
	return (i >= 0 && i < g_n_classes) ? g_classes[i] : "?";
}

int bc_class_index(const char *name)
{
	for (int i = 0; i < g_n_classes; i++)
		if (!strcasecmp(name, g_classes[i]))
			return i;
	return -1;
}

#include "../src/config.c"

/* --- harness -------------------------------------------------------------- */

static int failures;

static void check(int ok, const char *what)
{
	printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
	if (!ok)
		failures++;
}

static const char *PATH = "build/config_test.conf";

/* Is `needle` a line of the file, inside section `section`? */
static int in_section(const char *section, const char *needle)
{
	FILE *f = fopen(PATH, "r");
	if (!f)
		return 0;
	char line[512], cur[128] = "";
	int found = 0;
	while (fgets(line, sizeof line, f)) {
		if (line[0] == '[') {
			snprintf(cur, sizeof cur, "%s", line + 1);
			char *e = strchr(cur, ']');
			if (e)
				*e = 0;
			continue;
		}
		if (strcmp(cur, section))
			continue;
		if (!strncasecmp(line, needle, strlen(needle))) {
			found = 1;
			break;
		}
	}
	fclose(f);
	return found;
}

static void write_file(const char *body)
{
	FILE *f = fopen(PATH, "w");
	fputs(body, f);
	fclose(f);
}

/*
 * The regression. A file written before some global key existed must still be
 * rewritten -- and the rewrite is not about that key, it is the same pass that
 * puts the running model's classes under [classes]. Switching it off for a
 * missing key froze the class list at whichever taxonomy last wrote it, for
 * every existing config file, permanently.
 */
static void test_a_missing_global_key_still_rewrites(void)
{
	g_classes = NUDE;
	write_file("[global]\n"
	           "enabled = true\n"
	           "\n[classes]\n"
	           "HEAD  = false\n"
	           "BUST  = false\n");

	struct bc_config c;
	bc_config_load(&c, PATH);

	check(in_section("global", "join"),
	      "a key the file never had is written out");
	check(in_section("classes", "FACE_FEMALE"),
	      "  and the running model's classes reach [classes]");
	check(in_section("classes.inactive", "HEAD"),
	      "  and the other taxonomy's are filed as inactive, not lost");
}

/* The same again from the other direction: a complete, current file must be
 * left alone, or every load would rewrite and the GUI would reload forever. */
static void test_a_current_file_is_left_alone(void)
{
	g_classes = BOORU;
	struct bc_config c;
	bc_config_load(&c, PATH);   /* whatever state: normalises the file */
	bc_config_save(&c);

	FILE *f = fopen(PATH, "r");
	char before[4096] = "";
	size_t n = fread(before, 1, sizeof before - 1, f);
	before[n] = 0;
	fclose(f);

	struct bc_config d;
	bc_config_load(&d, PATH);

	f = fopen(PATH, "r");
	char after[4096] = "";
	n = fread(after, 1, sizeof after - 1, f);
	after[n] = 0;
	fclose(f);

	check(!strcmp(before, after), "a file that is already current is not rewritten");
}

/*
 * A model swap is exactly this: load the same file again with a different
 * taxonomy in place. The classes must change over, and the toggles of the
 * model being left must survive so switching back restores them.
 */
static void test_a_swap_moves_the_class_list_over(void)
{
	g_classes = BOORU;
	write_file("[global]\n"
	           "enabled = true\n"
	           "\n[classes]\n"
	           "BOOB = true\n"
	           "HEAD = false\n");
	struct bc_config c;
	bc_config_load(&c, PATH);
	check(in_section("classes", "BOOB"), "before the swap, [classes] is booru's");

	g_classes = NUDE;             /* the swap */
	bc_config_load(&c, PATH);

	check(in_section("classes", "ANUS_EXPOSED"),
	      "after it, [classes] is the new model's");
	check(!in_section("classes", "BOOB"),
	      "  and the old model's are no longer offered");
	check(in_section("classes.inactive", "BOOB"),
	      "  but are kept for switching back");

	g_classes = BOORU;            /* and back again */
	bc_config_load(&c, PATH);
	check(in_section("classes", "BOOB") && c.censor[2],
	      "switching back restores the toggle, not the default");
}

/*
 * Text colour. The interesting one is not the hex parsing but its collision
 * with comment stripping: a colour is spelled #RRGGBB, and a stripper that
 * cuts at any '#' turns the value into the empty string. That resolves to
 * "auto", so the setting is ignored while the file still plainly shows the
 * colour that was asked for -- nothing to see and nothing logged.
 */
static void test_text_color(void)
{
	int mode; float rgb[3];

	check(parse_text_color("auto", &mode, rgb) == 0 && mode == 0,
	      "auto leaves each effect its own colour");
	check(parse_text_color("rainbow", &mode, rgb) == 0 && mode == 2,
	      "rainbow selects the moving hue");
	check(parse_text_color("#40C0FF", &mode, rgb) == 0 && mode == 1 &&
	      rgb[0] > 0.24f && rgb[0] < 0.26f && rgb[2] == 1.0f,
	      "a hex triplet is read as a solid colour");
	check(parse_text_color("cyan", &mode, rgb) == 0 && mode == 1,
	      "so is a colour name");

	/* Unreadable must fall back to the effect's own colour, never to
	 * black: black glyphs on the near-black bar read as text that has
	 * disappeared, which sends you looking for a bug in the effect. */
	rgb[0] = rgb[1] = rgb[2] = 9.0f;
	check(parse_text_color("#abc", &mode, rgb) != 0 && mode == 0,
	      "a value that is not a colour falls back to auto, not black");

	/* And end to end, through the file, with a trailing comment. */
	g_classes = BOORU;
	write_file("[global]\n"
	           "enabled    = true\n"
	           "text_color = #40c0ff   # glyph colour\n");
	struct bc_config c;
	bc_config_load(&c, PATH);
	check(c.text_mode == 1 && c.text_rgb[2] == 1.0f,
	      "a colour survives the trailing-comment stripper");
	check(!strcmp(c.text_color, "#40c0ff"),
	      "  and is written back as the user spelled it");

	/*
	 * The other half of the same rule. An empty optional path is written as
	 * `key =    # comment`, and after trim() that comment sits at the head
	 * of the value where the "must be preceded by whitespace" test does not
	 * reach it -- so the setting read back as its own documentation, was
	 * written out again with the comment now doubled, and an unset model
	 * became a path made of prose.
	 */
	write_file("[global]\n"
	           "enabled    = true\n"
	           "seg_model  =    # optional: clips covers to the outline\n"
	           "model      = vendor/real.onnx   # detector to load\n");
	bc_config_load(&c, PATH);
	check(c.seg_model[0] == 0,
	      "a value that is only a comment reads as empty, not as the comment");
	check(!strcmp(c.model, "vendor/real.onnx"),
	      "  while a real value beside a comment is untouched");
}

/*
 * Coarseness is edited live: the GUI writes `strength_<effect>` and the running
 * censor picks it up through bc_config_poll(). `c.strength` -- the one value the
 * renderer actually reads -- is derived from the per-effect table, and that
 * derivation used to happen in bc_config_load() only. So the first load was
 * right and every reload after it fell back to the legacy `strength =` key in
 * [global], which the GUI never writes. Symptom: the coarseness box moves, the
 * file changes, and nothing on screen does.
 */
static void test_coarseness_survives_a_reload(void)
{
	g_classes = BOORU;
	write_file("[global]\n"
	           "effect     = blur\n"
	           "strength   = 11.0\n"
	           "[strength]\n"
	           "strength_blur     = 42.0\n"
	           "strength_pixelate = 5.0\n");

	struct bc_config c;
	bc_config_load(&c, PATH);
	check(c.strength > 41.9f && c.strength < 42.1f,
	      "the effect's own coarseness wins over the legacy key");

	/* What the GUI does: rewrite with a new per-effect value, leaving the
	 * legacy key alone. mtime is forced because the test writes twice
	 * inside one second. */
	write_file("[global]\n"
	           "effect     = blur\n"
	           "strength   = 11.0\n"
	           "[strength]\n"
	           "strength_blur     = 77.0\n"
	           "strength_pixelate = 5.0\n");
	c.mtime = 0;
	check(bc_config_poll(&c) == 1, "a changed file reloads");
	check(c.strength > 76.9f && c.strength < 77.1f,
	      "  and the new coarseness reaches the renderer");

	/* Switching effect retargets it, without the file's numbers moving. */
	write_file("[global]\n"
	           "effect     = pixelate\n"
	           "strength   = 11.0\n"
	           "[strength]\n"
	           "strength_blur     = 77.0\n"
	           "strength_pixelate = 5.0\n");
	c.mtime = 0;
	bc_config_poll(&c);
	check(c.strength > 4.9f && c.strength < 5.1f,
	      "  and a change of effect switches to that effect's value");
}

/*
 * Drawing mode is a request made of a running core, not a setting.
 *
 * The core clears the flag when the drawing ends, so a file still holding
 * `true` means the last run died mid-draw. Honouring it at startup would grab
 * the pointer and keyboard for the whole output before there is anything on
 * screen to explain why -- and the way out of that mode is a key the user has
 * no reason to press yet. So a load forces it off, and only the live reload can
 * turn it on.
 */
/*
 * Outlines round-trip, and the bounding box is derived rather than stored.
 *
 * Everything downstream covers the bounding box and clips it to the points, so
 * a box that disagreed with its own outline would cover the wrong rectangle
 * and clip it to a shape somewhere else -- silently, because both halves would
 * look reasonable on their own.
 */
static void test_outlines_round_trip(void)
{
	g_classes = BOORU;
	write_file("[global]\n"
	           "enabled    = true\n"
	           "outline    = 0.0\n"
	           "[regions]\n"
	           "p1 = poly:blur 0.2000,0.2000 0.5000,0.2500 0.4500,0.6000\n"
	           "r1 = 0.7000,0.1000,0.2000,0.2000\n"
	           "p2 = poly 0.1,0.1 0.2,0.2\n");

	struct bc_config c;
	bc_config_load(&c, PATH);
	check(c.n_regions == 2,
	      "an outline and a rectangle load, a two-point outline does not");
	check(c.regions[0].n_pts == 3, "  the outline keeps its points");
	check(c.regions[0].effect == BC_CFG_BLUR, "  and its effect");
	check(c.regions[1].n_pts == 0, "  a rectangle has no points");

	/* Derived, not read: x,y,w,h never appear on a poly line. */
	const struct bc_region *p = &c.regions[0];
	check(p->x > 0.199f && p->x < 0.201f &&
	      p->y > 0.199f && p->y < 0.201f &&
	      p->w > 0.299f && p->w < 0.301f &&
	      p->h > 0.399f && p->h < 0.401f,
	      "  and the bounding box is computed from the points");

	/* The GUI rewrites the whole file on every edit, so a shape that did
	 * not survive a save would be lost by touching an unrelated setting. */
	bc_config_save(&c);
	struct bc_config d;
	bc_config_load(&d, PATH);
	check(d.n_regions == 2 && d.regions[0].n_pts == 3 &&
	      d.regions[0].effect == BC_CFG_BLUR,
	      "  and all of it survives a save");
	check(d.regions[0].px[2] > 0.4499f && d.regions[0].px[2] < 0.4501f &&
	      d.regions[0].py[2] > 0.5999f && d.regions[0].py[2] < 0.6001f,
	      "  with the points where they were put");
}

/*
 * The eye bar's effect and whether the face is covered under it used to be one
 * number: naming an effect made the bar additive *and* kept the face cover, and
 * leaving it off meant the band replaced the face. So "a bar with its own
 * effect over an otherwise untouched face" was unsayable.
 *
 * Splitting them has to not change what existing files do, and the only way to
 * manage that is to derive the new key from the old one when the file predates
 * it. A fixed default would silently start covering faces that were not covered
 * before -- or stop covering ones that were, which is the direction that
 * matters.
 */
static void test_eye_bar_face_is_derived_for_old_files(void)
{
	g_classes = BOORU;
	struct bc_config c;

	/* No eye_bar_face key, an effect named: the face used to be covered. */
	write_file("[global]\n"
	           "enabled    = true\n"
	           "eye_bar    = true\n"
	           "eye_bar_effect = fill\n");
	bc_config_load(&c, PATH);
	check(c.eye_bar_face == 1,
	      "an old file naming a bar effect still covers the face");

	/* No key, no effect: the band replaced the face, so the face was not
	 * separately covered. */
	write_file("[global]\n"
	           "enabled    = true\n"
	           "eye_bar    = true\n"
	           "eye_bar_effect = off\n");
	bc_config_load(&c, PATH);
	check(c.eye_bar_face == 0,
	      "  and an old file with no bar effect still does not");

	/* The combination that could not be asked for before. */
	write_file("[global]\n"
	           "enabled    = true\n"
	           "eye_bar    = true\n"
	           "eye_bar_effect = fill\n"
	           "eye_bar_face = false\n");
	bc_config_load(&c, PATH);
	check(c.eye_bar_effect == BC_CFG_FILL && c.eye_bar_face == 0,
	      "  a bar with its own effect over an uncovered face is sayable");

	/* An explicit value must beat the derivation, in both directions. */
	write_file("[global]\n"
	           "enabled    = true\n"
	           "eye_bar    = true\n"
	           "eye_bar_effect = off\n"
	           "eye_bar_face = true\n");
	bc_config_load(&c, PATH);
	check(c.eye_bar_face == 1,
	      "  and what the file says wins over what would be derived");

	/* It has to survive the live reload too -- that path resets to defaults
	 * and re-parses, so a derivation done only in load() would be lost. */
	write_file("[global]\n"
	           "enabled    = true\n"
	           "eye_bar    = true\n"
	           "eye_bar_effect = blur\n");
	c.mtime = 0;
	check(bc_config_poll(&c) == 1, "  the file reloads");
	check(c.eye_bar_face == 1, "  and a reload derives it the same way");
}

static void test_drawing_mode_is_never_inherited(void)
{
	g_classes = BOORU;
	write_file("[global]\n"
	           "enabled    = true\n"
	           "draw       = true\n");

	struct bc_config c;
	bc_config_load(&c, PATH);
	check(c.draw == 0, "a config that says draw = true does not start drawing");

	/* But a live change to a running core does, which is the whole
	 * mechanism the GUI button works through. */
	write_file("[global]\n"
	           "enabled    = true\n"
	           "draw       = true\n");
	c.mtime = 0;
	check(bc_config_poll(&c) == 1, "  the file reloads");
	check(c.draw == 1, "  and a reload is what turns drawing on");

	write_file("[global]\n"
	           "enabled    = true\n"
	           "draw       = false\n");
	c.mtime = 0;
	bc_config_poll(&c);
	check(c.draw == 0, "  and clearing it in the file turns it off again");
}

/*
 * Per-class effects. A class line now carries either a boolean, as it always
 * has, or the name of the effect that covers that class -- which also means
 * "on", so there is no way to name an effect and leave the class off.
 *
 * The pin has to survive a save, because the GUI's every edit rewrites the
 * whole file: a save that spelled a pinned class "true" would discard the pin
 * the first time anything else was touched.
 */
static void test_per_class_effects(void)
{
	g_classes = NUDE;
	write_file("[global]\n"
	           "effect = pixelate\n"
	           "[strength]\n"
	           "strength_pixelate = 5.0\n"
	           "strength_fill     = 42.0\n"
	           "\n[classes]\n"
	           "FACE_FEMALE           = fill\n"
	           "FACE_MALE             = true\n"
	           "BUTTOCKS_EXPOSED      = false\n");

	struct bc_config c;
	bc_config_load(&c, PATH);

	int face_f = bc_class_index("FACE_FEMALE");
	int face_m = bc_class_index("FACE_MALE");
	int butt   = bc_class_index("BUTTOCKS_EXPOSED");

	check(c.censor[face_f] == 1, "naming an effect switches the class on");
	check(bc_cfg_class_effect(&c, face_f) == BC_CFG_FILL,
	      "  and pins it to that effect");
	check(bc_cfg_class_effect(&c, face_m) == BC_CFG_PIXELATE,
	      "a plain 'true' still follows the global effect");
	check(c.censor[butt] == 0, "'false' is still off");

	/*
	 * The strength a pinned class draws with is its own effect's, not the
	 * global effect's -- a face pinned to fill must not be sized by the
	 * pixelate slider.
	 */
	check(c.strength_by[bc_cfg_class_effect(&c, face_f)] > 41.9f &&
	      c.strength_by[bc_cfg_class_effect(&c, face_f)] < 42.1f,
	      "  and draws at that effect's own strength");

	/* Survives the rewrite. */
	bc_config_save(&c);
	check(in_section("classes", "FACE_FEMALE                = fill"),
	      "a pinned class is saved as its effect, not as 'true'");
	check(in_section("classes", "FACE_MALE                  = true"),
	      "  while an unpinned one keeps the boolean spelling");

	struct bc_config c2;
	c2.mtime = 0;
	bc_config_load(&c2, PATH);
	check(bc_cfg_class_effect(&c2, face_f) == BC_CFG_FILL,
	      "  and reloads to the same effect");

	/*
	 * Changing the global effect moves every class that was never pinned,
	 * and no class that was. This is why the sentinel is -1 rather than a
	 * copy of the global effect at write time.
	 */
	c2.effect = BC_CFG_BLUR;
	check(bc_cfg_class_effect(&c2, face_m) == BC_CFG_BLUR,
	      "changing the global effect moves the unpinned classes");
	check(bc_cfg_class_effect(&c2, face_f) == BC_CFG_FILL,
	      "  and leaves the pinned ones where they were");

	/* An unreadable value is not an effect and not true: off, as before. */
	write_file("[global]\n"
	           "effect = pixelate\n"
	           "\n[classes]\n"
	           "FACE_FEMALE = wobble\n");
	struct bc_config c3;
	c3.mtime = 0;
	bc_config_load(&c3, PATH);
	check(c3.censor[face_f] == 0,
	      "an effect name nobody recognises leaves the class off");
}

/*
 * Per-class detection floors, written as an `@score` suffix on the class line.
 *
 * The suffix rides on the existing line rather than living in a section of its
 * own so that it survives the retained block -- a class the current model does
 * not have is kept verbatim, and a threshold in a separate section would be
 * orphaned from the line it belongs to on the first model swap.
 */
static void test_per_class_confidence(void)
{
	g_classes = NUDE;
	write_file("[global]\n"
	           "effect     = pixelate\n"
	           "confidence = 0.40\n"
	           "\n[classes]\n"
	           "FACE_FEMALE           = fill@0.15\n"
	           "FACE_MALE             = true@0.75\n"
	           "BUTTOCKS_EXPOSED      = true\n"
	           "ANUS_EXPOSED          = true@wobble\n");

	struct bc_config c;
	bc_config_load(&c, PATH);

	const int face_f = bc_class_index("FACE_FEMALE");
	const int face_m = bc_class_index("FACE_MALE");
	const int butt   = bc_class_index("BUTTOCKS_EXPOSED");
	const int anus   = bc_class_index("ANUS_EXPOSED");

	check(bc_cfg_class_conf(&c, face_f) > 0.149f &&
	      bc_cfg_class_conf(&c, face_f) < 0.151f,
	      "a class may be pinned below the global floor");
	check(bc_cfg_class_conf(&c, face_m) > 0.749f &&
	      bc_cfg_class_conf(&c, face_m) < 0.751f,
	      "  and above it");
	check(bc_cfg_class_conf(&c, butt) > 0.399f &&
	      bc_cfg_class_conf(&c, butt) < 0.401f,
	      "a class without one follows the global confidence");

	/* The suffix must not disturb the value in front of it. */
	check(c.censor[face_f] == 1 &&
	      bc_cfg_class_effect(&c, face_f) == BC_CFG_FILL,
	      "the effect in front of the suffix still parses");
	check(c.censor[face_m] == 1, "  and so does a plain boolean");

	/*
	 * An unparseable threshold falls back to the global floor, never to
	 * zero: zero admits every channel at any score, which is the whole
	 * screen covered rather than a censor.
	 */
	check(bc_cfg_class_conf(&c, anus) > 0.399f &&
	      bc_cfg_class_conf(&c, anus) < 0.401f,
	      "an unreadable threshold falls back to the global, not to zero");
	check(c.censor[anus] == 1,
	      "  and leaves the class switched on as it was written");

	/* Survives the rewrite, and only where one was actually set. */
	bc_config_save(&c);
	check(in_section("classes", "FACE_FEMALE                = fill@0.15"),
	      "a pinned threshold is written back onto the class line");
	check(in_section("classes", "BUTTOCKS_EXPOSED           = true\n"),
	      "  and a class without one keeps the bare spelling");

	struct bc_config c2;
	c2.mtime = 0;
	bc_config_load(&c2, PATH);
	check(bc_cfg_class_conf(&c2, face_f) > 0.149f &&
	      bc_cfg_class_conf(&c2, face_f) < 0.151f,
	      "  and reloads to the same threshold");

	/*
	 * Same sentinel argument as the effect pin: moving the global floor
	 * must move every class that was never given one of its own, and no
	 * class that was.
	 */
	c2.confidence = 0.60f;
	check(bc_cfg_class_conf(&c2, butt) > 0.599f &&
	      bc_cfg_class_conf(&c2, butt) < 0.601f,
	      "changing the global floor moves the unpinned classes");
	check(bc_cfg_class_conf(&c2, face_f) > 0.149f &&
	      bc_cfg_class_conf(&c2, face_f) < 0.151f,
	      "  and leaves the pinned ones where they were");

	/* An old file, with no suffix anywhere, must behave exactly as before. */
	write_file("[global]\n"
	           "confidence = 0.30\n"
	           "\n[classes]\n"
	           "FACE_FEMALE = true\n");
	struct bc_config c3;
	c3.mtime = 0;
	bc_config_load(&c3, PATH);
	check(bc_cfg_class_conf(&c3, face_f) > 0.299f &&
	      bc_cfg_class_conf(&c3, face_f) < 0.301f,
	      "a file written before the suffix existed is unchanged by it");
}

int main(void)
{
	printf("Config file:\n");
	test_text_color();
	test_coarseness_survives_a_reload();
	test_a_missing_global_key_still_rewrites();
	test_a_current_file_is_left_alone();
	test_a_swap_moves_the_class_list_over();
	test_eye_bar_face_is_derived_for_old_files();
	test_outlines_round_trip();
	test_drawing_mode_is_never_inherited();
	test_per_class_effects();
	test_per_class_confidence();
	remove(PATH);

	printf(failures ? "\n%d check(s) FAILED\n" : "\nAll checks passed.\n",
	       failures);
	return failures ? 1 : 0;
}
