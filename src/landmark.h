#ifndef BC_LANDMARK_H
#define BC_LANDMARK_H

/*
 * Anime face landmarks: hysts/anime-face-detector-hrnetv2, exported to ONNX by
 * tools/export_anime_landmarks.py.
 *
 * Why this exists at all. The eye bar needs to know which way a head is
 * turned, and on drawn faces nothing else here can say. Five sources were
 * measured and every one failed: the body axis is geometrically wrong (it
 * assumes the torso sits straight below the head), the y11 EYE class never
 * fires, the dedicated eye model's boxes jitter until smoothing ratchets them
 * over the whole face, WIDERFACE landmarks are trained on photographs and
 * regress somewhere else on a drawing, and the face box aspect measured
 * 1.04-1.14 whether the subject was upright or lying down. Without an angle the
 * bar has to cover the middle of the face at every rotation, which works but
 * covers far more than the eyes.
 *
 * This model is trained on drawn faces and gives 28 points, six of them per
 * eye -- so an eye centre is an average rather than a single keypoint taken on
 * trust, and the line between the two centres is the angle directly.
 *
 * Top-down: it runs on a crop of one face, not on the frame, so it costs one
 * inference per face. That is the price, and why the caller decides how many
 * faces are worth it.
 */

#define BC_LM_POINTS 28

/*
 * Which points are which. Confirmed against the package's FLIP_PAIRS, which
 * mirrors 11..16 onto 17..22 -- so these two runs are the left and right eye
 * and not, say, an eye and an eyebrow.
 */
#define BC_LM_LEYE_FIRST 11
#define BC_LM_LEYE_LAST  16
#define BC_LM_REYE_FIRST 17
#define BC_LM_REYE_LAST  22

struct bc_landmark;

/*
 * One face's result, in the coordinates of the crop the model was given --
 * the caller maps back to the screen, since only it knows the crop.
 */
struct bc_lm_face {
	float x[BC_LM_POINTS], y[BC_LM_POINTS];
	float score[BC_LM_POINTS];   /* heatmap peak, 0..1 */
	/* Eye centres, averaged over each eye's six points, and the mean peak
	 * across those twelve -- the number worth gating on. */
	float lx, ly, rx, ry;
	float eye_score;
	/*
	 * Half-extents of the twelve eye points about their midpoint, measured
	 * in the eyes' own frame: `along` runs down the line between the two
	 * eye centres, `across` perpendicular to it.
	 *
	 * A bar sized from eye *separation* alone assumes eyes of a particular
	 * size relative to their spacing, which is a photographic-face
	 * assumption -- drawn eyes are far larger for the same separation, and
	 * a foreshortened face brings the centres together without shrinking
	 * the eyes at all. Both make a separation-scaled bar too small. These
	 * are measured from the points themselves, so neither can happen.
	 */
	float ext_along, ext_across;
};

/*
 * Load the model. `threads` is the intra-op count; CUDA is used when
 * available and CPU otherwise, exactly as the detector does.
 *
 * Returns NULL if the model will not load or is not this model -- the output
 * shape is checked, because a wrong model here would decode noise into
 * confident-looking eye positions.
 */
struct bc_landmark *bc_landmark_create(const char *model_path, int threads);
void bc_landmark_destroy(struct bc_landmark *l);

/* The input crop size the model wants, in pixels (square). */
int bc_landmark_input(const struct bc_landmark *l);

/*
 * Run `n` faces, whose preprocessed crops sit consecutively in `tensor`, each
 * 3*input*input floats, NCHW, ImageNet-normalised. Writes `n` results.
 *
 * Synchronous: the caller already has the crops and wants the angles for this
 * frame's boxes. Returns the number decoded, or -1.
 */
int bc_landmark_run(struct bc_landmark *l, const float *tensor, int n,
                    struct bc_lm_face *out, int max);

/* Milliseconds the last bc_landmark_run() took, for all faces together. */
double bc_landmark_last_ms(const struct bc_landmark *l);

/* Is it running on the GPU? */
int bc_landmark_on_gpu(const struct bc_landmark *l);

#endif /* BC_LANDMARK_H */
