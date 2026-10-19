#ifndef BC_DETECT_H
#define BC_DETECT_H

#include <stddef.h>

/*
 * Upper bound on classes, not the count. The taxonomy comes from whichever
 * model is loaded (NudeNet v3 has 18, Hotscreen's YOLOv11 models have 16 in a
 * different order), so anything that walks classes must use bc_class_count().
 */
#define BC_MAX_CLASSES 32
#define BC_MAX_DET     64

/*
 * NudeNet v3 detector (DESIGN.md §3[5]).
 *
 * Runs on a worker thread. The display loop must not block on inference: at
 * 180 Hz a frame is 5.5 ms and inference is far slower than that, so detection
 * rate and display rate are decoupled and boxes persist between detections.
 * That is the §2 design, not a workaround.
 */

/*
 * Class names, indexed to match the model, read from the loaded model's own
 * ONNX metadata. Before a model is loaded these describe NudeNet v3, which is
 * the default model; bc_detect_create() replaces them if the model says
 * otherwise.
 *
 * Getting this wrong is not a cosmetic bug: the orders genuinely differ between
 * models -- index 1 is FACE_FEMALE in NudeNet and MALE_FACE in the YOLOv11
 * models -- so a stale table censors faces and leaves genitals uncovered.
 */
const char *bc_class_name(int cls);

/* How many classes the loaded model actually has. */
int bc_class_count(void);

/* Non-zero once the table came from a model's metadata rather than the
 * built-in default. See bc_class_from_model() in detect.c. */
int bc_class_from_model(void);

/* Reverse lookup, case-insensitive. Returns -1 if the name is not a class. */
int bc_class_index(const char *name);

/* Upper bound on keypoints per detection. A 5-point face model uses 5. */
#define BC_MAX_KPT 8

/* A detection in *network* coordinates; the caller maps back to screen. */
struct bc_det {
	float x, y, w, h;   /* top-left, size */
	float score;
	int   cls;
	/*
	 * Which tile of the preprocessor's grid this came from. Its coordinates
	 * are in *that tile's* network space, so mapping back to the screen
	 * needs this; 0 when the grid is 1x1, which is every non-sectioned run.
	 */
	int   tile;
	/*
	 * Keypoints, for models with a Pose head. Predicted jointly with the
	 * box, which is what makes them steadier than a second model's output:
	 * they cannot disagree with the box they came from. n_kpt is 0 for a
	 * plain detection model, which is every model here except the 5-point
	 * face ones.
	 */
	int   n_kpt;
	float kx[BC_MAX_KPT], ky[BC_MAX_KPT];   /* network coordinates */
	float ks[BC_MAX_KPT];                   /* per-point confidence */
};

struct bc_detector;

struct bc_detector *bc_detect_create(const char *model_path,
                                     int net_w, int net_h, int threads);

/*
 * As above, but `adopt_taxonomy = 0` leaves the global class table alone.
 * Needed for a second, single-purpose model running alongside the main one --
 * its taxonomy would otherwise overwrite the primary model's, and the config,
 * the GUI and every class index in flight are keyed to that table.
 */
struct bc_detector *bc_detect_create_ex(const char *model_path,
                                        int net_w, int net_h, int threads,
                                        int adopt_taxonomy);
void bc_detect_destroy(struct bc_detector *d);

/*
 * Hand a tensor to the worker. Non-blocking.
 * Returns 1 if accepted, 0 if the worker is still busy (frame is skipped).
 */
int bc_detect_submit(struct bc_detector *d, const float *tensor);

/*
 * The same, for a tensor that is already in VRAM (DESIGN.md §0).
 *
 * `dev` is a device pointer to `tiles` consecutive tensors, from
 * bc_preproc_device_ptr(). Nothing is copied, so the buffer behind it must stay
 * untouched and mapped until the run finishes -- see bc_detect_busy(), which is
 * how the caller knows, and the ownership note on bc_preproc_gpu_register().
 *
 * Returns 1 if accepted, 0 if the worker is busy or zero-copy is unavailable,
 * in which case bc_detect_submit() is still there and still correct.
 */
int bc_detect_submit_device(struct bc_detector *d, const void *dev);

/*
 * Can this detector read a tensor straight out of VRAM? False on a CPU
 * session, or if ORT would not describe CUDA memory for us.
 */
int bc_detect_zero_copy(const struct bc_detector *d);

/*
 * Block until nothing is in flight.
 *
 * For the caller that is about to invalidate the memory a run is reading -- a
 * grid change reallocates the tensor buffer. Bounded by one inference.
 */
void bc_detect_wait_idle(struct bc_detector *d);

/*
 * Run `n` tensors per submission instead of one -- the tiles of a sectioned
 * frame, laid out consecutively. Each result is tagged with the tile it came
 * from. Resizes the internal buffers, so it must not be called while a run is
 * in flight; bc_detect_submit() then expects n * floats.
 *
 * A submission is all-or-nothing: if any tile's inference fails the whole
 * submission is reported as an error and nothing is published. Publishing the
 * tiles that did work would leave the failed tile's part of the screen looking
 * clean rather than unexamined, which is the one direction §6 forbids.
 */
int bc_detect_set_tiles(struct bc_detector *d, int n);

/*
 * Would a submission be accepted right now?
 *
 * Lets the caller skip building a tensor it could not hand over. That is not a
 * micro-optimisation: preprocessing ends in a readback that stalls the GL
 * pipeline, and with a tile grid it is one readback per tile, so producing a
 * tensor for a frame that will be dropped is the most expensive thing the loop
 * can do for no result.
 */
int bc_detect_busy(const struct bc_detector *d);

/*
 * What became of the latest submission. Every accepted submission ends in
 * exactly one OK or FAILED, handed out once by bc_detect_collect().
 *
 * FAILED is its own answer rather than a negative count because the two
 * readings of "-1" -- "not finished yet" and "finished with nothing usable" --
 * want opposite responses: the first is to wait, the second is §6's to act on.
 * When they shared a value, every caller had to rebuild the difference from
 * side counters, and two of them got it wrong (a failure read as a clean
 * screen in video mode, and as an unchanged one by the live change gate).
 */
enum bc_det_status {
	BC_DET_PENDING = 0,   /* nothing has finished since the last collect */
	BC_DET_OK,            /* a run finished: `n` detections, 0 = screen clean */
	BC_DET_FAILED,        /* a run finished with no usable output -- NOT clean */
};

struct bc_det_result {
	enum bc_det_status status;
	int    n;     /* detections written to `out`; BC_DET_OK only */
	double ms;    /* wall time of the run, all tiles; OK and FAILED */
};

/*
 * Collect the outcome of the latest finished run, if there is one the caller
 * has not seen. A newer outcome replaces an uncollected older one, success or
 * failure alike: only the latest describes the screen.
 */
struct bc_det_result bc_detect_collect(struct bc_detector *d,
                                       struct bc_det *out, int max);

/*
 * Change the confidence floor while running (the config file is editable at
 * runtime, so this cannot be a create-time constant). Takes effect on the next
 * inference; the one in flight finishes with the old value.
 */
void bc_detect_set_conf(struct bc_detector *d, float conf);

/*
 * Statistics only: runs that failed, and runs that succeeded. To
 * react to a run, use bc_detect_collect() -- a counter can say that something
 * failed since it was last read, but not whether that was the latest run.
 */
unsigned long bc_detect_errors(const struct bc_detector *d);
unsigned long bc_detect_count(const struct bc_detector *d);

/*
 * The input geometry the loaded model actually uses. May differ from what was
 * requested: models with a fixed input size dictate it, and feeding the wrong
 * size makes every inference fail rather than degrade.
 */
int bc_detect_net_w(const struct bc_detector *d);
int bc_detect_net_h(const struct bc_detector *d);

/* Non-zero if inference is running on the GPU rather than the CPU. */
int bc_detect_on_gpu(const struct bc_detector *d);

/* Keypoints per detection the loaded model predicts; 0 for a plain detector. */
int bc_detect_n_kpt(const struct bc_detector *d);

/*
 * Segmentation, for a model that has that head (a second output holding mask
 * prototypes). Everything below returns 0 / -1 on a plain detector.
 *
 * What is published is one silhouette for the whole frame, not a mask per
 * detection: every instance's mask, cropped to its own box and OR-ed together,
 * one byte per pixel and only ever 0 or 255. A censor asks whether a pixel is
 * on a subject, not which subject -- and a per-detection mask would have to
 * ride a box through easing, prediction and fusion, carrying geometry from an
 * inference that box has already moved away from.
 *
 * Coordinates are the model's *letterboxed network* space at prototype
 * resolution, so the caller maps them back the same way it maps boxes.
 */
int bc_detect_has_mask(const struct bc_detector *d);
int bc_detect_mask_w(const struct bc_detector *d);
int bc_detect_mask_h(const struct bc_detector *d);

/*
 * Copy the latest silhouette out, if one has arrived since the last call.
 * Returns the bytes written, or -1 if nothing is ready or `max` is too small.
 *
 * A run that could not read its prototypes publishes nothing rather than an
 * empty mask: empty reads as "no subject anywhere", which would uncover
 * everything the mask gates.
 */
int bc_detect_take_mask(struct bc_detector *d, unsigned char *out, size_t max);

#endif /* BC_DETECT_H */
