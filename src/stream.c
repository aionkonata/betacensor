/*
 * The per-frame pipeline shared by bc-video and the animated half of bc-image.
 * See stream.h.
 *
 * This was the body of bc-video's frame loop, moved out unchanged in
 * behaviour (bc-video's output is byte-identical before and after) so that an
 * animated image is censored by the same code as a video of the same frames.
 */

#include "stream.h"
#include "eyebar.h"
#include "join.h"
#include "util.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * A face's bar, steadied over time.
 *
 * Landmarks move a few pixels from frame to frame on a face that has not, and
 * a bar drawn straight from them would shimmer. So the bar is kept as
 * fractions of its face -- which makes it follow the face through the
 * tracker's own smoothing for free -- and each new measurement is blended
 * into the last. The angle is blended as a doubled angle because a bar is
 * the same after a half turn: averaging +89 and -89 degrees must give
 * 90, not 0.
 *
 * Where this frame has no believable landmarks the last good bar is reused
 * for a few frames, since a blink or a turned head makes the model lose a
 * face for a moment and not for good. After that it is the band, which
 * covers more than the eyes -- the fail-closed direction.
 */
#define VBAR_NEW   0.5f           /* weight of a fresh measurement */
#define VBAR_HOLD  8              /* frames a last-good bar is reused */

static struct bc_eyebar steady_bar(struct bc_stream_vbar *s, double born,
                                   const struct bc_off_face *f,
                                   const struct bc_eyebar *now, int measured)
{
	const float fcx = f->x + f->w * 0.5f, fcy = f->y + f->h * 0.5f;
	if (s->born != born)
		s->valid = 0;
	s->born = born;

	if (measured) {
		float dcx = (now->cx - fcx) / f->w, dcy = (now->cy - fcy) / f->h;
		float fw = now->w / f->w, fh = now->h / f->h, ang = now->angle;
		if (s->valid) {
			dcx = s->dcx + (dcx - s->dcx) * VBAR_NEW;
			dcy = s->dcy + (dcy - s->dcy) * VBAR_NEW;
			fw  = s->fw  + (fw  - s->fw)  * VBAR_NEW;
			fh  = s->fh  + (fh  - s->fh)  * VBAR_NEW;
			const float sa = (1.0f - VBAR_NEW) * sinf(2.0f * s->angle) +
			                 VBAR_NEW * sinf(2.0f * ang);
			const float ca = (1.0f - VBAR_NEW) * cosf(2.0f * s->angle) +
			                 VBAR_NEW * cosf(2.0f * ang);
			ang = bc_eyebar_wrap(0.5f * atan2f(sa, ca));
		}
		s->dcx = dcx; s->dcy = dcy; s->fw = fw; s->fh = fh; s->angle = ang;
		s->valid = 1;
		s->missed = 0;
	} else if (s->valid && ++s->missed > VBAR_HOLD) {
		s->valid = 0;
	} else if (!s->valid) {
		return *now;          /* the band, from the caller */
	}
	if (!s->valid)
		return *now;
	struct bc_eyebar out = {
		.cx = fcx + s->dcx * f->w, .cy = fcy + s->dcy * f->h,
		.w = s->fw * f->w, .h = s->fh * f->h, .angle = s->angle
	};
	return out;
}

void bc_stream_init(struct bc_stream *s)
{
	const struct bc_config *cfg = s->cfg;
	memset(&s->tr, 0, sizeof s->tr);
	bc_tracker_init(&s->tr, cfg->confidence, 0.30f, cfg->sticky_ms);
	s->tr.same_class = bc_off_classes_similar;
	s->tr.low_score  = cfg->confidence * 0.5f;
	memset(&s->eb, 0, sizeof s->eb);
	memset(s->vbar, 0, sizeof s->vbar);
	s->prev_ok = 0;
	s->frames = s->censored_frames = s->blanked_frames = 0;
	s->reused = 0;
	s->covered_px = 0.0;
}

void bc_stream_free(struct bc_stream *s)
{
	bc_off_eb_free(&s->eb);
	free(s->prev);
	s->prev = NULL;
	s->prev_cap = s->prev_len = 0;
}

int bc_stream_frame(struct bc_stream *s, const unsigned char *rgb, double t_ms,
                    unsigned char *out, struct bc_box *boxes_out, int *nb_out)
{
	const struct bc_config *cfg = s->cfg;
	struct bc_off_gl *g = s->g;
	const size_t fsz = (size_t)g->w * g->h * 3;
	if (nb_out)
		*nb_out = 0;

	bc_off_upload(g, rgb);

	/* A frame the detector has already answered for needs no second answer. */
	struct bc_det dets[BC_MAX_DET];
	struct bc_det_result res = { BC_DET_FAILED, 0, 0.0 };
	if (s->prev_ok && s->prev_len == fsz && !memcmp(s->prev, rgb, fsz)) {
		memcpy(dets, s->prev_dets, sizeof dets);
		res.status = BC_DET_OK;
		res.n = s->prev_n;
		s->reused++;
	} else {
		int hung = 0;
		res = bc_off_detect(g, s->pp, s->det, s->tensor, dets, BC_MAX_DET, &hung);
		/*
		 * No outcome at all: the detector is hung. Carrying on would hand
		 * its eventual late result to the next frame, so the caller must
		 * stop here, with everything written so far still censored.
		 */
		if (hung)
			return BC_STREAM_HUNG;
		s->prev_ok = 0;
		if (res.status == BC_DET_OK) {
			if (s->prev_cap < fsz) {
				free(s->prev);
				s->prev = malloc(fsz);
				s->prev_cap = s->prev ? fsz : 0;
			}
			if (s->prev) {
				memcpy(s->prev, rgb, fsz);
				s->prev_len = fsz;
				memcpy(s->prev_dets, dets, sizeof dets);
				s->prev_n = res.n;
				s->prev_ok = 1;
			}
		}
	}
	const int n_det = res.status == BC_DET_OK ? res.n : 0;

	/*
	 * §6, offline. No result for this frame -- preprocess failed, the
	 * detector refused it, or the run errored -- means nobody looked at it,
	 * and zero detections would write it out uncensored while the run still
	 * reported SUCCESS. Write it black instead, and say so at the end.
	 */
	if (res.status != BC_DET_OK) {
		memset(out, 0, fsz);
		s->blanked_frames++;
		s->censored_frames++;
		s->covered_px += (double)g->w * g->h;
		s->frames++;
		return BC_STREAM_BLANKED;
	}

	/* Network coordinates to frame pixels, with §2's dilation. */
	struct bc_det sd[BC_MAX_DET];
	int ns = 0;
	for (int i = 0; i < n_det && ns < BC_MAX_DET; i++) {
		if (dets[i].cls < 0 || dets[i].cls >= BC_MAX_CLASSES ||
		    !s->cover[dets[i].cls])
			continue;
		float x0, y0, x1, y1;
		bc_preproc_to_screen(s->pp, dets[i].x, dets[i].y, &x0, &y0);
		bc_preproc_to_screen(s->pp, dets[i].x + dets[i].w,
		                     dets[i].y + dets[i].h, &x1, &y1);
		float w = x1 - x0, h = y1 - y0;
		float dx = w * cfg->dilate / 200.0f;
		float dy = h * cfg->dilate / 200.0f;
		sd[ns] = dets[i];
		sd[ns].x = x0 - dx;
		sd[ns].y = y0 - dy;
		sd[ns].w = w + 2 * dx;
		sd[ns].h = h + 2 * dy;
		ns++;
	}

	/* With the eye bar on, an EYE detection inside a head is the bar's
	 * business and must not open a cover of its own. */
	ns = bc_off_drop_eyes(cfg, sd, ns, &s->eb.eyes_dropped);

	bc_tracker_update(&s->tr, sd, ns, t_ms, s->cover, NULL);

	/*
	 * The eye bars for this frame, one landmark run for every head being
	 * covered. Done before the covers so a face given a bar can skip its own
	 * cover when the config asks.
	 */
	struct bc_eyebar bar_of[BC_TRK_MAX];
	struct bc_off_face face_of[BC_TRK_MAX];
	int has_bar[BC_TRK_MAX] = { 0 };
	if (cfg->eye_bar) {
		struct bc_off_face hf[BC_TRK_MAX];
		struct bc_eyebar   hb[BC_TRK_MAX];
		int                hk[BC_TRK_MAX], hi[BC_TRK_MAX], nh = 0;
		for (int k = 0; k < BC_TRK_MAX; k++) {
			float x, y, w, h;
			if (!bc_trk_box(&s->tr.t[k], &x, &y, &w, &h) ||
			    s->tr.t[k].cls < 0 || s->tr.t[k].cls >= BC_MAX_CLASSES ||
			    !s->cover[s->tr.t[k].cls] ||
			    !bc_is_head_class(bc_class_name(s->tr.t[k].cls)))
				continue;
			hf[nh] = (struct bc_off_face){ x, y, w, h };
			hi[nh++] = k;
		}
		bc_off_eb_bars(&s->eb, cfg, g, hf, nh, hb, hk);
		for (int m = 0; m < nh; m++) {
			const int k = hi[m];
			bar_of[k] = steady_bar(&s->vbar[k], s->tr.t[k].born, &hf[m],
			                       &hb[m], hk[m] == BC_OFF_EB_EYES);
			face_of[k] = hf[m];
			has_bar[k] = 1;
		}
	}
	for (int k = 0; k < BC_TRK_MAX; k++)
		if (!has_bar[k])
			s->vbar[k].valid = 0;   /* the face is gone; so is its history */

	struct bc_box boxes[BC_STREAM_MAX_BOX];
	int nb = 0;
	for (int k = 0; k < BC_TRK_MAX && nb < BC_STREAM_MAX_BOX; k++) {
		float x, y, w, h;
		if (!bc_trk_box(&s->tr.t[k], &x, &y, &w, &h))
			continue;
		if (s->tr.t[k].cls >= 0 && s->tr.t[k].cls < BC_MAX_CLASSES &&
		    !s->cover[s->tr.t[k].cls])
			continue;
		/* The bar is the whole cover when the face is left alone. */
		if (has_bar[k] && !cfg->eye_bar_face)
			continue;
		/* Clamp to the frame; a scissor outside it draws nothing. */
		float x0 = x < 0.0f ? 0.0f : x;
		float y0 = y < 0.0f ? 0.0f : y;
		float x1 = x + w > g->w ? (float)g->w : x + w;
		float y1 = y + h > g->h ? (float)g->h : y + h;
		if (x1 - x0 < 1.0f || y1 - y0 < 1.0f)
			continue;
		/* Styled by the class that opened the track. */
		boxes[nb] = bc_off_box(cfg, s->tr.t[k].cls, (int)x0, (int)y0,
		                       (int)(x1 - x0), (int)(y1 - y0));
		s->covered_px += (double)boxes[nb].w * boxes[nb].h;
		nb++;
	}
	/*
	 * A box pinned regardless of what was detected, so the render path can be
	 * exercised on content that has nothing to find in it.
	 */
	if (s->test_box[2] > 0 && nb < BC_STREAM_MAX_BOX) {
		boxes[nb] = (struct bc_box){
			.x = s->test_box[0], .y = s->test_box[1],
			.w = s->test_box[2], .h = s->test_box[3],
			.effect = bc_off_effect(cfg->effect), .strength = cfg->strength,
			.text_mode = cfg->text_mode,
			.text_rgb = { cfg->text_rgb[0], cfg->text_rgb[1],
			              cfg->text_rgb[2] }
		};
		snprintf(boxes[nb].text, sizeof boxes[nb].text, "%s",
		         cfg->caption[0] ? cfg->caption : "CENSORED");
		s->covered_px += (double)boxes[nb].w * boxes[nb].h;
		nb++;
	}
	/* Same joining as the live pass, from the same config key, so a setting
	 * eyeballed on screen is what lands in the file. */
	if (cfg->join && nb > 1)
		nb = bc_join_boxes(boxes, nb, 6, cfg->join_waste);

	/* The bars last: on top of their faces, and never joined into them (an
	 * upright band would otherwise merge and lose its effect). Grown by the
	 * dilation, as in the live censor. */
	const float grow = 1.0f + cfg->dilate / 100.0f;
	for (int k = 0; k < BC_TRK_MAX && nb < BC_STREAM_MAX_BOX; k++) {
		struct bc_box bx;
		if (has_bar[k] &&
		    bc_off_eb_box(cfg, s->tr.t[k].cls, &face_of[k], &bar_of[k],
		                  grow, g->w, g->h, &bx)) {
			s->covered_px += (double)bx.w * bx.h;
			boxes[nb++] = bx;
		}
	}

	if (s->adjust)
		nb = s->adjust(s->adjust_user, boxes, nb, BC_STREAM_MAX_BOX);

	if (nb)
		s->censored_frames++;

	bc_off_draw(g, boxes, nb, t_ms / 1000.0);
	if (s->harden)
		bc_off_harden(g, boxes, nb, s->harden_seed);
	bc_off_read(g, out);
	if (s->bdg)
		bc_off_badge_apply(s->bdg, out);

	if (boxes_out && nb_out) {
		memcpy(boxes_out, boxes, (size_t)nb * sizeof *boxes);
		*nb_out = nb;
	}
	s->frames++;
	return BC_STREAM_OK;
}
