#include "eyebar.h"

#include <math.h>
#include <string.h>

int bc_is_head_class(const char *n)
{
	static const char *heads[] = {
		"HEAD", "HDRAGO", "HPONY", "HFOX", "HRABB",
		"HCAT", "HBEAR", "HHORSE", "HBIRD",
	};
	if (!n)
		return 0;
	if (strstr(n, "FACE"))
		return 1;
	for (unsigned i = 0; i < sizeof heads / sizeof heads[0]; i++)
		if (!strcmp(n, heads[i]))
			return 1;
	return 0;
}

float bc_eyebar_wrap(float a)
{
	while (a >   1.57079633f) a -= 3.14159265f;
	while (a <= -1.57079633f) a += 3.14159265f;
	return a;
}

int bc_eyebar_from_eyes(float ex0, float ey0, float ex1, float ey1,
                        float ext_along, float ext_across,
                        float pad, int nofloor,
                        float fx, float fy, float fw, float fh,
                        struct bc_eyebar *out)
{
	const float dx = ex1 - ex0, dy = ey1 - ey0;
	const float sep = sqrtf(dx * dx + dy * dy);
	const float span = fw > fh ? fw : fh;
	const float mx = fw * BC_EB_KPT_IN_FACE, my = fh * BC_EB_KPT_IN_FACE;
	const int sane = sep >= span * BC_EB_KPT_SEP_MIN &&
	                 sep <= span * BC_EB_KPT_SEP_MAX &&
	                 ex0 >= fx - mx && ex0 <= fx + fw + mx &&
	                 ex1 >= fx - mx && ex1 <= fx + fw + mx &&
	                 ey0 >= fy - my && ey0 <= fy + fh + my &&
	                 ey1 >= fy - my && ey1 <= fy + fh + my;
	if (!sane)
		return -1;
	if (!(sep > 1.0f))
		return 0;

	/* screen y grows down, GL's up */
	out->angle = bc_eyebar_wrap(atan2f(-dy, dx));

	/* Measured extents when we have them, padded so the bar covers the
	 * lids and lashes too, not just the points themselves. */
	float barw, barh;
	if (ext_along > 0.0f && ext_across > 0.0f) {
		barw = 2.0f * ext_along  * pad;
		barh = 2.0f * ext_across * pad;
	} else {
		barw = sep * BC_EB_KPT_BAR_WIDE;
		barh = sep * BC_EB_KPT_BAR_TALL;
	}
	if (!nofloor) {
		if (barw < sep * BC_EB_KPT_BAR_WIDE)
			barw = sep * BC_EB_KPT_BAR_WIDE;
		if (barh < sep * BC_EB_KPT_BAR_TALL)
			barh = sep * BC_EB_KPT_BAR_TALL;
	}
	out->cx = (ex0 + ex1) * 0.5f;
	out->cy = (ey0 + ey1) * 0.5f;
	out->w = barw;
	out->h = barh;
	return 1;
}

void bc_eyebar_band(float fx, float fy, float fw, float fh,
                    float top, float bot, struct bc_eyebar *out)
{
	const float x0 = fx + fw * BC_EB_INSET;
	const float y0 = fy + fh * top;
	out->w = fw * (1.0f - 2.0f * BC_EB_INSET);
	out->h = fh * (bot - top);
	out->cx = x0 + out->w * 0.5f;
	out->cy = y0 + out->h * 0.5f;
	out->angle = 0.0f;
}
