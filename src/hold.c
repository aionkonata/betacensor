#include "hold.h"

#include <math.h>
#include <stddef.h>

float bc_hold_region_mad(const float *cur, const float *prev,
                         int w, int h, int ch,
                         int x0, int y0, int x1, int y1)
{
	if (!cur || !prev || w <= 0 || h <= 0 || ch <= 0)
		return -1.0f;

	/* Clamp into the tensor. A box can hang off the screen edge -- dilate
	 * and growth both push outward -- and the part that is inside is still
	 * a fair sample of whether anything moved. */
	if (x0 < 0) x0 = 0;
	if (y0 < 0) y0 = 0;
	if (x1 > w) x1 = w;
	if (y1 > h) y1 = h;
	if (x1 <= x0 || y1 <= y0)
		return -1.0f;

	const size_t plane = (size_t)w * (size_t)h;
	double sum = 0.0;
	long   n   = 0;

	for (int c = 0; c < ch; c++) {
		const float *a = cur  + (size_t)c * plane;
		const float *b = prev + (size_t)c * plane;
		for (int y = y0; y < y1; y++) {
			const size_t row = (size_t)y * (size_t)w;
			for (int x = x0; x < x1; x++) {
				sum += fabs((double)a[row + x] - (double)b[row + x]);
				n++;
			}
		}
	}

	if (n == 0)
		return -1.0f;
	return (float)(sum / (double)n);
}
