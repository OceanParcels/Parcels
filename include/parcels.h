#include <stdio.h>

/* Grid dimension to be defined during the JIT process */
extern const int GRID_XDIM;
extern const int GRID_YDIM;

/* Local search to update grid index */
static inline int advance_index(float x, int i, int size, float *xvals)
{
    while (i < size-1 && x > xvals[i+1]) ++i;
    while (i > 0 && x < xvals[i]) --i;
    return i;
}


/* Bilinear interpolation routine for 2D grid */
static inline float interpolate_bilinear(float x, float y, int xi, int yi,
                                        float xvals[GRID_XDIM], float yvals[GRID_YDIM],
                                        float qvals[GRID_YDIM][GRID_XDIM])
{
  int i = xi, j = yi;
  i = advance_index(x, i, GRID_XDIM, xvals);
  j = advance_index(y, j, GRID_YDIM, yvals);
  return (qvals[i][j] * (xvals[i+1] - x) * (yvals[j+1] - y)
        + qvals[i+1][j] * (x - xvals[i]) * (yvals[j+1] - y)
        + qvals[i][j+1] * (xvals[i+1] - x) * (y - yvals[j])
        + qvals[i+1][j+1] * (x - xvals[i]) * (y - yvals[j]))
        / ((xvals[i+1] - xvals[i]) * (yvals[j+1] - yvals[j]));
}
