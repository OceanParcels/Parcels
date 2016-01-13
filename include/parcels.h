#include <stdio.h>

typedef struct
{
  int xdim, ydim;
  float *lon, *lat;
  float **data;
} CField;


/* Local search to update grid index */
static inline int advance_index(float x, int i, int size, float *xvals)
{
    while (i < size-1 && x > xvals[i+1]) ++i;
    while (i > 0 && x < xvals[i]) --i;
    return i;
}


/* Bilinear interpolation routine for 2D grid */
static inline float interpolate_bilinear(float x, float y, int xi, int yi, CField *f)
{
  /* Cast data array intp data[lat][lon] as per NEMO data convention */
  float (*data)[f->ydim] = (float (*)[f->ydim]) f->data;
  int i = xi, j = yi;
  i = advance_index(x, i, f->xdim, f->lon);
  j = advance_index(y, j, f->ydim, f->lat);
  return (data[i][j] * (f->lon[i+1] - x) * (f->lat[j+1] - y)
        + data[i+1][j] * (x - f->lon[i]) * (f->lat[j+1] - y)
        + data[i][j+1] * (f->lon[i+1] - x) * (y - f->lat[j])
        + data[i+1][j+1] * (x - f->lon[i]) * (y - f->lat[j]))
        / ((f->lon[i+1] - f->lon[i]) * (f->lat[j+1] - f->lat[j]));
}
