#include <stdio.h>

typedef struct
{
  int xdim, ydim, tdim;
  float *lon, *lat;
  double *time;
  float ***data;
} CField;


/* Local linear search to update grid index */
static inline int search_linear_float(float x, int i, int size, float *xvals)
{
    while (i < size-1 && x > xvals[i+1]) ++i;
    while (i > 0 && x < xvals[i]) --i;
    return i;
}

/* Local linear search to update time index */
static inline int search_linear_double(double t, int i, int size, double *tvals)
{
    while (i < size-1 && t > tvals[i+1]) ++i;
    while (i > 0 && t < tvals[i]) --i;
    return i;
}

/* Bilinear interpolation routine for 2D grid */
static inline float spatial_interpolation_bilinear(float x, float y, int i, int j, int ydim,
                                                   float *lon, float *lat, float **f_data)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  float (*data)[ydim] = (float (*)[ydim]) f_data;
  return (data[i][j] * (lon[i+1] - x) * (lat[j+1] - y)
        + data[i+1][j] * (x - lon[i]) * (lat[j+1] - y)
        + data[i][j+1] * (lon[i+1] - x) * (y - lat[j])
        + data[i+1][j+1] * (x - lon[i]) * (y - lat[j]))
        / ((lon[i+1] - lon[i]) * (lat[j+1] - lat[j]));
}

/* Linear interpolation along the time axis */
static inline float temporal_interpolation_linear(float x, float y, int xi, int yi,
                                                  double time, CField *f)
{
  /* Cast data array intp data[time][lat][lon] as per NEMO convention */
  float (*data)[f->xdim][f->ydim] = (float (*)[f->xdim][f->ydim]) f->data;
  float f0, f1;
  double t0, t1;
  int i = xi, j = yi, t = 0;
  /* Identify grid cell to sample through local linear search */
  i = search_linear_float(x, i, f->xdim, f->lon);
  j = search_linear_float(y, j, f->ydim, f->lat);
  /* Find time index for temporal interpolation */
  t = search_linear_double(time, t, f->tdim, f->time);

  if (t < f->tdim-1) {
    t0 = f->time[t]; t1 = f->time[t+1];
    f0 = spatial_interpolation_bilinear(x, y, i, j, f->ydim, f->lon, f->lat, (float**)(data[t]));
    f1 = spatial_interpolation_bilinear(x, y, i, j, f->ydim, f->lon, f->lat, (float**)(data[t+1]));
    return f0 + (f1 - f0) * (float)((time - t0) / (t1 - t0));
  } else {
    return spatial_interpolation_bilinear(x, y, i, j, f->ydim, f->lon, f->lat, (float**)(data[t]));
  }
}
