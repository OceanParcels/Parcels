#include <stdio.h>
#include <stdlib.h>

typedef enum
  {
    SUCCESS=0, REPEAT=1, DELETE=2, FAIL=3, FAIL_OUT_OF_BOUNDS=4
  } KernelOp;

typedef struct
{
  int xdim, ydim, tdim, tidx;
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
static inline float spatial_interpolation_bilinear(float x, float y, int i, int j, int xdim,
                                                   float *lon, float *lat, float **f_data)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  float (*data)[xdim] = (float (*)[xdim]) f_data;
  return (data[j][i] * (lon[i+1] - x) * (lat[j+1] - y)
        + data[j][i+1] * (x - lon[i]) * (lat[j+1] - y)
        + data[j+1][i] * (lon[i+1] - x) * (y - lat[j])
        + data[j+1][i+1] * (x - lon[i]) * (y - lat[j]))
        / ((lon[i+1] - lon[i]) * (lat[j+1] - lat[j]));
}

/* Linear interpolation along the time axis */
static inline float temporal_interpolation_linear(float x, float y, int xi, int yi,
                                                  double time, CField *f)
{
  /* Cast data array intp data[time][lat][lon] as per NEMO convention */
  float (*data)[f->ydim][f->xdim] = (float (*)[f->ydim][f->xdim]) f->data;
  float f0, f1;
  double t0, t1;
  int i = xi, j = yi;
  /* Identify grid cell to sample through local linear search */
  i = search_linear_float(x, i, f->xdim, f->lon);
  j = search_linear_float(y, j, f->ydim, f->lat);
  /* Find time index for temporal interpolation */
  f->tidx = search_linear_double(time, f->tidx, f->tdim, f->time);
  if (f->tidx < f->tdim-1 && time > f->time[f->tidx]) {
    t0 = f->time[f->tidx]; t1 = f->time[f->tidx+1];
    f0 = spatial_interpolation_bilinear(x, y, i, j, f->xdim, f->lon, f->lat, (float**)(data[f->tidx]));
    f1 = spatial_interpolation_bilinear(x, y, i, j, f->xdim, f->lon, f->lat, (float**)(data[f->tidx+1]));
    return f0 + (f1 - f0) * (float)((time - t0) / (t1 - t0));
  } else {
    return spatial_interpolation_bilinear(x, y, i, j, f->xdim, f->lon, f->lat, (float**)(data[f->tidx]));
  }
}

/**************************************************/
/*   Random number generation (RNG) functions     */
/**************************************************/

static void parcels_seed(int seed)
{
  srand(seed);
}

static inline float parcels_random()
{
  return (float)rand()/(float)(RAND_MAX);
}

static inline float parcels_uniform(float low, float high)
{
  return (float)rand()/(float)(RAND_MAX / (high-low)) + low;
}

static inline int parcels_randint(int low, int high)
{
  return (rand() % (high-low)) + low;
}
