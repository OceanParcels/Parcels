#include <stdio.h>
#include <stdlib.h>

typedef enum
  {
    SUCCESS=0, REPEAT=1, DELETE=2, ERROR=3, ERROR_OUT_OF_BOUNDS=4
  } ErrorCode;

typedef enum
  {
    LINEAR=0, NEAREST=1
  } InterpCode;

#define CHECKERROR(res) do {if (res != SUCCESS) return res;} while (0)

typedef struct
{
  int xdim, ydim, tdim, tidx, allow_time_extrapolation;
  float *lon, *lat;
  double *time;
  float ***data;
} CField;


/* Local linear search to update grid index */
static inline ErrorCode search_linear_float(float x, int size, float *xvals, int *index)
{
  if (x < xvals[0] || xvals[size-1] < x) {return ERROR_OUT_OF_BOUNDS;}
  while (*index < size-1 && x > xvals[*index+1]) ++(*index);
  while (*index > 0 && x < xvals[*index]) --(*index);
  return SUCCESS;
}

/* Local linear search to update time index */
static inline ErrorCode search_linear_double(double t, int size, double *tvals, int *index)
{
  while (*index < size-1 && t > tvals[*index+1]) ++(*index);
  while (*index > 0 && t < tvals[*index]) --(*index);
  return SUCCESS;
}

/* Bilinear interpolation routine for 2D grid */
static inline ErrorCode spatial_interpolation_bilinear(float x, float y, int i, int j, int xdim,
                                                       float *lon, float *lat, float **f_data,
                                                       float *value)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  float (*data)[xdim] = (float (*)[xdim]) f_data;
  *value = (data[j][i] * (lon[i+1] - x) * (lat[j+1] - y)
            + data[j][i+1] * (x - lon[i]) * (lat[j+1] - y)
            + data[j+1][i] * (lon[i+1] - x) * (y - lat[j])
            + data[j+1][i+1] * (x - lon[i]) * (y - lat[j]))
            / ((lon[i+1] - lon[i]) * (lat[j+1] - lat[j]));
  return SUCCESS;
}

/* Nearest neighbour interpolation routine for 2D grid */
static inline ErrorCode spatial_interpolation_nearest2D(float x, float y, int i, int j, int xdim,
                                                        float *lon, float *lat, float **f_data,
                                                        float *value)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  float (*data)[xdim] = (float (*)[xdim]) f_data;
  int ii, jj;
  if (x - lon[i] < lon[i+1] - x) {ii = i;} else {ii = i + 1;}
  if (y - lat[j] < lat[j+1] - y) {jj = j;} else {jj = j + 1;}
  *value = data[jj][ii];
  return SUCCESS;
}

/* Linear interpolation along the time axis */
static inline ErrorCode temporal_interpolation_linear(float x, float y, int xi, int yi,
                                                      double time, CField *f, float *value,
                                                      int interp_method)
{
  ErrorCode err;
  /* Cast data array intp data[time][lat][lon] as per NEMO convention */
  float (*data)[f->ydim][f->xdim] = (float (*)[f->ydim][f->xdim]) f->data;
  float f0, f1;
  double t0, t1;
  int i = xi, j = yi;
  /* Identify grid cell to sample through local linear search */
  err = search_linear_float(x, f->xdim, f->lon, &i); CHECKERROR(err);
  err = search_linear_float(y, f->ydim, f->lat, &j); CHECKERROR(err);
  /* Find time index for temporal interpolation */
  if (f->allow_time_extrapolation == 0 && (time < f->time[0] || time > f->time[f->tdim-1])){
    CHECKERROR(ERROR);
  }
  err = search_linear_double(time, f->tdim, f->time, &(f->tidx));
  if (f->tidx < f->tdim-1 && time > f->time[f->tidx]) {
    t0 = f->time[f->tidx]; t1 = f->time[f->tidx+1];
    if (interp_method == LINEAR){
      err = spatial_interpolation_bilinear(x, y, i, j, f->xdim, f->lon, f->lat,
                                          (float**)(data[f->tidx]), &f0);
      err = spatial_interpolation_bilinear(x, y, i, j, f->xdim, f->lon, f->lat,
                                          (float**)(data[f->tidx+1]), &f1);
    }
    else if  (interp_method == NEAREST){
      err = spatial_interpolation_nearest2D(x, y, i, j, f->xdim, f->lon, f->lat,
                                           (float**)(data[f->tidx]), &f0);
      err = spatial_interpolation_nearest2D(x, y, i, j, f->xdim, f->lon, f->lat,
                                           (float**)(data[f->tidx+1]), &f1);
    }
    else {
        return ERROR;
    }
    *value = f0 + (f1 - f0) * (float)((time - t0) / (t1 - t0));
    return SUCCESS;
  } else {
    if (interp_method == LINEAR){
      err = spatial_interpolation_bilinear(x, y, i, j, f->xdim, f->lon, f->lat,
                                          (float**)(data[f->tidx]), value);
    }
    else if (interp_method == NEAREST){
      err = spatial_interpolation_nearest2D(x, y, i, j, f->xdim, f->lon, f->lat,
                                           (float**)(data[f->tidx]), value);
    }
    else {
        return ERROR;    
    }
    return SUCCESS;
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
