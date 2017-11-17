#ifndef _PARCELS_H
#define _PARCELS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef enum
  {
    SUCCESS=0, REPEAT=1, DELETE=2, ERROR=3, ERROR_OUT_OF_BOUNDS=4, ERROR_TIME_EXTRAPOLATION =5
  } ErrorCode;

typedef enum
  {
    STRUCTURED_GRID=0, SEMI_STRUCTURED_GRID=1
  } GridCode;

typedef enum
  {
    LINEAR=0, NEAREST=1
  } InterpCode;

#define CHECKERROR(res) do {if (res != SUCCESS) return res;} while (0)

typedef struct
{
  char *name;
  int gtype;
  void *grid;
} CGrid;

typedef struct
{
  char *name;
  int xdim, ydim, zdim, tdim, tidx;
  float *lon, *lat, *depth;
  double *time;
} CStructuredGrid;

typedef struct
{
  int xdim, ydim, zdim, tdim, tidx, allow_time_extrapolation, time_periodic;
  float ***data;
  CGrid *grid;
} CField;

typedef struct
{
  char *name;
  int xi, yi, zi, pad;
  //CGrid *grid;
} CGridIndex;  

typedef struct
{ 
  int size;
  CGridIndex *gridIndices;
} CGridIndexSet;  



/* Local linear search to update grid index */
static inline ErrorCode search_linear_float(float x, float y, float z, int sizeX, int sizeY, int sizeZ,
                                            float *xvals, float *yvals, float *zvals,
                                            int *i, int *j, int *k)
{
  if (x < xvals[0] || x > xvals[sizeX-1]) {return ERROR_OUT_OF_BOUNDS;}
  while (*i < sizeX-1 && x > xvals[*i+1]) ++(*i);
  while (*i > 0 && x < xvals[*i]) --(*i);

  /* Lowering index by 1 if last index, to avoid out-of-array sampling
  for index+1 in spatial-interpolation*/
  if (*i == sizeX-1) {--*i;}

  if (y < yvals[0] || y > yvals[sizeY-1]) {return ERROR_OUT_OF_BOUNDS;}
  while (*j < sizeY-1 && y > yvals[*j+1]) ++(*j);
  while (*j > 0 && y < yvals[*j]) --(*j);

  /* Lowering index by 1 if last index, to avoid out-of-array sampling
  for index+1 in spatial-interpolation*/
  if (*j == sizeY-1) {--*j;}

  if (sizeZ > 1)
  {
    if (z < zvals[0] || z > zvals[sizeZ-1]) {return ERROR_OUT_OF_BOUNDS;}
    while (*k < sizeZ-1 && z > zvals[*k+1]) ++(*k);
    while (*k > 0 && x < zvals[*k]) --(*k);

    /* Lowering index by 1 if last index, to avoid out-of-array sampling
       for index+1 in spatial-interpolation*/
    if (*k == sizeZ-1) {--*k;}
  }
  return SUCCESS;
}

/* Local linear search to update time index */
static inline ErrorCode search_linear_double(double *t, int size, double *tvals, int *index, int time_periodic)
{
  if (time_periodic == 1){
    if (*t < tvals[0]){
      *index = size-1;      
      int periods = floor( (*t-tvals[0])/(tvals[size-1]-tvals[0]));
      *t -= periods * (tvals[size-1]-tvals[0]);
      search_linear_double(t, size, tvals, index, time_periodic);
    }  
    else if (*t > tvals[size-1]){
      *index = 0;      
      int periods = floor( (*t-tvals[0])/(tvals[size-1]-tvals[0]));
      *t -= periods * (tvals[size-1]-tvals[0]);
      search_linear_double(t, size, tvals, index, time_periodic);
    }  
  }          
  while (*index < size-1 && *t >= tvals[*index+1]) ++(*index);
  while (*index > 0 && *t < tvals[*index]) --(*index);
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

/* Trilinear interpolation routine for 3D grid */
static inline ErrorCode spatial_interpolation_trilinear(float x, float y, float z, int i, int j, int k,
                                                        int xdim, int ydim, float *lon, float *lat,
                                                        float *depth, float **f_data, float *value)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  float (*data)[ydim][xdim] = (float (*)[ydim][xdim]) f_data;
  float f0, f1, z0, z1;
  z0 = depth[k]; z1 = depth[k+1];
  f0 = (data[k][j][i] * (lon[i+1] - x) * (lat[j+1] - y)
        + data[k][j][i+1] * (x - lon[i]) * (lat[j+1] - y)
        + data[k][j+1][i] * (lon[i+1] - x) * (y - lat[j])
        + data[k][j+1][i+1] * (x - lon[i]) * (y - lat[j]))
        / ((lon[i+1] - lon[i]) * (lat[j+1] - lat[j]));
  f1 = (data[k+1][j][i] * (lon[i+1] - x) * (lat[j+1] - y)
        + data[k+1][j][i+1] * (x - lon[i]) * (lat[j+1] - y)
        + data[k+1][j+1][i] * (lon[i+1] - x) * (y - lat[j])
        + data[k+1][j+1][i+1] * (x - lon[i]) * (y - lat[j]))
        / ((lon[i+1] - lon[i]) * (lat[j+1] - lat[j]));
  *value = f0 + (f1 - f0) * (float)((z - z0) / (z1 - z0));
  return SUCCESS;
}

/* Nearest neighbour interpolation routine for 2D grid */
static inline ErrorCode spatial_interpolation_nearest2D(float x, float y, int i, int j, int xdim, int ydim,
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

/* Nearest neighbour interpolation routine for 3D grid */
static inline ErrorCode spatial_interpolation_nearest3D(float x, float y, float z, int i, int j, int k,
                                                        int xdim, int ydim, int zdim, float *lon, float *lat,
                                                        float *depth, float **f_data, float *value)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  float (*data)[ydim][xdim] = (float (*)[ydim][xdim]) f_data;
  int ii, jj, kk;
  if (x - lon[i] < lon[i+1] - x) {ii = i;} else {ii = i + 1;}
  if (y - lat[j] < lat[j+1] - y) {jj = j;} else {jj = j + 1;}
  if (z - depth[k] < depth[k+1] - z) {kk = k;} else {kk = k + 1;}
  *value = data[kk][jj][ii];
  return SUCCESS;
}

/* Linear interpolation along the time axis */
static inline ErrorCode temporal_interpolation_linear_structured_grid(float x, float y, float z, void *gridIndexSet,
                                                                      double time, CField *f,
                                                                      float *value, int interp_method)
{
  ErrorCode err;
  /* Cast data array intp data[time][lat][lon] as per NEMO convention */
  CStructuredGrid *grid = f->grid->grid;
  int iGrid;
  CGridIndexSet *giset = (CGridIndexSet *) gridIndexSet;
  CGridIndex *gridIndex = NULL;
  for(iGrid = 0; iGrid < giset->size; iGrid++){
    CGridIndex *gIndex = &giset->gridIndices[iGrid];
    if(grid->name == gIndex->name){
      gridIndex = gIndex;
      break;
    }
  }
  if (gridIndex == NULL){
    printf("Grid not found\n");
    exit(-1);
  }
  /* Identify grid cell to sample through local linear search */
  err = search_linear_float(x, y, z, grid->xdim, grid->ydim, grid->zdim, grid->lon, grid->lat, grid->depth, &gridIndex->xi, &gridIndex->yi, &gridIndex->zi); CHECKERROR(err);
  int i = gridIndex->xi;
  int j = gridIndex->yi;
  int k = gridIndex->zi;

  float (*data)[f->zdim][f->ydim][f->xdim] = (float (*)[f->zdim][f->ydim][f->xdim]) f->data;
  float f0, f1;
  double t0, t1;
  /* Find time index for temporal interpolation */
  if (f->time_periodic == 0 && f->allow_time_extrapolation == 0 && (time < grid->time[0] || time > grid->time[grid->tdim-1])){
    return ERROR_TIME_EXTRAPOLATION;
  }
  err = search_linear_double(&time, grid->tdim, grid->time, &(grid->tidx), f->time_periodic);
  if (grid->tidx < grid->tdim-1 && time > grid->time[grid->tidx]) {
    t0 = grid->time[grid->tidx]; t1 = grid->time[grid->tidx+1];
    if (interp_method == LINEAR){
      if (grid->zdim==1){
        err = spatial_interpolation_bilinear(x, y, i, j, grid->xdim, grid->lon, grid->lat,
                                             (float**)(data[grid->tidx]), &f0);
        err = spatial_interpolation_bilinear(x, y, i, j, grid->xdim, grid->lon, grid->lat,
                                             (float**)(data[grid->tidx+1]), &f1);
      } else {
        err = spatial_interpolation_trilinear(x, y, z, i, j, k, grid->xdim, grid->ydim,
                                              grid->lon, grid->lat, grid->depth,
                                              (float**)(data[grid->tidx]), &f0);
        err = spatial_interpolation_trilinear(x, y, z, i, j, k, grid->xdim, grid->ydim,
                                              grid->lon, grid->lat, grid->depth,
                                              (float**)(data[grid->tidx+1]), &f1);
      }
    }
    else if  (interp_method == NEAREST){
      if (grid->zdim==1){
        err = spatial_interpolation_nearest2D(x, y, i, j, grid->xdim, grid->ydim, grid->lon,
                                              grid->lat, (float**)(data[grid->tidx]), &f0);
        err = spatial_interpolation_nearest2D(x, y, i, j, grid->xdim, grid->ydim, grid->lon,
                                              grid->lat, (float**)(data[grid->tidx+1]), &f1);
      } else {
        err = spatial_interpolation_nearest3D(x, y, z, i, j, k, grid->xdim, grid->ydim,
                                              grid->zdim, grid->lon, grid->lat, grid->depth,
                                              (float**)(data[grid->tidx]), &f0);
        err = spatial_interpolation_nearest3D(x, y, z, i, j, k, grid->xdim, grid->ydim,
                                              grid->zdim, grid->lon, grid->lat, grid->depth,
                                              (float**)(data[grid->tidx+1]), &f1);
      }
    }
    else {
        return ERROR;
    }
    *value = f0 + (f1 - f0) * (float)((time - t0) / (t1 - t0));
    return SUCCESS;
  } else {
    if (interp_method == LINEAR){
      if (grid->zdim==1){
        err = spatial_interpolation_bilinear(x, y, i, j, grid->xdim, grid->lon, grid->lat,
                                             (float**)(data[grid->tidx]), value);
      } else {
        err = spatial_interpolation_trilinear(x, y, z, i, j, k, grid->xdim, grid->ydim,
                                              grid->lon, grid->lat, grid->depth,
                                              (float**)(data[grid->tidx]), value);
      }
    }
    else if (interp_method == NEAREST){
      if (grid->zdim==1){
        err = spatial_interpolation_nearest2D(x, y, i, j, grid->xdim, grid->ydim, grid->lon,
                                              grid->lat, (float**)(data[grid->tidx]), value);
      } else {
        err = spatial_interpolation_nearest3D(x, y, z, i, j, k, grid->xdim, grid->ydim,
                                              grid->zdim, grid->lon, grid->lat, grid->depth,
                                              (float**)(data[grid->tidx]), value);
      }
    }
    else {
        return ERROR;    
    }
    return SUCCESS;
  }
}

static inline ErrorCode temporal_interpolation_linear(float x, float y, float z, void *gridIndexSet, double time, CField *f,
                                                      float *value, int interp_method)
{
  CGrid *_grid = f->grid;
  GridCode gcode = _grid->gtype;
  if (gcode == STRUCTURED_GRID)
    return temporal_interpolation_linear_structured_grid(x, y, z, gridIndexSet, time, f, value, interp_method);
  else{
    printf("Only STRUCTURED_GRID grids are currently implemented\n");
    return ERROR;
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

static inline float parcels_normalvariate(float loc, float scale)
/* Function to create a Gaussian random variable with mean loc and standard deviation scale */
/* Uses Box-Muller transform, adapted from ftp://ftp.taygeta.com/pub/c/boxmuller.c          */
/*     (c) Copyright 1994, Everett F. Carter Jr. Permission is granted by the author to use */
/*     this software for any application provided this copyright notice is preserved.       */
{
  float x1, x2, w, y1;
  static float y2;

  do {
    x1 = 2.0 * (float)rand()/(float)(RAND_MAX) - 1.0;
    x2 = 2.0 * (float)rand()/(float)(RAND_MAX) - 1.0;
    w = x1 * x1 + x2 * x2;
  } while ( w >= 1.0 );

  w = sqrt( (-2.0 * log( w ) ) / w );
  y1 = x1 * w;
  y2 = x2 * w;
  return( loc + y1 * scale );
}
#ifdef __cplusplus
}
#endif
#endif
