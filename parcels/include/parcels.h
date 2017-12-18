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
    RECTILINEAR_Z_GRID=0, RECTILINEAR_S_GRID=1, CURVILINEAR_Z_GRID=2, CURVILINEAR_S_GRID=3
  } GridCode;

typedef enum
  {
    LINEAR=0, NEAREST=1
  } InterpCode;

#define CHECKERROR(res) do {if (res != SUCCESS) return res;} while (0)

typedef struct
{
  int gtype;
  void *grid;
} CGrid;

typedef struct
{
  int xdim, ydim, zdim, tdim, z4d;
  float *lon, *lat, *depth;
  double *time;
} CRectilinearGrid;

typedef struct
{
  int xdim, ydim, zdim, tdim, allow_time_extrapolation, time_periodic;
  float ***data;
  CGrid *grid;
} CField;

typedef struct
{
  int xi, yi, zi, ti;
} CGridIndex;  

typedef struct
{ 
  int size;
  CGridIndex *gridIndices;
} CGridIndexSet;  

static inline ErrorCode search_indices_vertical_z(float z, int zdim, float *zvals, int *k, double *zeta)
{
  if (z < zvals[0] || z > zvals[zdim-1]) {return ERROR_OUT_OF_BOUNDS;}
  while (*k < zdim-1 && z > zvals[*k+1]) ++(*k);
  while (*k > 0 && z < zvals[*k]) --(*k);
  if (*k == zdim-1) {--*k;}

  *zeta = (z - zvals[*k]) / (zvals[*k+1] - zvals[*k]);
  return SUCCESS;
}

static inline ErrorCode search_indices_vertical_s(float z, int xdim, int ydim, int zdim, float *zvals,
                                    int i, int j, int *k, double xsi, double eta, double *zeta,
                                    int z4d, int ti, int tdim, double time, double t0, double t1)
{
  float zcol[zdim];
  int iz;
  if (z4d == 1){
    float (*zvalstab)[ydim][zdim][tdim] = (float (*)[ydim][zdim][tdim]) zvals;
    int ti1 = ti;
    if (ti < tdim-1)
       ti1= ti+1;
    double zt0, zt1;
    for (iz=0; iz < zdim; iz++){
      zt0 = (1-xsi)*(1-eta) * zvalstab[i  ][j  ][iz][ti]
          + (  xsi)*(1-eta) * zvalstab[i+1][j  ][iz][ti]
          + (  xsi)*(  eta) * zvalstab[i+1][j+1][iz][ti]
          + (1-xsi)*(  eta) * zvalstab[i  ][j+1][iz][ti];
      zt1 = (1-xsi)*(1-eta) * zvalstab[i  ][j  ][iz][ti1]
          + (  xsi)*(1-eta) * zvalstab[i+1][j  ][iz][ti1]
          + (  xsi)*(  eta) * zvalstab[i+1][j+1][iz][ti1]
          + (1-xsi)*(  eta) * zvalstab[i  ][j+1][iz][ti1];
      zcol[iz] = zt0 + (zt1 - zt0) * (float)((time - t0) / (t1 - t0));
    }

  }
  else{
    float (*zvalstab)[ydim][zdim] = (float (*)[ydim][zdim]) zvals;
    for (iz=0; iz < zdim; iz++){
      zcol[iz] = (1-xsi)*(1-eta) * zvalstab[i  ][j  ][iz]
               + (  xsi)*(1-eta) * zvalstab[i+1][j  ][iz]
               + (  xsi)*(  eta) * zvalstab[i+1][j+1][iz]
               + (1-xsi)*(  eta) * zvalstab[i  ][j+1][iz];
    }
  }

  if (z < zcol[0] || z > zcol[zdim-1]) {return ERROR_OUT_OF_BOUNDS;}
  while (*k < zdim-1 && z > zcol[*k+1]) ++(*k);
  while (*k > 0 && z < zcol[*k]) --(*k);
  if (*k == zdim-1) {--*k;}

  *zeta = (z - zcol[*k]) / (zcol[*k+1] - zcol[*k]);
  return SUCCESS;
}

static inline ErrorCode search_indices_rectilinear(float x, float y, float z, int xdim, int ydim, int zdim,
                                            float *xvals, float *yvals, float *zvals, GridCode gcode,
                                            int *i, int *j, int *k, double *xsi, double *eta, double *zeta,
                                            int z4d, int ti, int tdim, double time, double t0, double t1)
{
  if (x < xvals[0] || x > xvals[xdim-1]) {return ERROR_OUT_OF_BOUNDS;}
  while (*i < xdim-1 && x > xvals[*i+1]) ++(*i);
  while (*i > 0 && x < xvals[*i]) --(*i);
  /* Lowering index by 1 if last index, to avoid out-of-array sampling
  for index+1 in spatial-interpolation*/
  if (*i == xdim-1) {--*i;}

  if (y < yvals[0] || y > yvals[ydim-1]) {return ERROR_OUT_OF_BOUNDS;}
  while (*j < ydim-1 && y > yvals[*j+1]) ++(*j);
  while (*j > 0 && y < yvals[*j]) --(*j);
  /* Lowering index by 1 if last index, to avoid out-of-array sampling
  for index+1 in spatial-interpolation*/
  if (*j == ydim-1) {--*j;}

  *xsi = (x - xvals[*i]) / (xvals[*i+1] - xvals[*i]);
  *eta = (y - yvals[*j]) / (yvals[*j+1] - yvals[*j]);

  ErrorCode err;
  if (zdim > 1){
    switch(gcode){
      case RECTILINEAR_Z_GRID:
        err = search_indices_vertical_z(z, zdim, zvals, k, zeta);
        break;
      case RECTILINEAR_S_GRID:
        err = search_indices_vertical_s(z, xdim, ydim, zdim, zvals,
                                        *i, *j, k, *xsi, *eta, zeta,
                                        z4d, ti, tdim, time, t0, t1);
        break;
      default:
        err = ERROR;
    }
    CHECKERROR(err);
  }
  else
    *zeta = 0;

  if ( (*xsi < 0) || (*xsi > 1) ) return ERROR_OUT_OF_BOUNDS;
  if ( (*eta < 0) || (*eta > 1) ) return ERROR_OUT_OF_BOUNDS;
  if ( (*zeta < 0) || (*zeta > 1) ) return ERROR_OUT_OF_BOUNDS;

  return SUCCESS;
}

static inline ErrorCode search_indices_curvilinear(float x, float y, float z, int xdim, int ydim, int zdim,
                                            float *xvals, float *yvals, float *zvals, GridCode gcode,
                                            int *i, int *j, int *k, double *xsi, double *eta, double *zeta,
                                            int z4d, int ti, int tdim, double time, double t0, double t1)
{
  // NEMO convention
  float (* xgrid)[xdim] = (float (*)[xdim]) xvals;
  float (* ygrid)[xdim] = (float (*)[xdim]) yvals;

  float a[4], b[4];

  *xsi = *eta = -1;
  int maxIterSearch = 1e6, it = 0;
  while ( (*xsi < 0) || (*xsi > 1) || (*eta < 0) || (*eta > 1) ){
    a[0] =  xgrid[*j][*i];
    a[1] = -xgrid[*j][*i] + xgrid[*j][*i+1];
    a[2] = -xgrid[*j][*i]                                       + xgrid[*j+1][*i];
    a[3] =  xgrid[*j][*i] - xgrid[*j][*i+1] + xgrid[*j+1][*i+1] - xgrid[*j+1][*i];
    b[0] =  ygrid[*j][*i];
    b[1] = -ygrid[*j][*i] + ygrid[*j][*i+1];
    b[2] = -ygrid[*j][*i]                                       + ygrid[*j+1][*i];
    b[3] =  ygrid[*j][*i] - ygrid[*j][*i+1] + ygrid[*j+1][*i+1] - ygrid[*j+1][*i];

    double aa = a[3]*b[2] - a[2]*b[3];
    if (fabs(aa) < 1e-6){  // Rectilinear  cell, or quasi
      *xsi = ( (x-xgrid[*j][*i]) / (xgrid[*j][*i+1]-xgrid[*j][*i])
           +   (x-xgrid[*j+1][*i]) / (xgrid[*j+1][*i+1]-xgrid[*j+1][*i]) ) * .5;
      *eta = ( (y-ygrid[*j][*i]) / (ygrid[*j+1][*i]-ygrid[*j][*i])
           +   (y-ygrid[*j][*i+1]) / (ygrid[*j+1][*i+1]-ygrid[*j][*i+1]) ) * .5;
    }
    else{
      double bb = a[3]*b[0] - a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + x*b[3] - y*a[3];
      double cc = a[1]*b[0] - a[0]*b[1] + x*b[1] - y*a[1];
      double det = sqrt(bb*bb-4*aa*cc);
      *eta = (-bb+det)/(2*aa);
      *xsi = (x-a[0]-a[2]* (*eta)) / (a[1]+a[3]* (*eta));
    }
    if ( (*xsi < 0) && (*eta < 0) && (*i == 0) && (*j == 0) )
      return ERROR_OUT_OF_BOUNDS;
    if ( (*xsi > 1) && (*eta > 1) && (*i == xdim-1) && (*j == ydim-1) )
      return ERROR_OUT_OF_BOUNDS;
    if ( (*xsi < 0) && (*i > 0) )
      (*i)--;
    if ( (*xsi > 1) && (*i < xdim-1) )
      (*i)++;
    if ( (*eta < 0) && (*j > 0) )
      (*j)--;
    if ( (*eta > 1) && (*j < ydim-1) )
      (*j)++;
    it++;
    if ( it > maxIterSearch){
      printf("Correct cell not found after %d iterations\n", maxIterSearch);
      return ERROR_OUT_OF_BOUNDS;
    }
  }
  if ( (*xsi != *xsi) || (*eta != *eta) ){  // check if nan
      printf("xsi and or eta are nan values\n");
      return ERROR_OUT_OF_BOUNDS;
  }
  ErrorCode err;

  if (zdim > 1){
    switch(gcode){
      case CURVILINEAR_Z_GRID:
        err = search_indices_vertical_z(z, zdim, zvals, k, zeta);
        break;
      case CURVILINEAR_S_GRID:
        err = search_indices_vertical_s(z, xdim, ydim, zdim, zvals,
                                        *i, *j, k, *xsi, *eta, zeta,
                                        z4d, ti, tdim, time, t0, t1);
        break;
      default:
        err = ERROR;
    }
    CHECKERROR(err);
  }
  else
    *zeta = 0;

  if ( (*xsi < 0) || (*xsi > 1) ) return ERROR_OUT_OF_BOUNDS;
  if ( (*eta < 0) || (*eta > 1) ) return ERROR_OUT_OF_BOUNDS;
  if ( (*zeta < 0) || (*zeta > 1) ) return ERROR_OUT_OF_BOUNDS;

  return SUCCESS;
}

/* Local linear search to update grid index
 * params ti, sizeT, time. t0, t1 are only used for 4D S grids
 * */
static inline ErrorCode search_indices(float x, float y, float z, int xdim, int ydim, int zdim,
                                            float *xvals, float *yvals, float *zvals,
                                            int *i, int *j, int *k, double *xsi, double *eta, double *zeta,
                                            GridCode gcode, int z4d,
                                            int ti, int tdim, double time, double t0, double t1, float *z0, float *z1)
{
  switch(gcode){
    case RECTILINEAR_Z_GRID:
    case RECTILINEAR_S_GRID:
      return search_indices_rectilinear(x, y, z, xdim, ydim, zdim, xvals, yvals, zvals, gcode, i, j, k, xsi, eta, zeta,
                                   z4d, ti, tdim, time, t0, t1);
      break;
    case CURVILINEAR_Z_GRID:
    case CURVILINEAR_S_GRID:
      return search_indices_curvilinear(x, y, z, xdim, ydim, zdim, xvals, yvals, zvals, gcode, i, j, k, xsi, eta, zeta,
                                   z4d, ti, tdim, time, t0, t1);
      break;
    default:
      printf("Only RECTILINEAR_Z_GRID, RECTILINEAR_S_GRID, CURVILINEAR_Z_GRID and CURVILINEAR_S_GRID grids are currently implemented\n");
      return ERROR;
  }
}

/* Local linear search to update time index */
static inline ErrorCode search_time_index(double *t, int size, double *tvals, int *index, int time_periodic)
{
  if (time_periodic == 1){
    if (*t < tvals[0]){
      *index = size-1;      
      int periods = floor( (*t-tvals[0])/(tvals[size-1]-tvals[0]));
      *t -= periods * (tvals[size-1]-tvals[0]);
      search_time_index(t, size, tvals, index, time_periodic);
    }  
    else if (*t > tvals[size-1]){
      *index = 0;      
      int periods = floor( (*t-tvals[0])/(tvals[size-1]-tvals[0]));
      *t -= periods * (tvals[size-1]-tvals[0]);
      search_time_index(t, size, tvals, index, time_periodic);
    }  
  }          
  while (*index < size-1 && *t >= tvals[*index+1]) ++(*index);
  while (*index > 0 && *t < tvals[*index]) --(*index);
  return SUCCESS;
}

/* Bilinear interpolation routine for 2D grid */
static inline ErrorCode spatial_interpolation_bilinear(double xsi, double eta, int i, int j, int xdim, float **f_data, float *value)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  float (*data)[xdim] = (float (*)[xdim]) f_data;
  *value = (1-xsi)*(1-eta) * data[j  ][i  ]
         +    xsi *(1-eta) * data[j  ][i+1]
         +    xsi *   eta  * data[j+1][i+1]
         + (1-xsi)*   eta  * data[j+1][i  ];
  return SUCCESS;
}

/* Trilinear interpolation routine for 3D grid */
static inline ErrorCode spatial_interpolation_trilinear(double xsi, double eta, double zeta, int i, int j, int k,
                                                        int xdim, int ydim, float **f_data, float *value)
{
  float (*data)[ydim][xdim] = (float (*)[ydim][xdim]) f_data;
  float f0, f1;
  f0 = (1-xsi)*(1-eta) * data[k  ][j  ][i  ]
     +    xsi *(1-eta) * data[k  ][j  ][i+1]
     +    xsi *   eta  * data[k  ][j+1][i+1]
     + (1-xsi)*   eta  * data[k  ][j+1][i  ];
  f1 = (1-xsi)*(1-eta) * data[k+1][j  ][i  ]
     +    xsi *(1-eta) * data[k+1][j  ][i+1]
     +    xsi *   eta  * data[k+1][j+1][i+1]
     + (1-xsi)*   eta  * data[k+1][j+1][i  ];
  *value = (1-zeta) * f0 + zeta * f1;
  return SUCCESS;
}

/* Nearest neighbour interpolation routine for 2D grid */
static inline ErrorCode spatial_interpolation_nearest2D(double xsi, double eta, int i, int j, int xdim,
                                                        float **f_data, float *value)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  float (*data)[xdim] = (float (*)[xdim]) f_data;
  int ii, jj;
  if (xsi < .5) {ii = i;} else {ii = i + 1;}
  if (eta < .5) {jj = j;} else {jj = j + 1;}
  *value = data[jj][ii];
  return SUCCESS;
}

/* Nearest neighbour interpolation routine for 3D grid */
static inline ErrorCode spatial_interpolation_nearest3D(double xsi, double eta, double zeta, int i, int j, int k,
                                                        int xdim, int ydim, float **f_data, float *value)
{
  float (*data)[ydim][xdim] = (float (*)[ydim][xdim]) f_data;
  int ii, jj, kk;
  if (xsi < .5) {ii = i;} else {ii = i + 1;}
  if (eta < .5) {jj = j;} else {jj = j + 1;}
  if (zeta < .5) {kk = k;} else {kk = k + 1;}
  *value = data[kk][jj][ii];
  return SUCCESS;
}

/* Linear interpolation along the time axis */
static inline ErrorCode temporal_interpolation_structured_grid(float x, float y, float z, double time, CField *f, 
                                                               GridCode gcode, CGridIndex *gridIndex,
                                                               float *value, int interp_method)
{
  ErrorCode err;
  CRectilinearGrid *grid = f->grid->grid;

  /* Find time index for temporal interpolation */
  if (f->time_periodic == 0 && f->allow_time_extrapolation == 0 && (time < grid->time[0] || time > grid->time[grid->tdim-1])){
    return ERROR_TIME_EXTRAPOLATION;
  }
  err = search_time_index(&time, grid->tdim, grid->time, &gridIndex->ti, f->time_periodic);

  /* Cast data array intp data[time][depth][lat][lon] as per NEMO convention */
  float (*data)[f->zdim][f->ydim][f->xdim] = (float (*)[f->zdim][f->ydim][f->xdim]) f->data;
  double xsi, eta, zeta;

  float z0 = 0, z1 = 0;

  if (gridIndex->ti < grid->tdim-1 && time > grid->time[gridIndex->ti]) {
    float f0, f1;
    double t0 = grid->time[gridIndex->ti]; double t1 = grid->time[gridIndex->ti+1];
    /* Identify grid cell to sample through local linear search */
    err = search_indices(x, y, z, grid->xdim, grid->ydim, grid->zdim, grid->lon, grid->lat, grid->depth, &gridIndex->xi, &gridIndex->yi, &gridIndex->zi, &xsi, &eta, &zeta, gcode, grid->z4d, gridIndex->ti, grid->tdim, time, t0, t1, &z0, &z1); CHECKERROR(err);
    int i = gridIndex->xi;
    int j = gridIndex->yi;
    int k = gridIndex->zi;
    if (interp_method == LINEAR){
      if (grid->zdim==1){
        err = spatial_interpolation_bilinear(xsi, eta, i, j, grid->xdim, (float**)(data[gridIndex->ti]), &f0);
        err = spatial_interpolation_bilinear(xsi, eta, i, j, grid->xdim, (float**)(data[gridIndex->ti+1]), &f1);
      } else {
        err = spatial_interpolation_trilinear(xsi, eta, zeta, i, j, k, grid->xdim, grid->ydim, (float**)(data[gridIndex->ti]), &f0);
        err = spatial_interpolation_trilinear(xsi, eta, zeta, i, j, k, grid->xdim, grid->ydim, (float**)(data[gridIndex->ti+1]), &f0);
      }
    }
    else if  (interp_method == NEAREST){
      if (grid->zdim==1){
        err = spatial_interpolation_nearest2D(xsi, eta, i, j, grid->xdim, (float**)(data[gridIndex->ti]), &f0);
        err = spatial_interpolation_nearest2D(xsi, eta, i, j, grid->xdim, (float**)(data[gridIndex->ti+1]), &f1);
      } else {
        err = spatial_interpolation_nearest3D(xsi, eta, zeta, i, j, k, grid->xdim, grid->ydim,
                                              (float**)(data[gridIndex->ti]), &f0);
        err = spatial_interpolation_nearest3D(xsi, eta, zeta, i, j, k, grid->xdim, grid->ydim,
                                              (float**)(data[gridIndex->ti+1]), &f1);
      }
    }
    else {
        return ERROR;
    }
    *value = f0 + (f1 - f0) * (float)((time - t0) / (t1 - t0));
    return SUCCESS;
  } else {
    double t0 = grid->time[gridIndex->ti];
    err = search_indices(x, y, z, grid->xdim, grid->ydim, grid->zdim, grid->lon, grid->lat, grid->depth, &gridIndex->xi, &gridIndex->yi, &gridIndex->zi, &xsi, &eta, &zeta, gcode, grid->z4d, gridIndex->ti, grid->tdim, t0, t0, t0+1, &z0, &z1); CHECKERROR(err);
    int i = gridIndex->xi;
    int j = gridIndex->yi;
    int k = gridIndex->zi;
    if (interp_method == LINEAR){
      if (grid->zdim==1)
        err = spatial_interpolation_bilinear(xsi, eta, i, j, grid->xdim, (float**)(data[gridIndex->ti]), value);
      else
        err = spatial_interpolation_trilinear(xsi, eta, zeta, i, j, k, grid->xdim, grid->ydim,
                                             (float**)(data[gridIndex->ti]), value);
    }
    else if (interp_method == NEAREST){
      if (grid->zdim==1)
        err = spatial_interpolation_nearest2D(xsi, eta, i, j, grid->xdim, (float**)(data[gridIndex->ti]), value);
      else {
        err = spatial_interpolation_nearest3D(xsi, eta, zeta, i, j, k, grid->xdim, grid->ydim,
                                             (float**)(data[gridIndex->ti]), value);
      }
    }
    else {
        return ERROR;    
    }
    return SUCCESS;
  }
}

static inline ErrorCode temporal_interpolation(float x, float y, float z, double time, CField *f, 
                                               void *gridIndexSet, int iGrid, float *value, int interp_method)
{
  CGrid *_grid = f->grid;
  GridCode gcode = _grid->gtype;
  CGridIndexSet *giset = (CGridIndexSet *) gridIndexSet;
  CGridIndex *gridIndex = &giset->gridIndices[iGrid];

  if (gcode == RECTILINEAR_Z_GRID || gcode == RECTILINEAR_S_GRID || gcode == CURVILINEAR_Z_GRID || gcode == CURVILINEAR_S_GRID)
    return temporal_interpolation_structured_grid(x, y, z, time, f, gcode, gridIndex, value, interp_method);
  else{
    printf("Only RECTILINEAR_Z_GRID, RECTILINEAR_S_GRID, CURVILINEAR_Z_GRID and CURVILINEAR_S_GRID grids are currently implemented\n");
    return ERROR;
  }
}

static inline ErrorCode temporal_interpolationUV(float x, float y, float z, double time,
                                                 CField *U, CField *V,
                                                 void *gridIndexSet, int uiGrid, int viGrid,
                                                 float *valueU, float *valueV, int interp_method)
{
  ErrorCode err;

  err = temporal_interpolation(x, y, z, time, U, gridIndexSet, uiGrid, valueU, interp_method); CHECKERROR(err);
  err = temporal_interpolation(x, y, z, time, V, gridIndexSet, viGrid, valueV, interp_method); CHECKERROR(err);

  return SUCCESS;
}

static inline ErrorCode temporal_interpolationUVrotation(float x, float y, float z, double time,
                                                 CField *U, CField *V, CField *cosU, CField *sinU, CField *cosV, CField *sinV,
                                                 void *gridIndexSet, int uiGrid, int viGrid, int cosuiGrid, 
                                                 int sinuiGrid, int cosviGrid, int sinviGrid,
                                                 float *valueU, float *valueV, int interp_method)
{
  ErrorCode err;

  float u_val, v_val, cosU_val, sinU_val, cosV_val, sinV_val;
  err = temporal_interpolation(x, y, z, time, U, gridIndexSet, uiGrid, &u_val, interp_method); CHECKERROR(err);
  err = temporal_interpolation(x, y, z, time, V, gridIndexSet, viGrid, &v_val, interp_method); CHECKERROR(err);
  err = temporal_interpolation(x, y, z, time, cosU, gridIndexSet, cosuiGrid, &cosU_val, interp_method); CHECKERROR(err);
  err = temporal_interpolation(x, y, z, time, sinU, gridIndexSet, sinuiGrid, &sinU_val, interp_method); CHECKERROR(err);
  err = temporal_interpolation(x, y, z, time, cosV, gridIndexSet, cosviGrid, &cosV_val, interp_method); CHECKERROR(err);
  err = temporal_interpolation(x, y, z, time, sinV, gridIndexSet, sinviGrid, &sinV_val, interp_method); CHECKERROR(err);

  *valueU = u_val * cosU_val - v_val * sinV_val;
  *valueV = u_val * sinU_val + v_val * cosV_val;

  return SUCCESS;
}



/**************************************************/


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
