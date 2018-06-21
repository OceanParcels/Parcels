#ifndef _PARCELS_H
#define _PARCELS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "random_parcels.h"
#include "index_search.h"

typedef enum
  {
    LINEAR=0, NEAREST=1
  } InterpCode;

typedef struct
{
  int gtype;
  void *grid;
} CGrid;

typedef struct
{
  int xdim, ydim, zdim, tdim, z4d;
  int sphere_mesh, zonal_periodic;
  float *lon, *lat, *depth;
  double *time;
} CStructuredGrid;

typedef struct
{
  int xdim, ydim, zdim, tdim, igrid, allow_time_extrapolation, time_periodic;
  float ***data;
  CGrid *grid;
} CField;

/* Bilinear interpolation routine for 2D grid */
static inline ErrorCode spatial_interpolation_bilinear(double xsi, double eta, int xi, int yi, int xdim, float **f_data, float *value)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  float (*data)[xdim] = (float (*)[xdim]) f_data;
  *value = (1-xsi)*(1-eta) * data[yi  ][xi  ]
         +    xsi *(1-eta) * data[yi  ][xi+1]
         +    xsi *   eta  * data[yi+1][xi+1]
         + (1-xsi)*   eta  * data[yi+1][xi  ];
  return SUCCESS;
}

/* Trilinear interpolation routine for 3D grid */
static inline ErrorCode spatial_interpolation_trilinear(double xsi, double eta, double zeta, int xi, int yi, int zi,
                                                        int xdim, int ydim, float **f_data, float *value)
{
  float (*data)[ydim][xdim] = (float (*)[ydim][xdim]) f_data;
  float f0, f1;
  f0 = (1-xsi)*(1-eta) * data[zi  ][yi  ][xi  ]
     +    xsi *(1-eta) * data[zi  ][yi  ][xi+1]
     +    xsi *   eta  * data[zi  ][yi+1][xi+1]
     + (1-xsi)*   eta  * data[zi  ][yi+1][xi  ];
  f1 = (1-xsi)*(1-eta) * data[zi+1][yi  ][xi  ]
     +    xsi *(1-eta) * data[zi+1][yi  ][xi+1]
     +    xsi *   eta  * data[zi+1][yi+1][xi+1]
     + (1-xsi)*   eta  * data[zi+1][yi+1][xi  ];
  *value = (1-zeta) * f0 + zeta * f1;
  return SUCCESS;
}

/* Nearest neighbour interpolation routine for 2D grid */
static inline ErrorCode spatial_interpolation_nearest2D(double xsi, double eta, int xi, int yi, int xdim,
                                                        float **f_data, float *value)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  float (*data)[xdim] = (float (*)[xdim]) f_data;
  int ii, jj;
  if (xsi < .5) {ii = xi;} else {ii = xi + 1;}
  if (eta < .5) {jj = yi;} else {jj = yi + 1;}
  *value = data[jj][ii];
  return SUCCESS;
}

/* Nearest neighbour interpolation routine for 3D grid */
static inline ErrorCode spatial_interpolation_nearest3D(double xsi, double eta, double zeta, int xi, int yi, int zi,
                                                        int xdim, int ydim, float **f_data, float *value)
{
  float (*data)[ydim][xdim] = (float (*)[ydim][xdim]) f_data;
  int ii, jj, kk;
  if (xsi < .5) {ii = xi;} else {ii = xi + 1;}
  if (eta < .5) {jj = yi;} else {jj = yi + 1;}
  if (zeta < .5) {kk = zi;} else {kk = zi + 1;}
  *value = data[kk][jj][ii];
  return SUCCESS;
}

/* Linear interpolation along the time axis */
static inline ErrorCode temporal_interpolation_structured_grid(float x, float y, float z, double time, CField *f, 
                                                               GridCode gcode, int *xi, int *yi, int *zi, int *ti,
                                                               float *value, int interp_method)
{
  ErrorCode err;
  CStructuredGrid *grid = f->grid->grid;
  int igrid = f->igrid;

  /* Find time index for temporal interpolation */
  if (f->time_periodic == 0 && f->allow_time_extrapolation == 0 && (time < grid->time[0] || time > grid->time[grid->tdim-1])){
    return ERROR_TIME_EXTRAPOLATION;
  }
  err = search_time_index(&time, grid->tdim, grid->time, &ti[igrid], f->time_periodic);

  /* Cast data array intp data[time][depth][lat][lon] as per NEMO convention */
  float (*data)[f->zdim][f->ydim][f->xdim] = (float (*)[f->zdim][f->ydim][f->xdim]) f->data;
  double xsi, eta, zeta;


  if (ti[igrid] < grid->tdim-1 && time > grid->time[ti[igrid]]) {
    float f0, f1;
    double t0 = grid->time[ti[igrid]]; double t1 = grid->time[ti[igrid]+1];
    /* Identify grid cell to sample through local linear search */
    err = search_indices(x, y, z, grid->xdim, grid->ydim, grid->zdim, grid->lon, grid->lat, grid->depth, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zeta, grid->sphere_mesh, grid->zonal_periodic, gcode, grid->z4d, ti[igrid], grid->tdim, time, t0, t1); CHECKERROR(err);
    if (interp_method == LINEAR){
      if (grid->zdim==1){
        err = spatial_interpolation_bilinear(xsi, eta, xi[igrid], yi[igrid], grid->xdim, (float**)(data[ti[igrid]]), &f0);
        err = spatial_interpolation_bilinear(xsi, eta, xi[igrid], yi[igrid], grid->xdim, (float**)(data[ti[igrid]+1]), &f1);
      } else {
        err = spatial_interpolation_trilinear(xsi, eta, zeta, xi[igrid], yi[igrid], zi[igrid], grid->xdim, grid->ydim, (float**)(data[ti[igrid]]), &f0);
        err = spatial_interpolation_trilinear(xsi, eta, zeta, xi[igrid], yi[igrid], zi[igrid], grid->xdim, grid->ydim, (float**)(data[ti[igrid]+1]), &f1);
      }
    }
    else if  (interp_method == NEAREST){
      if (grid->zdim==1){
        err = spatial_interpolation_nearest2D(xsi, eta, xi[igrid], yi[igrid], grid->xdim, (float**)(data[ti[igrid]]), &f0);
        err = spatial_interpolation_nearest2D(xsi, eta, xi[igrid], yi[igrid], grid->xdim, (float**)(data[ti[igrid]+1]), &f1);
      } else {
        err = spatial_interpolation_nearest3D(xsi, eta, zeta, xi[igrid], yi[igrid], zi[igrid], grid->xdim, grid->ydim,
                                              (float**)(data[ti[igrid]]), &f0);
        err = spatial_interpolation_nearest3D(xsi, eta, zeta, xi[igrid], yi[igrid], zi[igrid], grid->xdim, grid->ydim,
                                              (float**)(data[ti[igrid]+1]), &f1);
      }
    }
    else {
        return ERROR;
    }
    *value = f0 + (f1 - f0) * (float)((time - t0) / (t1 - t0));
    return SUCCESS;
  } else {
    double t0 = grid->time[ti[igrid]];
    err = search_indices(x, y, z, grid->xdim, grid->ydim, grid->zdim, grid->lon, grid->lat, grid->depth, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zeta, grid->sphere_mesh, grid->zonal_periodic, gcode, grid->z4d, ti[igrid], grid->tdim, t0, t0, t0+1); CHECKERROR(err);
    if (interp_method == LINEAR){
      if (grid->zdim==1)
        err = spatial_interpolation_bilinear(xsi, eta, xi[igrid], yi[igrid], grid->xdim, (float**)(data[ti[igrid]]), value);
      else
        err = spatial_interpolation_trilinear(xsi, eta, zeta, xi[igrid], yi[igrid], zi[igrid], grid->xdim, grid->ydim,
                                             (float**)(data[ti[igrid]]), value);
    }
    else if (interp_method == NEAREST){
      if (grid->zdim==1)
        err = spatial_interpolation_nearest2D(xsi, eta, xi[igrid], yi[igrid], grid->xdim, (float**)(data[ti[igrid]]), value);
      else {
        err = spatial_interpolation_nearest3D(xsi, eta, zeta, xi[igrid], yi[igrid], zi[igrid], grid->xdim, grid->ydim,
                                             (float**)(data[ti[igrid]]), value);
      }
    }
    else {
        return ERROR;    
    }
    return SUCCESS;
  }
}

static inline ErrorCode temporal_interpolation(float x, float y, float z, double time, CField *f, 
                                                void * vxi,  void * vyi,  void * vzi,  void * vti, float *value, int interp_method)
{
  CGrid *_grid = f->grid;
  GridCode gcode = _grid->gtype;
  int *xi = (int *) vxi;
  int *yi = (int *) vyi;
  int *zi = (int *) vzi;
  int *ti = (int *) vti;

  if (gcode == RECTILINEAR_Z_GRID || gcode == RECTILINEAR_S_GRID || gcode == CURVILINEAR_Z_GRID || gcode == CURVILINEAR_S_GRID)
    return temporal_interpolation_structured_grid(x, y, z, time, f, gcode, xi, yi, zi, ti, value, interp_method);
  else{
    printf("Only RECTILINEAR_Z_GRID, RECTILINEAR_S_GRID, CURVILINEAR_Z_GRID and CURVILINEAR_S_GRID grids are currently implemented\n");
    return ERROR;
  }
}

static inline ErrorCode temporal_interpolationUV(float x, float y, float z, double time,
                                                 CField *U, CField *V,  void * xi,  void * yi,  void * zi,  void * ti,
                                                 float *valueU, float *valueV, int interp_method)
{
  ErrorCode err;

  err = temporal_interpolation(x, y, z, time, U, xi, yi, zi, ti, valueU, interp_method); CHECKERROR(err);
  err = temporal_interpolation(x, y, z, time, V, xi, yi, zi, ti, valueV, interp_method); CHECKERROR(err);

  return SUCCESS;
}

static inline ErrorCode temporal_interpolationUVrotation(float x, float y, float z, double time,
                                                 CField *U, CField *V, CField *cosU, CField *sinU, CField *cosV, CField *sinV,
                                                  void * xi,  void * yi,  void * zi,  void * ti, float *valueU, float *valueV, int interp_method)
{
  ErrorCode err;

  float u_val, v_val, cosU_val, sinU_val, cosV_val, sinV_val;
  err = temporal_interpolation(x, y, z, time, U, xi, yi, zi, ti, &u_val, interp_method); CHECKERROR(err);
  err = temporal_interpolation(x, y, z, time, V, xi, yi, zi, ti, &v_val, interp_method); CHECKERROR(err);
  err = temporal_interpolation(x, y, z, time, cosU, xi, yi, zi, ti, &cosU_val, interp_method); CHECKERROR(err);
  err = temporal_interpolation(x, y, z, time, sinU, xi, yi, zi, ti, &sinU_val, interp_method); CHECKERROR(err);
  err = temporal_interpolation(x, y, z, time, cosV, xi, yi, zi, ti, &cosV_val, interp_method); CHECKERROR(err);
  err = temporal_interpolation(x, y, z, time, sinV, xi, yi, zi, ti, &sinV_val, interp_method); CHECKERROR(err);

  *valueU = u_val * cosU_val - v_val * sinV_val;
  *valueV = u_val * sinU_val + v_val * cosV_val;

  return SUCCESS;
}


#ifdef __cplusplus
}
#endif
#endif
