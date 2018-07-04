#ifndef _PARCELS_H
#define _PARCELS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "random.h"
#include "index_search.h"

typedef enum
  {
    LINEAR=0, NEAREST=1, CGRID_LINEAR=2
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

static double lonlatdist(float lon1, float lon2, float lat1, float lat2)
{
  double r = 360*60*1852 / (2*M_PI);
  double rad = M_PI / 180.;
  double dist;
  int planar = 1;
  if (planar == 0){
    double dlat = rad * (lat2-lat1);
    double dlon = rad * (lon2-lon1);
    double a = sin(dlat/2)*sin(dlat/2) + cos(rad*lat1) * cos(rad*lat2) * sin(dlon/2) * sin(dlon/2);
    dist = r * 2 * atan2(sqrt(a),sqrt(1-a));
  }
  else{
    double x1 = r*cos(rad*lon1) * cos(rad*lat1);
    double y1 = r*sin(rad*lon1) * cos(rad*lat1);
    double z1 = r*sin(rad*lat1);
    double x2 = r*cos(rad*lon2) * cos(rad*lat2);
    double y2 = r*sin(rad*lon2) * cos(rad*lat2);
    double z2 = r*sin(rad*lat2);
    dist = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1));
  }
  return dist;
}

/* Linear interpolation routine for 2D C grid */
static inline ErrorCode spatial_interpolation_UV_c_grid(double xsi, double eta, int xi, int yi, CStructuredGrid *grid, float **u_data, float **v_data, float *u, float *v)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  int xdim = grid->xdim;
  float (* xgrid)[xdim] = (float (*)[xdim]) grid->lon;
  float (* ygrid)[xdim] = (float (*)[xdim]) grid->lat;
  float (*dataU)[xdim] = (float (*)[xdim]) u_data;
  float (*dataV)[xdim] = (float (*)[xdim]) v_data;

  float xgrid_loc[4] = {xgrid[yi][xi], xgrid[yi][xi+1], xgrid[yi+1][xi+1], xgrid[yi+1][xi]};
  float ygrid_loc[4] = {ygrid[yi][xi], ygrid[yi][xi+1], ygrid[yi+1][xi+1], ygrid[yi+1][xi]};
  int i4;
  for (i4 = 1; i4 < 4; ++i4){
    if (xgrid_loc[i4] < xgrid_loc[0] - 180) xgrid_loc[i4] += 360;
    if (xgrid_loc[i4] > xgrid_loc[0] + 180) xgrid_loc[i4] -= 360;
  }

  double U0 = dataU[yi+1][xi]   * lonlatdist(xgrid_loc[3], xgrid_loc[0], ygrid_loc[3], ygrid_loc[0]);
  double U1 = dataU[yi+1][xi+1] * lonlatdist(xgrid_loc[1], xgrid_loc[2], ygrid_loc[1], ygrid_loc[2]);
  double V0 = dataV[yi][xi+1]   * lonlatdist(xgrid_loc[0], xgrid_loc[1], ygrid_loc[0], ygrid_loc[1]);
  double V1 = dataV[yi+1][xi+1] * lonlatdist(xgrid_loc[2], xgrid_loc[3], ygrid_loc[2], ygrid_loc[3]);
  double U = (1-xsi) * U0 + xsi * U1;
  double V = (1-eta) * V0 + eta * V1;

  double dphidxsi[4] = {eta-1, 1-eta, eta, -eta};
  double dphideta[4] = {xsi-1, -xsi, xsi, 1-xsi};
  double dxdxsi = 0; double dxdeta = 0;
  double dydxsi = 0; double dydeta = 0;
  int i;
  for(i=0; i<4; ++i){
    dxdxsi += xgrid_loc[i] *dphidxsi[i];
    dxdeta += xgrid_loc[i] *dphideta[i];
    dydxsi += ygrid_loc[i] *dphidxsi[i];
    dydeta += ygrid_loc[i] *dphideta[i];
  }
  double deg2m = 1852 * 60.;
  double rad = M_PI / 180.;
  double lat = (1-xsi) * (1-eta) * ygrid_loc[0]+
                  xsi  * (1-eta) * ygrid_loc[1]+
                  xsi  *    eta  * ygrid_loc[2]+
               (1-xsi) *    eta  * ygrid_loc[3];
  double jac = (dxdxsi*dydeta - dxdeta * dydxsi) * deg2m * deg2m * cos(rad * lat);

  *u = ( (-(1-eta) * U - (1-xsi) * V ) * xgrid_loc[0] +
         ( (1-eta) * U -  xsi    * V ) * xgrid_loc[1] +
         (    eta  * U +  xsi    * V ) * xgrid_loc[2] +
         (   -eta  * U + (1-xsi) * V ) * xgrid_loc[3] ) / jac;
  *v = ( (-(1-eta) * U - (1-xsi) * V ) * ygrid_loc[0] +
         ( (1-eta) * U -  xsi    * V ) * ygrid_loc[1] +
         (    eta  * U +  xsi    * V ) * ygrid_loc[2] +
         (   -eta  * U + (1-xsi) * V ) * ygrid_loc[3] ) / jac;

  return SUCCESS;
}



static inline ErrorCode temporal_interpolation_UV_c_grid(float x, float y, float z, double time, CField *U, CField *V,
                                                         GridCode gcode, int *xi, int *yi, int *zi, int *ti,
                                                         float *u, float *v)
{
  ErrorCode err;
  CStructuredGrid *grid = U->grid->grid;
  int igrid = U->igrid;

  /* Find time index for temporal interpolation */
  if (U->time_periodic == 0 && U->allow_time_extrapolation == 0 && (time < grid->time[0] || time > grid->time[grid->tdim-1])){
    return ERROR_TIME_EXTRAPOLATION;
  }
  err = search_time_index(&time, grid->tdim, grid->time, &ti[igrid], U->time_periodic);

  /* Cast data array intp data[time][depth][lat][lon] as per NEMO convention */
  float (*dataU)[U->zdim][U->ydim][U->xdim] = (float (*)[U->zdim][U->ydim][U->xdim]) U->data;
  float (*dataV)[V->zdim][V->ydim][V->xdim] = (float (*)[V->zdim][V->ydim][V->xdim]) V->data;
  double xsi, eta, zeta;


  if (ti[igrid] < grid->tdim-1 && time > grid->time[ti[igrid]]) {
    float u0, u1, v0, v1;
    double t0 = grid->time[ti[igrid]]; double t1 = grid->time[ti[igrid]+1];
    /* Identify grid cell to sample through local linear search */
    err = search_indices(x, y, z, grid->xdim, grid->ydim, grid->zdim, grid->lon, grid->lat, grid->depth, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zeta, grid->sphere_mesh, grid->zonal_periodic, gcode, grid->z4d, ti[igrid], grid->tdim, time, t0, t1); CHECKERROR(err);
    if (grid->zdim==1){
      err = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, (float**)(dataU[ti[igrid]])  , (float**)(dataV[ti[igrid]]),   &u0, &v0);
      err = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, (float**)(dataU[ti[igrid]+1]), (float**)(dataV[ti[igrid]+1]), &u1, &v1);
    } else {
      err = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, (float**)(dataU[ti[igrid]][zi[igrid]])  , (float**)(dataV[ti[igrid]][zi[igrid]]),   &u0, &v0);
      err = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, (float**)(dataU[ti[igrid]+1][zi[igrid]]), (float**)(dataV[ti[igrid]+1][zi[igrid]]), &u1, &v1);
    }
    *u = u0 + (u1 - u0) * (float)((time - t0) / (t1 - t0));
    *v = v0 + (v1 - v0) * (float)((time - t0) / (t1 - t0));
    return SUCCESS;
  } else {
    double t0 = grid->time[ti[igrid]];
    err = search_indices(x, y, z, grid->xdim, grid->ydim, grid->zdim, grid->lon, grid->lat, grid->depth, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zeta, grid->sphere_mesh, grid->zonal_periodic, gcode, grid->z4d, ti[igrid], grid->tdim, t0, t0, t0+1); CHECKERROR(err);
    if (grid->zdim==1){
      err = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, (float**)(dataU[ti[igrid]][zi[igrid]])  , (float**)(dataV[ti[igrid]]), u, v);
    }
    else{
      err = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, (float**)(dataU[ti[igrid]][zi[igrid]])  , (float**)(dataV[ti[igrid]][zi[igrid]]), &u, &v);
    }
    return SUCCESS;
  }
}


static inline ErrorCode temporal_interpolation(float x, float y, float z, double time, CField *f, 
                                               void *vxi, void *vyi, void *vzi, void *vti,
                                               float *value, int interp_method)
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
                                                 CField *U, CField *V,
                                                 void *vxi, void *vyi, void *vzi, void *vti,
                                                 float *valueU, float *valueV, int interp_method)
{
  ErrorCode err;
  if (interp_method == CGRID_LINEAR){
     CGrid *_grid = U->grid;
     GridCode gcode = _grid->gtype;
     int *xi = (int *) vxi;
     int *yi = (int *) vyi;
     int *zi = (int *) vzi;
     int *ti = (int *) vti;
     err = temporal_interpolation_UV_c_grid(x, y, z, time, U, V, gcode, xi, yi, zi, ti, valueU, valueV); CHECKERROR(err);
     return SUCCESS;
  }
  else{
     err = temporal_interpolation(x, y, z, time, U, vxi, vyi, vzi, vti, valueU, interp_method); CHECKERROR(err);
     err = temporal_interpolation(x, y, z, time, V, vxi, vyi, vzi, vti, valueV, interp_method); CHECKERROR(err);
     return SUCCESS;
  }
}

static inline ErrorCode temporal_interpolationUVW(float x, float y, float z, double time,
                                                  CField *U, CField *V, CField *W,
                                                  void *vxi, void *vyi, void *vzi, void *vti,
                                                  float *valueU, float *valueV, float *valueW, int interp_method)
{
  ErrorCode err;
  if (interp_method == CGRID_LINEAR){
    return ERROR;
  }
  else{
    temporal_interpolationUV(x, y, z, time, U, V, vxi, vyi, vzi, vti, valueU, valueV, interp_method);
    err = temporal_interpolation(x, y, z, time, W, vxi, vyi, vzi, vti, valueW, interp_method); CHECKERROR(err);
    return SUCCESS;
  }
}



#ifdef __cplusplus
}
#endif
#endif
