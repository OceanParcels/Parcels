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
#include "interpolation_utils.h"

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

typedef struct
{
  int xdim, ydim, zdim, tdim, igrid, allow_time_extrapolation, time_periodic;
  float ****data_chunks;
  CGrid *grid;
} CField;

/* Bilinear interpolation routine for 2D grid */
static inline ErrorCode spatial_interpolation_bilinear(double xsi, double eta, float data[2][2], float *value)
{
  *value = (1-xsi)*(1-eta) * data[0][0]
         +    xsi *(1-eta) * data[0][1]
         +    xsi *   eta  * data[1][1]
         + (1-xsi)*   eta  * data[1][0];
  return SUCCESS;
}

/* Trilinear interpolation routine for 3D grid */
static inline ErrorCode spatial_interpolation_trilinear(double xsi, double eta, double zeta,
                                                        float data[2][2][2], float *value)
{
  float f0, f1;
  f0 = (1-xsi)*(1-eta) * data[0][0][0]
     +    xsi *(1-eta) * data[0][0][1]
     +    xsi *   eta  * data[0][1][1]
     + (1-xsi)*   eta  * data[0][1][0];
  f1 = (1-xsi)*(1-eta) * data[1][0][0]
     +    xsi *(1-eta) * data[1][0][1]
     +    xsi *   eta  * data[1][1][1]
     + (1-xsi)*   eta  * data[1][1][0];
  *value = (1-zeta) * f0 + zeta * f1;
  return SUCCESS;
}

/* Nearest neighbour interpolation routine for 2D grid */
static inline ErrorCode spatial_interpolation_nearest2D(double xsi, double eta,
                                                        float data[2][2], float *value)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  int i, j;
  if (xsi < .5) {i = 0;} else {i = 1;}
  if (eta < .5) {j = 0;} else {j = 1;}
  *value = data[j][i];
  return SUCCESS;
}

/* C grid interpolation routine for tracers on 2D grid */
static inline ErrorCode spatial_interpolation_tracer_c_grid_2D(float data[2][2], float *value)
{
  *value = data[1][1];
  return SUCCESS;
}

/* Nearest neighbour interpolation routine for 3D grid */
static inline ErrorCode spatial_interpolation_nearest3D(double xsi, double eta, double zeta,
                                                        float data[2][2][2], float *value)
{
  int i, j, k;
  if (xsi < .5) {i = 0;} else {i = 1;}
  if (eta < .5) {j = 0;} else {j = 1;}
  if (zeta < .5) {k = 0;} else {k = 1;}
  *value = data[k][j][i];
  return SUCCESS;
}

/* C grid interpolation routine for tracers on 3D grid */
static inline ErrorCode spatial_interpolation_tracer_c_grid_3D(float data[2][2][2], float *value)
{
  *value = data[0][1][1];
  return SUCCESS;
}

static inline int getBlock2D(int *chunk_info, int yi, int xi, int *block, int *index_local)
{
  int ndim = chunk_info[0];
  if (ndim != 2)
    exit(-1);
  int i, j;

  int shape[ndim];
  int index[2] = {yi, xi};
  for(i=0; i<ndim; ++i){
    int chunk_sum = 0, shift = 0;
    for (j = 0; j < i; j++) shift += chunk_info[1+j];
    shape[i] = chunk_info[1+i];
    for (j=0; j<shape[i]; j++) {
      chunk_sum += chunk_info[1+ndim+shift+j];
      if (index[i] < chunk_sum) {
        chunk_sum -= chunk_info[1+ndim+shift+j];
        break;
      }
    }
    block[i] = j;
    index_local[i] = index[i] - chunk_sum;
  }

  int bid =  block[0]*shape[1] +
             block[1];
  return bid;
}

static inline ErrorCode getCell2D(CField *f, int xi, int yi, int ti, float cell_data[2][2][2], int first_tstep_only)
{
  CStructuredGrid *grid = f->grid->grid;
  int *chunk_info = grid->chunk_info;
  int ndim = chunk_info[0];
  int block[ndim];
  int ilocal[ndim];

  int tii, yii, xii;

  int blockid = getBlock2D(chunk_info, yi, xi, block, ilocal);
  if (grid->load_chunk[blockid] < 2){
    grid->load_chunk[blockid] = 1;
    return REPEAT;
  }
  grid->load_chunk[blockid] = 2;
  int zdim = 1;
  int ydim = chunk_info[1+ndim+block[0]];
  int yshift = chunk_info[1];
  int xdim = chunk_info[1+ndim+yshift+block[1]];

  if ((ilocal[0] == ydim-1) || (ilocal[1] == xdim-1))
  {
    // Cell is on multiple chunks
    for (tii=0; tii<2; ++tii){
      for (yii=0; yii<2; ++yii){
        for (xii=0; xii<2; ++xii){
          blockid = getBlock2D(chunk_info, yi+yii, xi+xii, block, ilocal);
          if (grid->load_chunk[blockid] < 2){
            grid->load_chunk[blockid] = 1;
            return REPEAT;
          }
          grid->load_chunk[blockid] = 2;
          zdim = 1;
          ydim = chunk_info[1+ndim+block[0]];
          yshift = chunk_info[1];
          xdim = chunk_info[1+ndim+yshift+block[1]];
          float (*data_block)[zdim][ydim][xdim] = (float (*)[zdim][ydim][xdim]) f->data_chunks[blockid];
          float (*data)[xdim] = (float (*)[xdim]) (data_block[ti+tii]);
          cell_data[tii][yii][xii] = data[ilocal[0]][ilocal[1]];
        }
      }
      if (first_tstep_only == 1)
         break;
    }
  }
  else
  {
    float (*data_block)[zdim][ydim][xdim] = (float (*)[zdim][ydim][xdim]) f->data_chunks[blockid];
    for (tii=0; tii<2; ++tii){
      float (*data)[xdim] = (float (*)[xdim]) (data_block[ti+tii]);
      for (yii=0; yii<2; ++yii)
        for (xii=0; xii<2; ++xii)
          cell_data[tii][yii][xii] = data[ilocal[0]+yii][ilocal[1]+xii];
      if (first_tstep_only == 1)
         break;
    }
  }
  return SUCCESS;
}

static inline int getBlock3D(int *chunk_info, int zi, int yi, int xi, int *block, int *index_local)
{
  int ndim = chunk_info[0];
  if (ndim != 3)
    exit(-1);
  int i, j;

  int shape[ndim];
  int index[3] = {zi, yi, xi};
  for(i=0; i<ndim; ++i){
    int chunk_sum = 0, shift = 0;
    for (j = 0; j < i; j++) shift += chunk_info[1+j];
    shape[i] = chunk_info[1+i];
    for (j=0; j<shape[i]; j++) {
      chunk_sum += chunk_info[1+ndim+shift+j];
      if (index[i] < chunk_sum) {
        chunk_sum -= chunk_info[1+ndim+shift+j];
        break;
      }
    }
    block[i] = j;
    index_local[i] = index[i] - chunk_sum;
  }

  int bid =  block[0]*shape[1]*shape[2] +
             block[1]*shape[2] +
             block[2];
  return bid;
}

static inline ErrorCode getCell3D(CField *f, int xi, int yi, int zi, int ti, float cell_data[2][2][2][2], int first_tstep_only)
{
  CStructuredGrid *grid = f->grid->grid;
  int *chunk_info = grid->chunk_info;
  int ndim = chunk_info[0];
  int block[ndim];
  int ilocal[ndim];

  int tii, zii, yii, xii;

  int blockid = getBlock3D(chunk_info, zi, yi, xi, block, ilocal);
  if (grid->load_chunk[blockid] < 2){
    grid->load_chunk[blockid] = 1;
    return REPEAT;
  }
  grid->load_chunk[blockid] = 2;
  int zdim = chunk_info[1+ndim+block[0]];
  int zshift = chunk_info[1];
  int ydim = chunk_info[1+ndim+zshift+block[1]];
  int yshift = chunk_info[1+1];
  int xdim = chunk_info[1+ndim+zshift+yshift+block[2]];

  if ((ilocal[0] == zdim-1) || (ilocal[1] == ydim-1) || (ilocal[2] == xdim-1))
  {
    // Cell is on multiple chunks\n
    for (tii=0; tii<2; ++tii){
      for (zii=0; zii<2; ++zii){
        for (yii=0; yii<2; ++yii){
          for (xii=0; xii<2; ++xii){
            blockid = getBlock3D(chunk_info, zi+zii, yi+yii, xi+xii, block, ilocal);
            if (grid->load_chunk[blockid] < 2){
              grid->load_chunk[blockid] = 1;
              return REPEAT;
            }
            grid->load_chunk[blockid] = 2;
            zdim = chunk_info[1+ndim+block[0]];
            zshift = chunk_info[1];
            ydim = chunk_info[1+ndim+zshift+block[1]];
            yshift = chunk_info[1+1];
            xdim = chunk_info[1+ndim+zshift+yshift+block[2]];
            float (*data_block)[zdim][ydim][xdim] = (float (*)[zdim][ydim][xdim]) f->data_chunks[blockid];
            float (*data)[ydim][xdim] = (float (*)[ydim][xdim]) (data_block[ti+tii]);
            cell_data[tii][zii][yii][xii] = data[ilocal[0]][ilocal[1]][ilocal[2]];
          }
        }
      }
      if (first_tstep_only == 1)
         break;
    }
  }
  else
  {
    float (*data_block)[zdim][ydim][xdim] = (float (*)[zdim][ydim][xdim]) f->data_chunks[blockid];
    for (tii=0; tii<2; ++tii){
      float (*data)[ydim][xdim] = (float (*)[ydim][xdim]) (data_block[ti+tii]);
      for (zii=0; zii<2; ++zii)
        for (yii=0; yii<2; ++yii)
          for (xii=0; xii<2; ++xii)
            cell_data[tii][zii][yii][xii] = data[ilocal[0]+zii][ilocal[1]+yii][ilocal[2]+xii];
      if (first_tstep_only == 1)
         break;
    }
  }
  return SUCCESS;
}


/* Linear interpolation along the time axis */
static inline ErrorCode temporal_interpolation_structured_grid(type_coord x, type_coord y, type_coord z, double time, CField *f,
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
  err = search_time_index(&time, grid->tdim, grid->time, &ti[igrid], f->time_periodic, grid->tfull_min, grid->tfull_max, grid->periods); CHECKERROR(err);

  double xsi, eta, zeta;

  float data2D[2][2][2];
  float data3D[2][2][2][2];


  if (ti[igrid] < grid->tdim-1 && time > grid->time[ti[igrid]]) {
    float f0, f1;
    double t0 = grid->time[ti[igrid]]; double t1 = grid->time[ti[igrid]+1];
    /* Identify grid cell to sample through local linear search */
    err = search_indices(x, y, z, grid, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zeta, gcode, ti[igrid], time, t0, t1, interp_method); CHECKERROR(err);
    if (grid->zdim==1){
      err = getCell2D(f, xi[igrid], yi[igrid], ti[igrid], data2D, 0); CHECKERROR(err);
    } else{
      err = getCell3D(f, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D, 0); CHECKERROR(err);
    }
    if ((interp_method == LINEAR) || (interp_method == CGRID_VELOCITY) || (interp_method == BGRID_VELOCITY) || (interp_method == BGRID_W_VELOCITY)){
      if ((interp_method == CGRID_VELOCITY) || (interp_method == BGRID_W_VELOCITY)){ // interpolate w
        xsi = 1;
        eta = 1;
      }
      else if (interp_method == BGRID_VELOCITY){
          zeta = 0;
      }
      if (grid->zdim==1){
        err = spatial_interpolation_bilinear(xsi, eta, data2D[0], &f0); CHECKERROR(err);
        err = spatial_interpolation_bilinear(xsi, eta, data2D[1], &f1); CHECKERROR(err);
      } else {
        err = spatial_interpolation_trilinear(xsi, eta, zeta, data3D[0], &f0); CHECKERROR(err);
        err = spatial_interpolation_trilinear(xsi, eta, zeta, data3D[1], &f1); CHECKERROR(err);
      }
    }
    else if  (interp_method == NEAREST){
      if (grid->zdim==1){
        err = spatial_interpolation_nearest2D(xsi, eta, data2D[0], &f0); CHECKERROR(err);
        err = spatial_interpolation_nearest2D(xsi, eta, data2D[1], &f1); CHECKERROR(err);
      } else {
        err = spatial_interpolation_nearest3D(xsi, eta, zeta, data3D[0], &f0); CHECKERROR(err);
        err = spatial_interpolation_nearest3D(xsi, eta, zeta, data3D[1], &f1); CHECKERROR(err);
      }
    }
    else if  ((interp_method == CGRID_TRACER) || (interp_method == BGRID_TRACER)){
      if (grid->zdim==1){
        err = spatial_interpolation_tracer_c_grid_2D(data2D[0], &f0); CHECKERROR(err);
        err = spatial_interpolation_tracer_c_grid_2D(data2D[1], &f1); CHECKERROR(err);
      } else {
        err = spatial_interpolation_tracer_c_grid_3D(data3D[0], &f0); CHECKERROR(err);
        err = spatial_interpolation_tracer_c_grid_3D(data3D[1], &f1); CHECKERROR(err);
      }
    }
    else {
        return ERROR;
    }
    *value = f0 + (f1 - f0) * (float)((time - t0) / (t1 - t0));
    return SUCCESS;
  } else {
    double t0 = grid->time[ti[igrid]];
    err = search_indices(x, y, z, grid, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zeta, gcode, ti[igrid], t0, t0, t0+1, interp_method); CHECKERROR(err);
    if (grid->zdim==1){
      err = getCell2D(f, xi[igrid], yi[igrid], ti[igrid], data2D, 1); CHECKERROR(err);
    } else{
      err = getCell3D(f, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D, 1); CHECKERROR(err);
    }
    if ((interp_method == LINEAR) || (interp_method == CGRID_VELOCITY) || (interp_method == BGRID_VELOCITY) ||(interp_method == BGRID_W_VELOCITY)){
      if ((interp_method == CGRID_VELOCITY) || (interp_method == BGRID_W_VELOCITY)){ // interpolate w
        xsi = 1;
        eta = 1;
        if (grid->zdim==1)
          return ERROR;
      }
      else if (interp_method == BGRID_VELOCITY){
        zeta = 0;
      }    
      if (grid->zdim==1){
        err = spatial_interpolation_bilinear(xsi, eta, data2D[0], value); CHECKERROR(err);
      }
      else{
        err = spatial_interpolation_trilinear(xsi, eta, zeta, data3D[0], value); CHECKERROR(err);
      }
    }
    else if (interp_method == NEAREST){
      if (grid->zdim==1){
        err = spatial_interpolation_nearest2D(xsi, eta, data2D[0], value); CHECKERROR(err);
      }
      else {
        err = spatial_interpolation_nearest3D(xsi, eta, zeta, data3D[0], value); CHECKERROR(err);
      }
    }
    else if ((interp_method == CGRID_TRACER) || (interp_method == BGRID_TRACER)){
      if (grid->zdim==1){
        err = spatial_interpolation_tracer_c_grid_2D(data2D[0], value); CHECKERROR(err);
      }
      else {
        err = spatial_interpolation_tracer_c_grid_3D(data3D[0], value); CHECKERROR(err);
      }
    }
    else {
        return ERROR;    
    }
    return SUCCESS;
  }
}

static double dist(double lon1, double lon2, double lat1, double lat2, int sphere_mesh, double lat)
{
  if (sphere_mesh == 1){
    double rad = M_PI / 180.;
    double deg2m = 1852 * 60.;
    return sqrt((lon2-lon1)*(lon2-lon1) * deg2m * deg2m * cos(rad * lat) * cos(rad * lat) + (lat2-lat1)*(lat2-lat1) * deg2m * deg2m);
  }
  else{
    return sqrt((lon2-lon1)*(lon2-lon1) + (lat2-lat1)*(lat2-lat1));
  }
}

/* Linear interpolation routine for 2D C grid */
static inline ErrorCode spatial_interpolation_UV_c_grid(double xsi, double eta, int xi, int yi, CStructuredGrid *grid,
                                                        GridCode gcode, float dataU[2][2], float dataV[2][2], float *u, float *v)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  int xdim = grid->xdim;

  double xgrid_loc[4];
  double ygrid_loc[4];
  int iN;
  if( (gcode == RECTILINEAR_Z_GRID) || (gcode == RECTILINEAR_S_GRID) ){
    float *xgrid = grid->lon;
    float *ygrid = grid->lat;
    for (iN=0; iN < 4; ++iN){
      xgrid_loc[iN] = xgrid[xi+min(1, (iN%3))];
      ygrid_loc[iN] = ygrid[yi+iN/2];
    }
  }
  else{
    float (* xgrid)[xdim] = (float (*)[xdim]) grid->lon;
    float (* ygrid)[xdim] = (float (*)[xdim]) grid->lat;
    for (iN=0; iN < 4; ++iN){
      xgrid_loc[iN] = xgrid[yi+iN/2][xi+min(1, (iN%3))];
      ygrid_loc[iN] = ygrid[yi+iN/2][xi+min(1, (iN%3))];
    }
  }
  int i4;
  for (i4 = 1; i4 < 4; ++i4){
    if (xgrid_loc[i4] < xgrid_loc[0] - 180) xgrid_loc[i4] += 360;
    if (xgrid_loc[i4] > xgrid_loc[0] + 180) xgrid_loc[i4] -= 360;
  }


  double phi[4];
  phi2D_lin(0., eta, phi);
  double U0 = dataU[1][0] * dist(xgrid_loc[3], xgrid_loc[0], ygrid_loc[3], ygrid_loc[0], grid->sphere_mesh, dot_prod(phi, ygrid_loc, 4));
  phi2D_lin(1., eta, phi);
  double U1 = dataU[1][1] * dist(xgrid_loc[1], xgrid_loc[2], ygrid_loc[1], ygrid_loc[2], grid->sphere_mesh, dot_prod(phi, ygrid_loc, 4));
  phi2D_lin(xsi, 0., phi);
  double V0 = dataV[0][1] * dist(xgrid_loc[0], xgrid_loc[1], ygrid_loc[0], ygrid_loc[1], grid->sphere_mesh, dot_prod(phi, ygrid_loc, 4));
  phi2D_lin(xsi, 1., phi);
  double V1 = dataV[1][1] * dist(xgrid_loc[2], xgrid_loc[3], ygrid_loc[2], ygrid_loc[3], grid->sphere_mesh, dot_prod(phi, ygrid_loc, 4));
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
  double meshJac = 1;
  if (grid->sphere_mesh == 1){
    double deg2m = 1852 * 60.;
    double rad = M_PI / 180.;
    phi2D_lin(xsi, eta, phi);
    double lat = dot_prod(phi, ygrid_loc, 4);
    meshJac = deg2m * deg2m * cos(rad * lat);
  }
  double jac = (dxdxsi*dydeta - dxdeta * dydxsi) * meshJac;

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



static inline ErrorCode temporal_interpolationUV_c_grid(type_coord x, type_coord y, type_coord z, double time, CField *U, CField *V,
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
  err = search_time_index(&time, grid->tdim, grid->time, &ti[igrid], U->time_periodic, grid->tfull_min, grid->tfull_max, grid->periods); CHECKERROR(err);

  double xsi, eta, zeta;


  if (ti[igrid] < grid->tdim-1 && time > grid->time[ti[igrid]]) {
    float u0, u1, v0, v1;
    double t0 = grid->time[ti[igrid]]; double t1 = grid->time[ti[igrid]+1];
    /* Identify grid cell to sample through local linear search */
    err = search_indices(x, y, z, grid, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zeta, gcode, ti[igrid], time, t0, t1, CGRID_VELOCITY); CHECKERROR(err);
    if (grid->zdim==1){
      float data2D_U[2][2][2], data2D_V[2][2][2];
      err = getCell2D(U, xi[igrid], yi[igrid], ti[igrid], data2D_U, 0); CHECKERROR(err);
      err = getCell2D(V, xi[igrid], yi[igrid], ti[igrid], data2D_V, 0); CHECKERROR(err);
      err = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, gcode, data2D_U[0], data2D_V[0], &u0, &v0); CHECKERROR(err);
      err = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, gcode, data2D_U[1], data2D_V[1], &u1, &v1); CHECKERROR(err);

    } else {
      float data3D_U[2][2][2][2], data3D_V[2][2][2][2];
      err = getCell3D(U, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_U, 0); CHECKERROR(err);
      err = getCell3D(V, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_V, 0); CHECKERROR(err);
      err = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, gcode, data3D_U[0][0], data3D_V[0][0], &u0, &v0); CHECKERROR(err);
      err = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, gcode, data3D_U[1][0], data3D_V[1][0], &u1, &v1); CHECKERROR(err);
    }
    *u = u0 + (u1 - u0) * (float)((time - t0) / (t1 - t0));
    *v = v0 + (v1 - v0) * (float)((time - t0) / (t1 - t0));
    return SUCCESS;
  } else {
    double t0 = grid->time[ti[igrid]];
    err = search_indices(x, y, z, grid, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zeta, gcode, ti[igrid], t0, t0, t0+1, CGRID_VELOCITY); CHECKERROR(err);
    if (grid->zdim==1){
      float data2D_U[2][2][2], data2D_V[2][2][2];
      err = getCell2D(U, xi[igrid], yi[igrid], ti[igrid], data2D_U, 1); CHECKERROR(err);
      err = getCell2D(V, xi[igrid], yi[igrid], ti[igrid], data2D_V, 1); CHECKERROR(err);
      err = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, gcode, data2D_U[0], data2D_V[0], u, v); CHECKERROR(err);
    }
    else{
      float data3D_U[2][2][2][2], data3D_V[2][2][2][2];
      err = getCell3D(U, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_U, 1); CHECKERROR(err);
      err = getCell3D(V, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_V, 1); CHECKERROR(err);
      err = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, gcode, data3D_U[0][0], data3D_V[0][0], u, v); CHECKERROR(err);
    }
    return SUCCESS;
  }
}

/* Quadratic interpolation routine for 3D C grid */
static inline ErrorCode spatial_interpolation_UVW_c_grid(double xsi, double eta, double zet, int xi, int yi, int zi, int ti, CStructuredGrid *grid,
                                                        GridCode gcode, float dataU[2][2][2], float dataV[2][2][2], float dataW[2][2][2], float *u, float *v, float *w)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  int xdim = grid->xdim;
  int ydim = grid->ydim;
  int zdim = grid->zdim;

  float xgrid_loc[4];
  float ygrid_loc[4];
  int iN;
  if( gcode == RECTILINEAR_S_GRID ){
    float *xgrid = grid->lon;
    float *ygrid = grid->lat;
    for (iN=0; iN < 4; ++iN){
      xgrid_loc[iN] = xgrid[xi+min(1, (iN%3))];
      ygrid_loc[iN] = ygrid[yi+iN/2];
    }
  }
  else{
    float (* xgrid)[xdim] = (float (*)[xdim]) grid->lon;
    float (* ygrid)[xdim] = (float (*)[xdim]) grid->lat;
    for (iN=0; iN < 4; ++iN){
      xgrid_loc[iN] = xgrid[yi+iN/2][xi+min(1, (iN%3))];
      ygrid_loc[iN] = ygrid[yi+iN/2][xi+min(1, (iN%3))];
    }
  }
  int i4;
  for (i4 = 1; i4 < 4; ++i4){
    if (xgrid_loc[i4] < xgrid_loc[0] - 180) xgrid_loc[i4] += 360;
    if (xgrid_loc[i4] > xgrid_loc[0] + 180) xgrid_loc[i4] -= 360;
  }

  float u0 = dataU[0][1][0];
  float u1 = dataU[0][1][1];
  float v0 = dataV[0][0][1];
  float v1 = dataV[0][1][1];
  float w0 = dataW[0][1][1];
  float w1 = dataW[1][1][1];

  double px[8] = {xgrid_loc[0], xgrid_loc[1], xgrid_loc[2], xgrid_loc[3],
                  xgrid_loc[0], xgrid_loc[1], xgrid_loc[2], xgrid_loc[3]};
  double py[8] = {ygrid_loc[0], ygrid_loc[1], ygrid_loc[2], ygrid_loc[3],
                  ygrid_loc[0], ygrid_loc[1], ygrid_loc[2], ygrid_loc[3]};
  double pz[8];
  if (grid->z4d == 1){
    float (*zvals)[zdim][ydim][xdim] = (float (*)[zdim][ydim][xdim]) grid->depth;
    for (iN=0; iN < 4; ++iN){
      pz[iN] = zvals[ti][zi][yi+iN/2][xi+min(1, (iN%3))];
      pz[iN+4] = zvals[ti][zi+1][yi+iN/2][xi+min(1, (iN%3))];
    }
  }
  else{
    float (*zvals)[ydim][xdim] = (float (*)[ydim][xdim]) grid->depth;
    for (iN=0; iN < 4; ++iN){
      pz[iN] = zvals[zi][yi+iN/2][xi+min(1, (iN%3))];
      pz[iN+4] = zvals[zi+1][yi+iN/2][xi+min(1, (iN%3))];
    }
  }

  double U0 = u0 * jacobian3D_lin_face(px, py, pz, 0, eta, zet, ZONAL, grid->sphere_mesh);
  double U1 = u1 * jacobian3D_lin_face(px, py, pz, 1, eta, zet, ZONAL, grid->sphere_mesh);
  double V0 = v0 * jacobian3D_lin_face(px, py, pz, xsi, 0, zet, MERIDIONAL, grid->sphere_mesh);
  double V1 = v1 * jacobian3D_lin_face(px, py, pz, xsi, 1, zet, MERIDIONAL, grid->sphere_mesh);
  double W0 = w0 * jacobian3D_lin_face(px, py, pz, xsi, eta, 0, VERTICAL, grid->sphere_mesh);
  double W1 = w1 * jacobian3D_lin_face(px, py, pz, xsi, eta, 1, VERTICAL, grid->sphere_mesh);

  // Computing fluxes in half left hexahedron -> flux_u05
  double xxu[8] = {px[0], (px[0]+px[1])/2, (px[2]+px[3])/2, px[3], px[4], (px[4]+px[5])/2, (px[6]+px[7])/2, px[7]};
  double yyu[8] = {py[0], (py[0]+py[1])/2, (py[2]+py[3])/2, py[3], py[4], (py[4]+py[5])/2, (py[6]+py[7])/2, py[7]};
  double zzu[8] = {pz[0], (pz[0]+pz[1])/2, (pz[2]+pz[3])/2, pz[3], pz[4], (pz[4]+pz[5])/2, (pz[6]+pz[7])/2, pz[7]};
  double flux_u0 = u0 * jacobian3D_lin_face(xxu, yyu, zzu, 0, .5, .5, ZONAL, grid->sphere_mesh);
  double flux_v0_halfx = v0 * jacobian3D_lin_face(xxu, yyu, zzu, .5, 0, .5, MERIDIONAL, grid->sphere_mesh);
  double flux_v1_halfx = v1 * jacobian3D_lin_face(xxu, yyu, zzu, .5, 1, .5, MERIDIONAL, grid->sphere_mesh);
  double flux_w0_halfx = w0 * jacobian3D_lin_face(xxu, yyu, zzu, .5, .5, 0, VERTICAL, grid->sphere_mesh);
  double flux_w1_halfx = w1 * jacobian3D_lin_face(xxu, yyu, zzu, .5, .5, 1, VERTICAL, grid->sphere_mesh);
  double flux_u05 = flux_u0 + flux_v0_halfx - flux_v1_halfx + flux_w0_halfx - flux_w1_halfx;

  // Computing fluxes in half front hexahedron -> flux_v05
  double xxv[8] = {px[0], px[1], (px[1]+px[2])/2, (px[0]+px[3])/2, px[4], px[5], (px[5]+px[6])/2, (px[4]+px[7])/2};
  double yyv[8] = {py[0], py[1], (py[1]+py[2])/2, (py[0]+py[3])/2, py[4], py[5], (py[5]+py[6])/2, (py[4]+py[7])/2};
  double zzv[8] = {pz[0], pz[1], (pz[1]+pz[2])/2, (pz[0]+pz[3])/2, pz[4], pz[5], (pz[5]+pz[6])/2, (pz[4]+pz[7])/2};
  double flux_u0_halfy = u0 * jacobian3D_lin_face(xxv, yyv, zzv, 0, .5, .5, ZONAL, grid->sphere_mesh);
  double flux_u1_halfy = u1 * jacobian3D_lin_face(xxv, yyv, zzv, 1, .5, .5, ZONAL, grid->sphere_mesh);
  double flux_v0 = v0 * jacobian3D_lin_face(xxv, yyv, zzv, .5, 0, .5, MERIDIONAL, grid->sphere_mesh);
  double flux_w0_halfy = w0 * jacobian3D_lin_face(xxv, yyv, zzv, .5, .5, 0, VERTICAL, grid->sphere_mesh);
  double flux_w1_halfy = w1 * jacobian3D_lin_face(xxv, yyv, zzv, .5, .5, 1, VERTICAL, grid->sphere_mesh);
  double flux_v05 = flux_u0_halfy - flux_u1_halfy + flux_v0 + flux_w0_halfy - flux_w1_halfy;

  // Computing fluxes in half lower hexahedron -> flux_w05
  double xx[8] = {px[0], px[1], px[2], px[3], (px[0]+px[4])/2, (px[1]+px[5])/2, (px[2]+px[6])/2, (px[3]+px[7])/2};
  double yy[8] = {py[0], py[1], py[2], py[3], (py[0]+py[4])/2, (py[1]+py[5])/2, (py[2]+py[6])/2, (py[3]+py[7])/2};
  double zz[8] = {pz[0], pz[1], pz[2], pz[3], (pz[0]+pz[4])/2, (pz[1]+pz[5])/2, (pz[2]+pz[6])/2, (pz[3]+pz[7])/2};
  double flux_u0_halfz = u0 * jacobian3D_lin_face(xx, yy, zz, 0, .5, .5, ZONAL, grid->sphere_mesh);
  double flux_u1_halfz = u1 * jacobian3D_lin_face(xx, yy, zz, 1, .5, .5, ZONAL, grid->sphere_mesh);
  double flux_v0_halfz = v0 * jacobian3D_lin_face(xx, yy, zz, .5, 0, .5, MERIDIONAL, grid->sphere_mesh);
  double flux_v1_halfz = v1 * jacobian3D_lin_face(xx, yy, zz, .5, 1, .5, MERIDIONAL, grid->sphere_mesh);
  double flux_w0 = w0 * jacobian3D_lin_face(xx, yy, zz, .5, .5, 0, VERTICAL, grid->sphere_mesh);
  double flux_w05 = flux_u0_halfz - flux_u1_halfz + flux_v0_halfz - flux_v1_halfz + flux_w0;

  double surf_u05 = jacobian3D_lin_face(px, py, pz, .5, .5, .5, ZONAL, grid->sphere_mesh);
  double jac_u05 = jacobian3D_lin_face(px, py, pz, .5, eta, zet, ZONAL, grid->sphere_mesh);
  double U05 = flux_u05 / surf_u05 * jac_u05;

  double surf_v05 = jacobian3D_lin_face(px, py, pz, .5, .5, .5, MERIDIONAL, grid->sphere_mesh);
  double jac_v05 = jacobian3D_lin_face(px, py, pz, xsi, .5, zet, MERIDIONAL, grid->sphere_mesh);
  double V05 = flux_v05 / surf_v05 * jac_v05;

  double surf_w05 = jacobian3D_lin_face(px, py, pz, .5, .5, .5, VERTICAL, grid->sphere_mesh);
  double jac_w05 = jacobian3D_lin_face(px, py, pz, xsi, eta, .5, VERTICAL, grid->sphere_mesh);
  double W05 = flux_w05 / surf_w05 * jac_w05;

  double jac = jacobian3D_lin(px, py, pz, xsi, eta, zet, grid->sphere_mesh);

  double phi[3];
  phi1D_quad(xsi, phi);
  double uvec[3] = {U0, U05, U1};
  double dxsidt = dot_prod(phi, uvec, 3) / jac;
  phi1D_quad(eta, phi);
  double vvec[3] = {V0, V05, V1};
  double detadt = dot_prod(phi, vvec, 3) / jac;
  phi1D_quad(zet, phi);
  double wvec[3] = {W0, W05, W1};
  double dzetdt = dot_prod(phi, wvec, 3) / jac;

  double dphidxsi[8], dphideta[8], dphidzet[8];
  dphidxsi3D_lin(xsi, eta, zet, dphidxsi, dphideta, dphidzet);

  *u = dot_prod(dphidxsi, px, 8) * dxsidt + dot_prod(dphideta, px, 8) * detadt + dot_prod(dphidzet, px, 8) * dzetdt;
  *v = dot_prod(dphidxsi, py, 8) * dxsidt + dot_prod(dphideta, py, 8) * detadt + dot_prod(dphidzet, py, 8) * dzetdt;
  *w = dot_prod(dphidxsi, pz, 8) * dxsidt + dot_prod(dphideta, pz, 8) * detadt + dot_prod(dphidzet, pz, 8) * dzetdt;

  return SUCCESS;
}

static inline ErrorCode temporal_interpolationUVW_c_grid(type_coord x, type_coord y, type_coord z, double time, CField *U, CField *V, CField *W,
                                                         GridCode gcode, int *xi, int *yi, int *zi, int *ti,
                                                         float *u, float *v, float *w)
{
  ErrorCode err;
  CStructuredGrid *grid = U->grid->grid;
  int igrid = U->igrid;

  /* Find time index for temporal interpolation */
  if (U->time_periodic == 0 && U->allow_time_extrapolation == 0 && (time < grid->time[0] || time > grid->time[grid->tdim-1])){
    return ERROR_TIME_EXTRAPOLATION;
  }
  err = search_time_index(&time, grid->tdim, grid->time, &ti[igrid], U->time_periodic, grid->tfull_min, grid->tfull_max, grid->periods); CHECKERROR(err);

  double xsi, eta, zet;
  float data3D_U[2][2][2][2];
  float data3D_V[2][2][2][2];
  float data3D_W[2][2][2][2];


  if (ti[igrid] < grid->tdim-1 && time > grid->time[ti[igrid]]) {
    float u0, u1, v0, v1, w0, w1;
    double t0 = grid->time[ti[igrid]]; double t1 = grid->time[ti[igrid]+1];
    /* Identify grid cell to sample through local linear search */
    err = search_indices(x, y, z, grid, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zet, gcode, ti[igrid], time, t0, t1, CGRID_VELOCITY); CHECKERROR(err);
    err = getCell3D(U, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_U, 0); CHECKERROR(err);
    err = getCell3D(V, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_V, 0); CHECKERROR(err);
    err = getCell3D(W, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_W, 0); CHECKERROR(err);
    if (grid->zdim==1){
      return ERROR;
    } else {
      err = spatial_interpolation_UVW_c_grid(xsi, eta, zet, xi[igrid], yi[igrid], zi[igrid], ti[igrid],   grid, gcode, data3D_U[0], data3D_V[0], data3D_W[0], &u0, &v0, &w0); CHECKERROR(err);
      err = spatial_interpolation_UVW_c_grid(xsi, eta, zet, xi[igrid], yi[igrid], zi[igrid], ti[igrid]+1, grid, gcode, data3D_U[1], data3D_V[1], data3D_W[1], &u1, &v1, &w1); CHECKERROR(err);
    }
    *u = u0 + (u1 - u0) * (float)((time - t0) / (t1 - t0));
    *v = v0 + (v1 - v0) * (float)((time - t0) / (t1 - t0));
    *w = w0 + (w1 - w0) * (float)((time - t0) / (t1 - t0));
    return SUCCESS;
  } else {
    double t0 = grid->time[ti[igrid]];
    err = search_indices(x, y, z, grid, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zet, gcode, ti[igrid], t0, t0, t0+1, CGRID_VELOCITY); CHECKERROR(err);
    err = getCell3D(U, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_U, 1); CHECKERROR(err);
    err = getCell3D(V, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_V, 1); CHECKERROR(err);
    err = getCell3D(W, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_W, 1); CHECKERROR(err);
    if (grid->zdim==1){
      return ERROR;
    }
    else{
      err = spatial_interpolation_UVW_c_grid(xsi, eta, zet, xi[igrid], yi[igrid], zi[igrid], ti[igrid], grid, gcode, data3D_U[0], data3D_V[0], data3D_W[0], u, v, w); CHECKERROR(err);
    }
    return SUCCESS;
  }
}


static inline ErrorCode temporal_interpolation(type_coord x, type_coord y, type_coord z, double time, CField *f,
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

static inline ErrorCode temporal_interpolationUV(type_coord x, type_coord y, type_coord z, double time,
                                                 CField *U, CField *V,
                                                 void *vxi, void *vyi, void *vzi, void *vti,
                                                 float *valueU, float *valueV, int interp_method)
{
  ErrorCode err;
  if (interp_method == CGRID_VELOCITY){
    CGrid *_grid = U->grid;
    GridCode gcode = _grid->gtype;
    int *xi = (int *) vxi;
    int *yi = (int *) vyi;
    int *zi = (int *) vzi;
    int *ti = (int *) vti;
    err = temporal_interpolationUV_c_grid(x, y, z, time, U, V, gcode, xi, yi, zi, ti, valueU, valueV); CHECKERROR(err);
    return SUCCESS;
  }
  else{
    err = temporal_interpolation(x, y, z, time, U, vxi, vyi, vzi, vti, valueU, interp_method); CHECKERROR(err);
    err = temporal_interpolation(x, y, z, time, V, vxi, vyi, vzi, vti, valueV, interp_method); CHECKERROR(err);
    return SUCCESS;
  }
}

static inline ErrorCode temporal_interpolationUVW(type_coord x, type_coord y, type_coord z, double time,
                                                  CField *U, CField *V, CField *W,
                                                  void *vxi, void *vyi, void *vzi, void *vti,
                                                  float *valueU, float *valueV, float *valueW, int interp_method)
{
  ErrorCode err;
  if (interp_method == CGRID_VELOCITY){
    CGrid *_grid = U->grid;
    GridCode gcode = _grid->gtype;
    if (gcode == RECTILINEAR_S_GRID || gcode == CURVILINEAR_S_GRID){
      int *xi = (int *) vxi;
      int *yi = (int *) vyi;
      int *zi = (int *) vzi;
      int *ti = (int *) vti;
      err = temporal_interpolationUVW_c_grid(x, y, z, time, U, V, W, gcode, xi, yi, zi, ti, valueU, valueV, valueW); CHECKERROR(err);
      return SUCCESS;
    }
  }
  err = temporal_interpolationUV(x, y, z, time, U, V, vxi, vyi, vzi, vti, valueU, valueV, interp_method); CHECKERROR(err);
  if (interp_method == BGRID_VELOCITY)
    interp_method = BGRID_W_VELOCITY;
  err = temporal_interpolation(x, y, z, time, W, vxi, vyi, vzi, vti, valueW, interp_method); CHECKERROR(err);
  return SUCCESS;
}



#ifdef __cplusplus
}
#endif
#endif
