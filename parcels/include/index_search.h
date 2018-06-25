#ifndef _INDEX_SEARCH_H
#define _INDEX_SEARCH_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECKERROR(res) do {if (res != SUCCESS) return res;} while (0)

typedef enum
  {
    SUCCESS=0, REPEAT=1, DELETE=2, ERROR=3, ERROR_OUT_OF_BOUNDS=4, ERROR_TIME_EXTRAPOLATION =5
  } ErrorCode;

typedef enum
  {
    RECTILINEAR_Z_GRID=0, RECTILINEAR_S_GRID=1, CURVILINEAR_Z_GRID=2, CURVILINEAR_S_GRID=3
  } GridCode;

static inline ErrorCode search_indices_vertical_z(float z, int zdim, float *zvals, int *zi, double *zeta)
{
  if (z < zvals[0] || z > zvals[zdim-1]) {return ERROR_OUT_OF_BOUNDS;}
  while (*zi < zdim-1 && z > zvals[*zi+1]) ++(*zi);
  while (*zi > 0 && z < zvals[*zi]) --(*zi);
  if (*zi == zdim-1) {--*zi;}

  *zeta = (z - zvals[*zi]) / (zvals[*zi+1] - zvals[*zi]);
  return SUCCESS;
}

static inline ErrorCode search_indices_vertical_s(float z, int xdim, int ydim, int zdim, float *zvals,
                                    int xi, int yi, int *zi, double xsi, double eta, double *zeta,
                                    int z4d, int ti, int tdim, double time, double t0, double t1)
{
  float zcol[zdim];
  int zii;
  if (z4d == 1){
    float (*zvalstab)[zdim][ydim][xdim] = (float (*)[zdim][ydim][xdim]) zvals;
    int ti1 = ti;
    if (ti < tdim-1)
       ti1= ti+1;
    double zt0, zt1;
    for (zii=0; zii < zdim; zii++){
      zt0 = (1-xsi)*(1-eta) * zvalstab[ti ][zii][yi  ][xi  ]
          + (  xsi)*(1-eta) * zvalstab[ti ][zii][yi  ][xi+1]
          + (  xsi)*(  eta) * zvalstab[ti ][zii][yi+1][xi+1]
          + (1-xsi)*(  eta) * zvalstab[ti ][zii][yi+1][xi  ];
      zt1 = (1-xsi)*(1-eta) * zvalstab[ti1][zii][yi  ][xi  ]
          + (  xsi)*(1-eta) * zvalstab[ti1][zii][yi  ][xi+1]
          + (  xsi)*(  eta) * zvalstab[ti1][zii][yi+1][xi+1]
          + (1-xsi)*(  eta) * zvalstab[ti1][zii][yi+1][xi  ];
      zcol[zii] = zt0 + (zt1 - zt0) * (float)((time - t0) / (t1 - t0));
    }

  }
  else{
    float (*zvalstab)[ydim][xdim] = (float (*)[ydim][xdim]) zvals;
    for (zii=0; zii < zdim; zii++){
      zcol[zii] = (1-xsi)*(1-eta) * zvalstab[zii][yi  ][xi  ]
                + (  xsi)*(1-eta) * zvalstab[zii][yi  ][xi+1]
                + (  xsi)*(  eta) * zvalstab[zii][yi+1][xi+1]
                + (1-xsi)*(  eta) * zvalstab[zii][yi+1][xi  ];
    }
  }

  if (z < zcol[0] || z > zcol[zdim-1]) {return ERROR_OUT_OF_BOUNDS;}
  while (*zi < zdim-1 && z > zcol[*zi+1]) ++(*zi);
  while (*zi > 0 && z < zcol[*zi]) --(*zi);
  if (*zi == zdim-1) {--*zi;}

  *zeta = (z - zcol[*zi]) / (zcol[*zi+1] - zcol[*zi]);
  return SUCCESS;
}


static inline void fix_1d_index(int *xi, int xdim, int sphere_mesh)
{
  if (*xi < 0){
    if (sphere_mesh)
      (*xi) = xdim-2;
    else
      (*xi) = 0;
  }
  if (*xi > xdim-2){
    if (sphere_mesh)
      (*xi) = 0;
    else
      (*xi) = xdim-2;
  }
}


static inline void fix_2d_indices(int *xi, int *yi, int xdim, int ydim, int sphere_mesh)
{
  if (*xi < 0){
    if (sphere_mesh)
      (*xi) = xdim-2;
    else
      (*xi) = 0;
  }
  if (*xi > xdim-2){
    if (sphere_mesh)
      (*xi) = 0;
    else
      (*xi) = xdim-2;
  }
  if (*yi < 0){
    (*yi) = 0;
  }
  if (*yi > ydim-2){
    (*yi) = ydim-2;
    if (sphere_mesh)
      (*xi) = xdim - (*xi);
  }
}


static inline ErrorCode search_indices_rectilinear(float x, float y, float z, int xdim, int ydim, int zdim,
                                            float *xvals, float *yvals, float *zvals, int sphere_mesh, int zonal_periodic, GridCode gcode,
                                            int *xi, int *yi, int *zi, double *xsi, double *eta, double *zeta,
                                            int z4d, int ti, int tdim, double time, double t0, double t1)
{
  if (sphere_mesh == 0){
    if (x < xvals[0] || x > xvals[xdim-1]) {return ERROR_OUT_OF_BOUNDS;}
    while (*xi < xdim-1 && x > xvals[*xi+1]) ++(*xi);
    while (*xi > 0 && x < xvals[*xi]) --(*xi);
    *xsi = (x - xvals[*xi]) / (xvals[*xi+1] - xvals[*xi]);
  }
  else{
    if (zonal_periodic == 0){
      if ((xvals[0] < xvals[xdim-1]) && (x < xvals[0] || x > xvals[xdim-1])) {return ERROR_OUT_OF_BOUNDS;}
      else if ((xvals[0] >= xvals[xdim-1]) && (x < xvals[0] && x > xvals[xdim-1])) {return ERROR_OUT_OF_BOUNDS;}
    }

    float xvalsi = xvals[*xi];
    if (xvalsi < x - 225) xvalsi += 360;
    if (xvalsi > x + 225) xvalsi -= 360;
    float xvalsi1 = xvals[*xi+1];
    if (xvalsi1 < xvalsi - 180) xvalsi1 += 360;
    if (xvalsi1 > xvalsi + 180) xvalsi1 -= 360;

    int itMax = 10000;
    int it = 0;
    while ( (xvalsi > x) || (xvalsi1 < x) ){
      if (xvalsi1 < x)
        ++(*xi);
      else if (xvalsi > x)
        --(*xi);
      fix_1d_index(xi, xdim, 1);
      xvalsi = xvals[*xi];
      if (xvalsi < x - 225) xvalsi += 360;
      if (xvalsi > x + 225) xvalsi -= 360;
      xvalsi1 = xvals[*xi+1];
      if (xvalsi1 < xvalsi - 180) xvalsi1 += 360;
      if (xvalsi1 > xvalsi + 180) xvalsi1 -= 360;
      it++;
      if (it > itMax){
        return ERROR_OUT_OF_BOUNDS;
      }
    }

    *xsi = (x - xvalsi) / (xvalsi1 - xvalsi);
  }

  if (y < yvals[0] || y > yvals[ydim-1]) {return ERROR_OUT_OF_BOUNDS;}
  while (*yi < ydim-1 && y > yvals[*yi+1]) ++(*yi);
  while (*yi > 0 && y < yvals[*yi]) --(*yi);

  *eta = (y - yvals[*yi]) / (yvals[*yi+1] - yvals[*yi]);

  ErrorCode err;
  if (zdim > 1){
    switch(gcode){
      case RECTILINEAR_Z_GRID:
        err = search_indices_vertical_z(z, zdim, zvals, zi, zeta);
        break;
      case RECTILINEAR_S_GRID:
        err = search_indices_vertical_s(z, xdim, ydim, zdim, zvals,
                                        *xi, *yi, zi, *xsi, *eta, zeta,
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
                                            float *xvals, float *yvals, float *zvals, int sphere_mesh, int zonal_periodic, GridCode gcode,
                                            int *xi, int *yi, int *zi, double *xsi, double *eta, double *zeta,
                                            int z4d, int ti, int tdim, double time, double t0, double t1)
{
  // NEMO convention
  float (* xgrid)[xdim] = (float (*)[xdim]) xvals;
  float (* ygrid)[xdim] = (float (*)[xdim]) yvals;

  if (zonal_periodic == 0 || sphere_mesh == 0) {
    if ((xgrid[0][0] < xgrid[0][xdim-1]) && (x < xgrid[0][0] || x > xgrid[0][xdim-1])) {return ERROR_OUT_OF_BOUNDS;}
    else if ((xgrid[0][0] >= xgrid[0][xdim-1]) && (x < xgrid[0][0] && x > xgrid[0][xdim-1])) {return ERROR_OUT_OF_BOUNDS;}
  }

  double a[4], b[4];

  *xsi = *eta = -1;
  int maxIterSearch = 1e6, it = 0;
  while ( (*xsi < 0) || (*xsi > 1) || (*eta < 0) || (*eta > 1) ){
    double xgrid_loc[4] = {xgrid[*yi][*xi], xgrid[*yi][*xi+1], xgrid[*yi+1][*xi+1], xgrid[*yi+1][*xi]};
    if (sphere_mesh){ //we are on the sphere
      int i4;
      if (xgrid_loc[0] < x - 225) xgrid_loc[0] += 360;
      if (xgrid_loc[0] > x + 225) xgrid_loc[0] -= 360;
      for (i4 = 1; i4 < 4; ++i4){
        if (xgrid_loc[i4] < xgrid_loc[0] - 180) xgrid_loc[i4] += 360;
        if (xgrid_loc[i4] > xgrid_loc[0] + 180) xgrid_loc[i4] -= 360;
      }
    }
    double ygrid_loc[4] = {ygrid[*yi][*xi], ygrid[*yi][*xi+1], ygrid[*yi+1][*xi+1], ygrid[*yi+1][*xi]};

    a[0] =  xgrid_loc[0];
    a[1] = -xgrid_loc[0]    + xgrid_loc[1];
    a[2] = -xgrid_loc[0]                                              + xgrid_loc[3];
    a[3] =  xgrid_loc[0]    - xgrid_loc[1]      + xgrid_loc[2]        - xgrid_loc[3];
    b[0] =  ygrid_loc[0];
    b[1] = -ygrid_loc[0]    + ygrid_loc[1];
    b[2] = -ygrid_loc[0]                                              + ygrid_loc[3];
    b[3] =  ygrid_loc[0]    - ygrid_loc[1]      + ygrid_loc[2]        - ygrid_loc[3];

    double aa = a[3]*b[2] - a[2]*b[3];
    double bb = a[3]*b[0] - a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + x*b[3] - y*a[3];
    double cc = a[1]*b[0] - a[0]*b[1] + x*b[1] - y*a[1];
    if (fabs(aa) < 1e-12)  // Rectilinear  cell, or quasi
      *eta = -cc / bb;
    else{
      double det = sqrt(bb*bb-4*aa*cc);
      if (det == det)  // so, if det is nan we keep the xsi, eta from previous iter
        *eta = (-bb+det)/(2*aa);
    }
    *xsi = (x-a[0]-a[2]* (*eta)) / (a[1]+a[3]* (*eta));
    if ( (*xsi < 0) && (*eta < 0) && (*xi == 0) && (*yi == 0) )
      return ERROR_OUT_OF_BOUNDS;
    if ( (*xsi > 1) && (*eta > 1) && (*xi == xdim-1) && (*yi == ydim-1) )
      return ERROR_OUT_OF_BOUNDS;
    if (*xsi < 0) 
      (*xi)--;
    if (*xsi > 1)
      (*xi)++;
    if (*eta < 0)
      (*yi)--;
    if (*eta > 1)
      (*yi)++;
    fix_2d_indices(xi, yi, xdim, ydim, sphere_mesh);
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
        err = search_indices_vertical_z(z, zdim, zvals, zi, zeta);
        break;
      case CURVILINEAR_S_GRID:
        err = search_indices_vertical_s(z, xdim, ydim, zdim, zvals,
                                        *xi, *yi, zi, *xsi, *eta, zeta,
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
                                            int *xi, int *yi, int *zi, double *xsi, double *eta, double *zeta,
                                            int sphere_mesh, int zonal_periodic,
                                            GridCode gcode, int z4d,
                                            int ti, int tdim, double time, double t0, double t1)
{
  switch(gcode){
    case RECTILINEAR_Z_GRID:
    case RECTILINEAR_S_GRID:
      return search_indices_rectilinear(x, y, z, xdim, ydim, zdim, xvals, yvals, zvals, sphere_mesh, zonal_periodic, gcode, xi, yi, zi, xsi, eta, zeta,
                                   z4d, ti, tdim, time, t0, t1);
      break;
    case CURVILINEAR_Z_GRID:
    case CURVILINEAR_S_GRID:
      return search_indices_curvilinear(x, y, z, xdim, ydim, zdim, xvals, yvals, zvals, sphere_mesh, zonal_periodic, gcode, xi, yi, zi, xsi, eta, zeta,
                                   z4d, ti, tdim, time, t0, t1);
      break;
    default:
      printf("Only RECTILINEAR_Z_GRID, RECTILINEAR_S_GRID, CURVILINEAR_Z_GRID and CURVILINEAR_S_GRID grids are currently implemented\n");
      return ERROR;
  }
}

/* Local linear search to update time index */
static inline ErrorCode search_time_index(double *t, int size, double *tvals, int *ti, int time_periodic)
{
  if (*ti < 0)
    *ti = 0;
  if (time_periodic == 1){
    if (*t < tvals[0]){
      *ti = size-1;
      int periods = floor( (*t-tvals[0])/(tvals[size-1]-tvals[0]));
      *t -= periods * (tvals[size-1]-tvals[0]);
      search_time_index(t, size, tvals, ti, time_periodic);
    }  
    else if (*t > tvals[size-1]){
      *ti = 0;
      int periods = floor( (*t-tvals[0])/(tvals[size-1]-tvals[0]));
      *t -= periods * (tvals[size-1]-tvals[0]);
      search_time_index(t, size, tvals, ti, time_periodic);
    }  
  }          
  while (*ti < size-1 && *t >= tvals[*ti+1]) ++(*ti);
  while (*ti > 0 && *t < tvals[*ti]) --(*ti);
  return SUCCESS;
}


#ifdef __cplusplus
}
#endif
#endif
