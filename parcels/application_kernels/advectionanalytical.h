#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double compute_rs(double r, double B, double delta, double s_min, double tol){
  if (fabs(B) < tol){
    return -delta *s_min + r;
  } else {
    return (r + delta / B) * exp(-B * s_min) - delta / B;
  }
}

void compute_ds(double F0, double F1, double r, double direction, double tol,
                double *ds, double *B, double *delta){
  double up = F0 * (1-r) + F1 * r;
  double r_target = 0;
  if (direction * up >=0.){
    r_target = 1.;
  }
  *delta = -F0;
  *B = F0 - F1;
  double F_r1 = r_target + *delta / *B;
  double F_r0 = r + *delta / *B;
  if (fabs(*B) < tol){
    *B = 0;
    F_r1 = 0.0/0.0; // SET TO NAN
    F_r0 = 0.0/0.0; // SET TO NAN
  }

  if (fabs(*B) < tol && fabs(*delta) < tol){
      *ds = 1.0 /0.0;  // SET TO INFINITY
  } else if (*B == 0){
      *ds = -(r_target - r) / *delta;
  } else if (F_r1 * F_r0 < tol){
      *ds = 1.0 /0.0;  // SET TO INFINITY
  } else {
      *ds = - 1. / *B * log(F_r1 / F_r0);
  }

  if (fabs(*ds) < tol){
      *ds = 1.0 /0.0;  // SET TO INFINITY
  }
}


static inline StatusCode func(CField *fu, CField *fv, int *xi, int *yi, int *zi,
                              double *lon, double *lat, double *depth, double *time, double *dt,
                              double *particle_dlon, double *particle_dlat)
{

  StatusCode status;
  CStructuredGrid *grid = fu->grid->grid;
  GridCode gcode = RECTILINEAR_Z_GRID; // TODO generalize
  int gridindexingtype = NEMO; // TODO generalize
  int xdim = grid->xdim;
  int igrid = fu->igrid;

  double tol = 1e-10;
  int I_s = 10;  // number of intermediate time steps
  int direction = 1;
  bool withW = 0; // TODO also withW
  if (*dt < 0)
    direction = -1;
  double dz = 1; // TODO also for varying dz
  bool withTime = 0; // TODO also withTime

  int ti[1] = {0}; // TODO also input ti

  // if we're in between time indices, and not at the end of the timeseries,
  // we'll make sure to interpolate data between the two time values
  // otherwise, we'll only use the data at the current time index
  int tii = 1; // TODO fix //(ti[igrid] < grid->tdim-1 && time > grid->time[ti[igrid]]) ? 2 : 1;

  double t0 = 0; // TODO fix //grid->time[ti[igrid]];
  // we set our second time bound and search time depending on the
  // index critereon above
  double t1 = 1; // TODO fix //(tii == 2) ? grid->time[ti[igrid]+1] : t0+1;
  double tsrch = 0 ;// TODO fix //(tii == 2) ? time : t0;
  double ds_t = *dt; // TODO also support withTime

  double xsi, rs_x, ds_x, B_x, delta_x;
  double eta, rs_y, ds_y, B_y, delta_y;
  double zeta;

  status = search_indices(*lon, *lat, *depth, grid, xi, yi, zi,
			  &xsi, &eta, &zeta, gcode, ti[igrid],
			  tsrch, t0, t1, CGRID_VELOCITY, gridindexingtype);
  CHECKSTATUS(status);

  float dataU[2][2][2];
  float dataV[2][2][2];
  status = getCell2D(fu, *xi, *yi, ti[igrid], dataU, 1); CHECKSTATUS(status);
  status = getCell2D(fv, *xi, *yi, ti[igrid], dataV, 1); CHECKSTATUS(status);

  bool updateCells = 0;
  if (fabs(xsi - 1) < tol){
    if (dataU[0][1][1] > 0){
      *xi += 1;
      xsi = 0;
      updateCells = 1;
    }
  }

  if (fabs(eta - 1) < tol){
    if (dataV[0][1][1] > 0){
      *yi += 1;
      eta = 0;
      updateCells = 1;
    }
  }
  if (updateCells == 1){
    status = getCell2D(fu, *xi, *yi, ti[igrid], dataU, 1); CHECKSTATUS(status);
    status = getCell2D(fv, *xi, *yi, ti[igrid], dataV, 1); CHECKSTATUS(status);
  }

  double xgrid_loc[4];
  double ygrid_loc[4];
  int iN;
  if( (gcode == RECTILINEAR_Z_GRID) || (gcode == RECTILINEAR_S_GRID) ){
    float *xgrid = grid->lon;
    float *ygrid = grid->lat;
    for (iN=0; iN < 4; ++iN){
      xgrid_loc[iN] = xgrid[*xi+min(1, (iN%3))];
      ygrid_loc[iN] = ygrid[*yi+iN/2];
    }
  }
  else{
    float (* xgrid)[xdim] = (float (*)[xdim]) grid->lon;
    float (* ygrid)[xdim] = (float (*)[xdim]) grid->lat;
    for (iN=0; iN < 4; ++iN){
      xgrid_loc[iN] = xgrid[*yi+iN/2][*xi+min(1, (iN%3))];
      ygrid_loc[iN] = ygrid[*yi+iN/2][*xi+min(1, (iN%3))];
    }
  }
  int i4;
  for (i4 = 1; i4 < 4; ++i4){
    if (xgrid_loc[i4] < xgrid_loc[0] - 180) xgrid_loc[i4] += 360;
    if (xgrid_loc[i4] > xgrid_loc[0] + 180) xgrid_loc[i4] -= 360;
  }


  double phi[4];
  phi2D_lin(0., eta, phi);
  double U0 = direction * dataU[0][1][0] * dist(xgrid_loc[3], xgrid_loc[0], ygrid_loc[3], ygrid_loc[0], grid->sphere_mesh, dot_prod(phi, ygrid_loc, 4));
  phi2D_lin(1., eta, phi);
  double U1 = direction * dataU[0][1][1] * dist(xgrid_loc[1], xgrid_loc[2], ygrid_loc[1], ygrid_loc[2], grid->sphere_mesh, dot_prod(phi, ygrid_loc, 4));
  phi2D_lin(xsi, 0., phi);
  double V0 = direction * dataV[0][0][1] * dist(xgrid_loc[0], xgrid_loc[1], ygrid_loc[0], ygrid_loc[1], grid->sphere_mesh, dot_prod(phi, ygrid_loc, 4));
  phi2D_lin(xsi, 1., phi);
  double V1 = direction * dataV[0][1][1] * dist(xgrid_loc[2], xgrid_loc[3], ygrid_loc[2], ygrid_loc[3], grid->sphere_mesh, dot_prod(phi, ygrid_loc, 4));

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
  double dxdy = (dxdxsi*dydeta - dxdeta * dydxsi) * meshJac;

  compute_ds(U0, U1, xsi, direction, tol, &ds_x, &B_x, &delta_x);
  compute_ds(V0, V1, eta, direction, tol, &ds_y, &B_y, &delta_y);

  double s_min = min(min(fabs(ds_x), fabs(ds_y)), fabs(ds_t / (dxdy * dz)));

  rs_x = compute_rs(xsi, B_x, delta_x, s_min, tol);
  rs_y = compute_rs(eta, B_y, delta_y, s_min, tol);


  *particle_dlon = (1.-rs_x)*(1.-rs_y) * xgrid_loc[0] + rs_x * (1.-rs_y) * xgrid_loc[1] + rs_x * rs_y * xgrid_loc[2] + (1.-rs_x)*rs_y * xgrid_loc[3] - *lon;
  *particle_dlat = (1.-rs_x)*(1.-rs_y) * ygrid_loc[0] + rs_x * (1.-rs_y) * ygrid_loc[1] + rs_x * rs_y * ygrid_loc[2] + (1.-rs_x)*rs_y * ygrid_loc[3] - *lat;

  if (*dt > 0){
    *dt = max(direction * s_min * (dxdy * dz), 1e-7);
  } else {
    *dt = min(direction * s_min * (dxdy * dz), 1e-7);
  }
  printf("After %f %d %f\n", *lon, *xi, xsi);
  printf("After %f %d %f\n", *lat, *yi, eta);

  return SUCCESS;
}
