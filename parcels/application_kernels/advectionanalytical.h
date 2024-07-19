#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

  float tol = 1e-10;
  int I_s = 10;  // number of intermediate time steps
  int direction = 1;
  // if (*dt < 0)
  //   direction = -1;
  bool withW = 0; // TODO also withW
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

  double xsi, eta, zeta;

  status = search_indices(*lon, *lat, *depth, grid, xi, yi, zi,
			  &xsi, &eta, &zeta, gcode, ti[igrid],
			  tsrch, t0, t1, CGRID_VELOCITY, gridindexingtype);
  CHECKSTATUS(status);

  printf("Before %f %d %f\n", *lon, *xi, xsi);
  printf("Before %f %d %f\n", *lat, *yi, eta);

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

  // double px = {grid->lon[*xi], grid->lon[*xi+1], grid->lon[*xi+1], grid->lon[*xi]};
  // double py = {grid->lat[*yi], grid->lat[*yi], grid->lat[*yi+1], grid->lat[*yi+1]};

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

  // First compute U
  double up = U0 * (1-xsi) + U1 * xsi;
  double xsi_target = 0;
  if (direction * up >=0.){
    xsi_target = 1.;
  }
  double delta_x = -U0;
  double B_x = U0 - U1;
  double U_r1 = xsi_target + delta_x / B_x;
  double U_r0 = xsi + delta_x / B_x;
  if (fabs(B_x) < tol){
    B_x = 0;
    U_r1 = 0.0/0.0; // SET TO NAN
    U_r0 = 0.0/0.0; // SET TO NAN
  }

  double ds_x;
  if (fabs(B_x) < tol && fabs(delta_x) < tol){
      ds_x = 1.0 /0.0;  // SET TO INFINITY
  } else if (B_x == 0){
      ds_x = -(xsi_target - xsi) / delta_x;
  } else if (U_r1 * U_r0 < tol){
      ds_x = 1.0 /0.0;  // SET TO INFINITY
  } else {
      ds_x = - 1. / B_x * log(U_r1 / U_r0);
  }

  if (fabs(ds_x) < tol){
      ds_x = 1.0 /0.0;  // SET TO INFINITY
  }

  // Now compute V
  double vp = V0 * (1-eta) + V1 * eta;
  double eta_target = 0;
  if (direction * vp >=0.){
    eta_target = 1.;
  }
  double delta_y = -V0;
  double B_y = V0 - V1;
  double V_r1 = eta_target + delta_y / B_y;
  double V_r0 = eta + delta_y / B_y;
  if (fabs(B_y) < tol){
    B_y = 0;
    V_r1 = 0.0/0.0; // SET TO NAN
    V_r0 = 0.0/0.0; // SET TO NAN
  }

  double ds_y;
  if (fabs(B_y) < tol && fabs(delta_y) < tol){
      ds_y = 1.0 /0.0;  // SET TO INFINITY
  } else if (B_x == 0){
      ds_y = -(eta_target - eta) / delta_y;
  } else if (V_r1 * V_r0 < tol){
      ds_y = 1.0 /0.0;  // SET TO INFINITY
  } else {
      ds_y = - 1. / B_y * log(V_r1 / V_r0);
  }

  if (fabs(ds_y) < tol){
      ds_y = 1.0 /0.0;  // SET TO INFINITY
  }

  double s_min = min(min(fabs(ds_x), fabs(ds_y)), fabs(ds_t / (dxdy * dz)));

  double rs_x, rs_y;
  if (fabs(B_x) < tol){
    rs_x = -delta_x * s_min + xsi;
  } else {
    rs_x = (xsi + delta_x / B_x) * exp(-B_x * s_min) - delta_x / B_x;
  }

  if (fabs(B_y) < tol){
    rs_y = -delta_y * s_min + eta;
  } else {
    rs_y = (eta + delta_y / B_y) * exp(-B_y * s_min) - delta_y / B_y;
  }

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
