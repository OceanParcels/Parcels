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
  if (up >=0.){
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


static inline StatusCode calcAdvectionAnalytical_JIT(CField *fu, CField *fv, CField *fw, double *flow3D_dbl,
                              CField *e2u, CField *e1v, CField *e1t, CField *e2t, CField *e3t,
                              int *xi, int *yi, int *zi, int *ti,
                              double *lon, double *lat, double *depth,
                              double *time, double *dt,
                              double *particle_dlon, double *particle_dlat, double *particle_ddepth)
{
  StatusCode status;
  CStructuredGrid *grid = fu->grid->grid;
  GridType gtype = fu->grid->gtype;
  GridIndexingType gridindexingtype = fu->gridindexingtype;
  int xdim = grid->xdim;
  int igrid = fu->igrid;

  double tol_grid = 1e-4;  // Tolerance for grid search
  double tol_compute = 1e-10;  // Tolerance for analytical computation
  int maxCellupdates = 10;  // Maximum number of cell updates before throwing an interpolation error

  double direction = 1;
  if (*dt < 0)
    direction = -1;

  double xsi, rs_x, ds_x, B_x, delta_x;
  double eta, rs_y, ds_y, B_y, delta_y;
  double zeta, rs_z, ds_z, B_z, delta_z;

  double tau;
  double ds_t = *dt;
  int tii, first_tstep_only;

  if (grid->tdim == 1){
    tii = 0;
    tau = 0;
    status = search_indices(*lon, *lat, *depth, grid, &xi[igrid], &yi[igrid], &zi[igrid],
                            &xsi, &eta, &zeta, gtype, tii, 0, 0, 1, CGRID_VELOCITY, gridindexingtype);
    first_tstep_only = 1;
  } else {
    status = search_time_index(time, grid->tdim, grid->time, &ti[igrid], fu->time_periodic,
                               grid->tfull_min, grid->tfull_max, grid->periods); CHECKSTATUS(status);

    double t0 = grid->time[ti[igrid]];
    double t1 = grid->time[ti[igrid]+1];
    tau = (*time - t0) / (t1 - t0);
    double ds_tfull = 600;  // TODO now hardcoded to 10 minutes
    if (direction > 0){
      if (tau < tol_grid){
        ds_tfull /= 2;
      }
    } else {
      if (tau > 1 - tol_grid){
        ds_tfull /= 2;
      }
    }
    ds_t = min(ds_tfull, fabs(*dt))*direction;

    tii = ti[igrid];
    status = search_indices(*lon, *lat, *depth, grid, &xi[igrid], &yi[igrid], &zi[igrid],
                            &xsi, &eta, &zeta, gtype, tii, *time, t0, t1, CGRID_VELOCITY, gridindexingtype);
    first_tstep_only = 0;
  }
  CHECKSTATUS(status);

  float dataU_2D[2][2][2];
  float dataV_2D[2][2][2];
  float dataU_3D[2][2][2][2];
  float dataV_3D[2][2][2][2];
  float dataW_3D[2][2][2][2];

  int flow3D = (int) *flow3D_dbl;

  bool updateCells = 1;
  int numUpdates = 0;
  while ((updateCells == 1) && (numUpdates < maxCellupdates)){
    if (flow3D == 0){
      status = getCell2D(fu, *xi, *yi, tii, dataU_2D, first_tstep_only); CHECKSTATUS(status);
      status = getCell2D(fv, *xi, *yi, tii, dataV_2D, first_tstep_only); CHECKSTATUS(status);

      updateCells = 0;
      if (xsi > 1 - tol_grid){
        if (((grid->tdim > 1) && (((1-tau)*dataU_2D[0][1][1] + tau*dataU_2D[1][1][1])*direction > 0)) ||
           ((grid->tdim == 1) && (dataU_2D[0][1][1]*direction > 0))){
          *xi += 1;
          xsi = 0;
          updateCells = 1;
        }
      } else if (xsi < tol_grid){
        if (((grid->tdim > 1) && (((1-tau)*dataU_2D[0][1][0] + tau*dataU_2D[1][1][0])*direction < 0)) ||
           ((grid->tdim == 1) && (dataU_2D[0][1][0]*direction < 0))){
          *xi -= 1;
          xsi = 1;
          updateCells = 1;
        }
      }
      if (eta > 1 - tol_grid){
        if (((grid->tdim > 1) && (((1-tau)*dataV_2D[0][1][1] + tau*dataV_2D[1][1][1])*direction > 0)) ||
           ((grid->tdim == 1) && (dataV_2D[0][1][1]*direction > 0))){
          *yi += 1;
          eta = 0;
          updateCells = 1;
        }
      } else if (eta < tol_grid){
        if (((grid->tdim > 1) && (((1-tau)*dataV_2D[0][0][1] + tau*dataV_2D[1][0][1])*direction < 0)) ||
           ((grid->tdim == 1) && (dataV_2D[0][0][1]*direction < 0))){
          *yi -= 1;
          eta = 1;
          updateCells = 1;
        }
      }
    } else if (flow3D == 1){
      status = getCell3D(fu, *xi, *yi, *zi, tii, dataU_3D, first_tstep_only); CHECKSTATUS(status);
      status = getCell3D(fv, *xi, *yi, *zi, tii, dataV_3D, first_tstep_only); CHECKSTATUS(status);
      status = getCell3D(fw, *xi, *yi, *zi, tii, dataW_3D, first_tstep_only); CHECKSTATUS(status);

      updateCells = 0;
      if (xsi > 1 - tol_grid){
        if (((grid->tdim > 1) && (((1-tau)*dataU_3D[0][1][1][1] + tau*dataU_3D[1][1][1][1])*direction > 0)) ||
           ((grid->tdim == 1) && (dataU_3D[0][1][1][1]*direction > 0))){
          *xi += 1;
          xsi = 0;
          updateCells = 1;
        }
      } else if (xsi < tol_grid){
        if (((grid->tdim > 1) && (((1-tau)*dataU_3D[0][1][1][0] + tau*dataU_3D[1][1][1][0])*direction < 0)) ||
           ((grid->tdim == 1) && (dataU_3D[0][1][1][0]*direction < 0))){
          *xi -= 1;
          xsi = 1;
          updateCells = 1;
        }
      }
      if (eta > 1 - tol_grid){
        if (((grid->tdim > 1) && (((1-tau)*dataV_3D[0][1][1][1] + tau*dataV_3D[1][1][1][1])*direction > 0)) ||
           ((grid->tdim == 1) && (dataV_3D[0][1][1][1]*direction > 0))){
          *yi += 1;
          eta = 0;
          updateCells = 1;
        }
      } else if (eta < tol_grid){
        if (((grid->tdim > 1) && (((1-tau)*dataV_3D[0][1][0][1] + tau*dataV_3D[1][1][0][1])*direction < 0)) ||
           ((grid->tdim == 1) && (dataV_3D[0][1][0][1]*direction < 0))){
          *yi -= 1;
          eta = 1;
          updateCells = 1;
        }
      }
      if (zeta > 1 - tol_grid){
        if (((grid->tdim > 1) && (((1-tau)*dataW_3D[0][1][1][1] + tau*dataW_3D[1][1][1][1])*direction > 0)) ||
           ((grid->tdim == 1) && (dataW_3D[0][1][1][1]*direction > 0))){
          *zi += 1;
          zeta = 0;
          updateCells = 1;
        }
      } else if (zeta < tol_grid){
        if (((grid->tdim > 1) && (((1-tau)*dataW_3D[0][0][1][1] + tau*dataW_3D[1][0][1][1])*direction < 0)) ||
           ((grid->tdim == 1) && (dataW_3D[0][0][1][1]*direction < 0))){
          *zi -= 1;
          zeta = 1;
          updateCells = 1;
        }
      }
    }
    numUpdates++;
  }
  if (numUpdates >= maxCellupdates){
    printf("Number of cell updates exceeded maximum number of updates\n");
    return ERRORINTERPOLATION;
  }

  double px[4];
  double py[4];
  int iN;
  if( (gtype == RECTILINEAR_Z_GRID) || (gtype == RECTILINEAR_S_GRID) ){
    float *xgrid = grid->lon;
    float *ygrid = grid->lat;
    for (iN=0; iN < 4; ++iN){
      px[iN] = xgrid[*xi+min(1, (iN%3))];
      py[iN] = ygrid[*yi+iN/2];
    }
  }
  else{
    float (* xgrid)[xdim] = (float (*)[xdim]) grid->lon;
    float (* ygrid)[xdim] = (float (*)[xdim]) grid->lat;
    for (iN=0; iN < 4; ++iN){
      px[iN] = xgrid[*yi+iN/2][*xi+min(1, (iN%3))];
      py[iN] = ygrid[*yi+iN/2][*xi+min(1, (iN%3))];
    }
    for (int i4 = 1; i4 < 4; ++i4){
      if (px[i4] < px[0] - 180) px[i4] += 360;
      if (px[i4] > px[0] + 180) px[i4] -= 360;
    }
  }

  double pz[2];
  double dz = 1;
  if (flow3D == 1){
    pz[0] = grid->depth[*zi];
    pz[1] = grid->depth[*zi+1];
    float data_e3t[2][2][2][2];
    status = getCell3D(e3t, 0, 0, *zi, tii, data_e3t, 1); CHECKSTATUS(status);
    dz = data_e3t[0][0][0][0];
  }

  float data_e2u[2][2][2];
  float data_e1v[2][2][2];
  float data_e1t[2][2][2];
  float data_e2t[2][2][2];
  status = getCell2D(e2u, *xi, *yi, 0, data_e2u, 1); CHECKSTATUS(status);
  status = getCell2D(e1v, *xi, *yi, 0, data_e1v, 1); CHECKSTATUS(status);
  status = getCell2D(e1t, *xi, *yi, 0, data_e1t, 1); CHECKSTATUS(status);
  status = getCell2D(e2t, *xi, *yi, 0, data_e2t, 1); CHECKSTATUS(status);
  double c4 = data_e2u[0][1][0];
  double c2 = data_e2u[0][1][1];
  double c1 = data_e1v[0][0][1];
  double c3 = data_e1v[0][1][1];
  double dxdy = data_e1t[0][0][0] * data_e2t[0][0][0];

  double U0, U1, V0, V1, W0, W1;
  if (flow3D == 0){
    if (grid->tdim > 1){
      U0 = ((1-tau)*dataU_2D[0][1][0] + tau*dataU_2D[1][1][0]) * c4 * direction;
      U1 = ((1-tau)*dataU_2D[0][1][1] + tau*dataU_2D[1][1][1]) * c2 * direction;
      V0 = ((1-tau)*dataV_2D[0][0][1] + tau*dataV_2D[1][0][1]) * c1 * direction;
      V1 = ((1-tau)*dataV_2D[0][1][1] + tau*dataV_2D[1][1][1]) * c3 * direction;
    } else {
      U0 = dataU_2D[0][1][0] * c4 * direction;
      U1 = dataU_2D[0][1][1] * c2 * direction;
      V0 = dataV_2D[0][0][1] * c1 * direction;
      V1 = dataV_2D[0][1][1] * c3 * direction;
    }
  } else if (flow3D == 1){
    if (grid->tdim > 1){
      U0 = ((1-tau)*dataU_3D[0][1][1][0] + tau*dataU_3D[1][1][1][0]) * c4 * dz * direction;
      U1 = ((1-tau)*dataU_3D[0][1][1][1] + tau*dataU_3D[1][1][1][1]) * c2 * dz * direction;
      V0 = ((1-tau)*dataV_3D[0][1][0][1] + tau*dataV_3D[1][1][0][1]) * c1 * dz * direction;
      V1 = ((1-tau)*dataV_3D[0][1][1][1] + tau*dataV_3D[1][1][1][1]) * c3 * dz * direction;
      W0 = ((1-tau)*dataW_3D[0][0][1][1] + tau*dataW_3D[1][0][1][1]) * dxdy * direction;
      W1 = ((1-tau)*dataW_3D[0][1][1][1] + tau*dataW_3D[1][1][1][1]) * dxdy * direction;
    } else {
      U0 = dataU_3D[0][1][1][0] * c4 * dz * direction;
      U1 = dataU_3D[0][1][1][1] * c2 * dz * direction;
      V0 = dataV_3D[0][1][0][1] * c1 * dz * direction;
      V1 = dataV_3D[0][1][1][1] * c3 * dz * direction;
      W0 = dataW_3D[0][0][1][1] * dxdy * direction;
      W1 = dataW_3D[0][1][1][1] * dxdy * direction;
    }
  }

  compute_ds(U0, U1, xsi, direction, tol_compute, &ds_x, &B_x, &delta_x);
  compute_ds(V0, V1, eta, direction, tol_compute, &ds_y, &B_y, &delta_y);
  if (flow3D == 1){
    compute_ds(W0, W1, zeta, direction, tol_compute, &ds_z, &B_z, &delta_z);
  } else {
    ds_z = 1.0/0.0;
  }

  double s_min = min(min(min(fabs(ds_x), fabs(ds_y)), fabs(ds_z)), fabs(ds_t / (dxdy * dz)));

  rs_x = compute_rs(xsi, B_x, delta_x, s_min, tol_compute);
  rs_y = compute_rs(eta, B_y, delta_y, s_min, tol_compute);

  *particle_dlon += (1.-rs_x)*(1.-rs_y) * px[0] + rs_x * (1.-rs_y) * px[1] + rs_x * rs_y * px[2] + (1.-rs_x)*rs_y * px[3] - *lon;
  *particle_dlat += (1.-rs_x)*(1.-rs_y) * py[0] + rs_x * (1.-rs_y) * py[1] + rs_x * rs_y * py[2] + (1.-rs_x)*rs_y * py[3] - *lat;

  if (flow3D == 1){
    rs_z = compute_rs(zeta, B_z, delta_z, s_min, tol_compute);
    *particle_ddepth += (1.-rs_z) * pz[0] + rs_z * pz[1] - *depth;
  }

  if ((rs_x < -tol_grid) || (rs_x > 1+tol_grid) ||
      (rs_y < -tol_grid) || (rs_y > 1+tol_grid) ||
      ((flow3D == 1) && ((rs_z < -tol_grid) || (rs_z > 1+tol_grid)))){

    printf("Particle out of bounds\n");
    printf("rs_x, rs_y, rs_z, s_min = %f, %f, %f %f\n", rs_x, rs_y, rs_z, s_min);
    printf("xi, yi, zi, ti = %d, %d, %d, %d\n", *xi, *yi, *zi, tii);
    printf("xsi, eta, zeta, tau = %f, %f, %f %f\n", xsi, eta, zeta, tau);
    printf("c1, c2, c3, c4 = %f, %f, %f, %f\n", c1, c2, c3, c4);
    printf("U0, U1, V0, V1, W0, W1 = %f, %f, %f, %f, %f, %f\n", U0, U1, V0, V1, W0, W1);
    printf("px, py, pz = %f, %f, %f\n", px[0], py[0], pz[0]);
    printf("dxdy, dz = %f, %f\n", dxdy, dz);

    return ERRORINTERPOLATION;
  }

  if (*dt > 0){
    *dt = max(direction * s_min * dxdy * dz, 1e-7);
  } else {
    *dt = min(direction * s_min * dxdy * dz, 1e-7);
  }

  return SUCCESS;
}
