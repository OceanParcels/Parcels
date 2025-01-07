#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef enum
  {
    ZONAL=0, MERIDIONAL=1, VERTICAL=2,
  } Orientation;


static inline void phi2D_lin(double eta, double xsi, double *phi)
{
    phi[0] = (1-xsi) * (1-eta);
    phi[1] =    xsi  * (1-eta);
    phi[2] =    xsi  *    eta ;
    phi[3] = (1-xsi) *    eta ;
}


static inline void phi1D_quad(double xsi, double *phi)
{
    phi[0] = 2*xsi*xsi-3*xsi+1;
    phi[1] = -4*xsi*xsi+4*xsi;
    phi[2] = 2*xsi*xsi-xsi;
}


static inline void dphidxsi3D_lin(double zeta, double eta, double xsi, double *dphidzeta, double *dphideta, double *dphidxsi)
{
  dphidxsi[0] = - (1-eta) * (1-zeta);
  dphidxsi[1] =   (1-eta) * (1-zeta);
  dphidxsi[2] =   (  eta) * (1-zeta);
  dphidxsi[3] = - (  eta) * (1-zeta);
  dphidxsi[4] = - (1-eta) * (  zeta);
  dphidxsi[5] =   (1-eta) * (  zeta);
  dphidxsi[6] =   (  eta) * (  zeta);
  dphidxsi[7] = - (  eta) * (  zeta);

  dphideta[0] = - (1-xsi) * (1-zeta);
  dphideta[1] = - (  xsi) * (1-zeta);
  dphideta[2] =   (  xsi) * (1-zeta);
  dphideta[3] =   (1-xsi) * (1-zeta);
  dphideta[4] = - (1-xsi) * (  zeta);
  dphideta[5] = - (  xsi) * (  zeta);
  dphideta[6] =   (  xsi) * (  zeta);
  dphideta[7] =   (1-xsi) * (  zeta);

  dphidzeta[0] = - (1-xsi) * (1-eta);
  dphidzeta[1] = - (  xsi) * (1-eta);
  dphidzeta[2] = - (  xsi) * (  eta);
  dphidzeta[3] = - (1-xsi) * (  eta);
  dphidzeta[4] =   (1-xsi) * (1-eta);
  dphidzeta[5] =   (  xsi) * (1-eta);
  dphidzeta[6] =   (  xsi) * (  eta);
  dphidzeta[7] =   (1-xsi) * (  eta);
}

static inline void dxdxsi3D_lin(double *pz, double *py, double *px, double zeta, double eta, double xsi, double *jacM, int sphere_mesh)
{
  double dphidxsi[8], dphideta[8], dphidzeta[8];
  dphidxsi3D_lin(zeta, eta, xsi, dphidzeta, dphideta, dphidxsi);

  int i;
  for(i=0; i<9; ++i)
      jacM[i] = 0;

  double deg2m = 1852 * 60.;
  double rad = M_PI / 180.;
  double lat = (1-xsi) * (1-eta) * py[0]+
                  xsi  * (1-eta) * py[1]+
                  xsi  *    eta  * py[2]+
               (1-xsi) *    eta  * py[3];
  double jac_lon = (sphere_mesh == 1) ? (deg2m * cos(rad * lat) ) : 1;
  double jac_lat = (sphere_mesh == 1) ? deg2m : 1;

  for(i=0; i<8; ++i){
    jacM[3*0+0] += px[i] * dphidxsi[i] * jac_lon; // dxdxsi
    jacM[3*0+1] += px[i] * dphideta[i] * jac_lon; // dxdeta
    jacM[3*0+2] += px[i] * dphidzeta[i] * jac_lon; // dxdzeta
    jacM[3*1+0] += py[i] * dphidxsi[i] * jac_lat; // dydxsi
    jacM[3*1+1] += py[i] * dphideta[i] * jac_lat; // dydeta
    jacM[3*1+2] += py[i] * dphidzeta[i] * jac_lat; // dydzeta
    jacM[3*2+0] += pz[i] * dphidxsi[i];           // dzdxsi
    jacM[3*2+1] += pz[i] * dphideta[i];           // dzdeta
    jacM[3*2+2] += pz[i] * dphidzeta[i];           // dzdzeta
  }
}

static inline double jacobian3D_lin_face(double *pz, double *py, double *px,
                                         double zeta, double eta, double xsi,
                                         Orientation orientation, int sphere_mesh)
{
  double jacM[9];
  dxdxsi3D_lin(pz, py, px, zeta, eta, xsi, jacM, sphere_mesh);

  double j[3];

  if (orientation == ZONAL){
    j[0] = jacM[3*1+1]*jacM[3*2+2]-jacM[3*1+2]*jacM[3*2+1];
    j[1] =-jacM[3*0+1]*jacM[3*2+2]+jacM[3*0+2]*jacM[3*2+1];
    j[2] = jacM[3*0+1]*jacM[3*1+2]-jacM[3*0+2]*jacM[3*1+1];
  }
  else if (orientation == MERIDIONAL){
    j[0] = jacM[3*1+0]*jacM[3*2+2]-jacM[3*1+2]*jacM[3*2+0];
    j[1] =-jacM[3*0+0]*jacM[3*2+2]+jacM[3*0+2]*jacM[3*2+0];
    j[2] = jacM[3*0+0]*jacM[3*1+2]-jacM[3*0+2]*jacM[3*1+0];
  }
  else if (orientation == VERTICAL){
    j[0] = jacM[3*1+0]*jacM[3*2+1]-jacM[3*1+1]*jacM[3*2+0];
    j[1] =-jacM[3*0+0]*jacM[3*2+1]+jacM[3*0+1]*jacM[3*2+0];
    j[2] = jacM[3*0+0]*jacM[3*1+1]-jacM[3*0+1]*jacM[3*1+0];
  }

  return sqrt(j[0]*j[0]+j[1]*j[1]+j[2]*j[2]);
}

static inline double jacobian3D_lin(double *pz, double *py, double *px,
                                    double zeta, double eta, double xsi,
                                    int sphere_mesh)
{
  double jacM[9];
  dxdxsi3D_lin(pz, py, px, zeta, eta, xsi, jacM, sphere_mesh);

  double jac = jacM[3*0+0] * (jacM[3*1+1]*jacM[3*2+2] - jacM[3*2+1]*jacM[3*1+2])
             - jacM[3*0+1] * (jacM[3*1+0]*jacM[3*2+2] - jacM[3*2+0]*jacM[3*1+2])
             + jacM[3*0+2] * (jacM[3*1+0]*jacM[3*2+1] - jacM[3*2+0]*jacM[3*1+1]);


  return jac;
}

static inline double dot_prod(double *a, double *b, size_t n)
{
  double val = 0;
  int i = 0;
  for(i=0; i<n; ++i)
    val += a[i]*b[i];
  return val;
}
