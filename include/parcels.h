#include <stdio.h>

/* Grid dimension to be defined during the JIT process */
extern const int GRID_XDIM;
extern const int GRID_YDIM;

/* Default definition of Particle type structure */
#ifndef PARCELS_PTYPE
#define PARCELS_PTYPE
typedef struct
{
  float lon, lat;
  int xi, yi;
} Particle;
#endif


/* Local search to update grid index */
static inline int advance_index(float x, int i, int size, float *xvals)
{
    while (i < size-1 && x > xvals[i+1]) ++i;
    while (i > 0 && x < xvals[i]) --i;
    return i;
}


/* Bilinear interpolation routine for 2D grid */
static inline float interpolate_bilinear(float x, float y, int xi, int yi,
                                        float xvals[GRID_XDIM], float yvals[GRID_YDIM],
                                        float qvals[GRID_YDIM][GRID_XDIM])
{
  int i = xi, j = yi;
  i = advance_index(x, i, GRID_XDIM, xvals);
  j = advance_index(y, j, GRID_YDIM, yvals);
  return (qvals[i][j] * (xvals[i+1] - x) * (yvals[j+1] - y)
        + qvals[i+1][j] * (x - xvals[i]) * (yvals[j+1] - y)
        + qvals[i][j+1] * (xvals[i+1] - x) * (y - yvals[j])
        + qvals[i+1][j+1] * (x - xvals[i]) * (y - yvals[j]))
        / ((xvals[i+1] - xvals[i]) * (yvals[j+1] - yvals[j]));
}


/* 4th-order Runge-Kutta algorithm for 2D grid */
static inline void runge_kutta4(Particle *p, float dt,
                                float lon_u[GRID_XDIM], float lat_u[GRID_YDIM],
                                float lon_v[GRID_XDIM], float lat_v[GRID_YDIM],
                                float u[GRID_YDIM][GRID_XDIM], float v[GRID_YDIM][GRID_XDIM])
{
  float f, u1, v1, u2, v2, u3, v3, u4, v4;
  float lon1, lat1, lon2, lat2, lon3, lat3;

  f = dt / 1000. / 1.852 / 60.;
  u1 = interpolate_bilinear(p->lat, p->lon, p->yi, p->xi, lat_u, lon_u, u);
  v1 = interpolate_bilinear(p->lat, p->lon, p->yi, p->xi, lat_v, lon_v, v);
  lon1 = p->lon + u1*.5*f; lat1 = p->lat + v1*.5*f;
  u2 = interpolate_bilinear(lat1, lon1, p->yi, p->xi, lat_u, lon_u, u);
  v2 = interpolate_bilinear(lat1, lon1, p->yi, p->xi, lat_v, lon_v, v);
  lon2 = p->lon + u2*.5*f; lat2 = p->lat + v2*.5*f;
  u3 = interpolate_bilinear(lat2, lon2, p->yi, p->xi, lat_u, lon_u, u);
  v3 = interpolate_bilinear(lat2, lon2, p->yi, p->xi, lat_v, lon_v, v);
  lon3 = p->lon + u3*f; lat3 = p->lat + v3*f;
  u4 = interpolate_bilinear(lat3, lon3, p->yi, p->xi, lat_u, lon_u, u);
  v4 = interpolate_bilinear(lat3, lon3, p->yi, p->xi, lat_v, lon_v, v);

  // Advance particle position in space and on the grid
  p->lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * f;
  p->lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * f;
  p->xi = advance_index(p->lon, p->xi, GRID_XDIM, lon_u);
  p->yi = advance_index(p->lat, p->yi, GRID_YDIM, lat_u);
}
