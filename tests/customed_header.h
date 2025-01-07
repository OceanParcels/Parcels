
static inline StatusCode func(CField *f, double *particle_dlon, double *dt)
{
  float data2D[2][2][2];
  StatusCode status = getCell2D(f, 0, 2, 1, data2D, 1); CHECKSTATUS(status);
  float u = data2D[0][0][0];
  *particle_dlon = +u * *dt;
  return SUCCESS;
}
