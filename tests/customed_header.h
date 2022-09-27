
static inline StatusCode func(CField *f, double *lon, double *dt)
{
  float data2D[2][2][2];
  StatusCode status = getCell2D(f, 1, 2, 0, data2D, 1); CHECKSTATUS(status);
  float u = data2D[0][0][0];
  *lon += u * *dt;
  return SUCCESS;
}
