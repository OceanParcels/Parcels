
static inline ErrorCode func(CField *f, double *lon, double *dt)
{
  float data2D[2][2][2];
  ErrorCode err = getCell2D(f, 1, 2, 0, data2D, 1); CHECKERROR(err);
  float u = data2D[0][0][0];
  *lon += u * *dt;
  return SUCCESS;
}
