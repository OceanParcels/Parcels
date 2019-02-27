
static inline void func(CField *f, double *lon, double *dt)
{
  float (*data)[f->xdim] = (float (*)[f->xdim]) f->data;
  float u = data[2][1];
  *lon += u * *dt;
}
