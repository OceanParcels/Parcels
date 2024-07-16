
static inline StatusCode func(CField *f, double *lon, double *dt)
{

  float tol = 1e-10;
  int I_s = 10;  // number of intermediate time steps
  int direction = 1;
  if (*dt < 0)
    direction = -1;
  bool withW = 0; // TODO also withW
  bool withTime = 0; // TODO also withTime

  double xsi, eta, zeta;

  StatusCode status;
  CStructuredGrid *grid = f->grid->grid;
  int igrid = f->igrid;

  // status = search_indices(x, y, z, grid, &xi[igrid], &yi[igrid], &zi[igrid],
	// 		  &xsi, &eta, &zeta, gcode, ti[igrid],
	// 		  tsrch, t0, t1, interp_method, gridindexingtype);
  CHECKSTATUS(status);

  return SUCCESS;
}
