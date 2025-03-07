# TODO

List of tasks that are important to do before the release of version 4 (but can't be done now via code changes in `v4-dev`).

- [ ] Make migration guide for v3 to v4
- [ ] Just prior to release: Update conda feedstock recipe dependencies (remove cgen and compiler dependencies). Make sure that recipe is up-to-date.
- [ ] Revamp the oceanparcels.org landing page, and perhaps also consider new logo/branding?
- [ ] Rerun all the tutorials so that their output is in line with new v4 print statements etc
- Documentation
  - [ ] Look into xarray and whether users can create periodic datasets without increasing the size of the original dataset (i.e., no compromise alternative to `time_periodic` param in v3). Update docs accordingly.
  - [ ] Look into xarray and whether users can create datasets from snapshots assigning different time dimensions without increasing the size of the original dataset (i.e., no compromise alternative to `timestamps` param in v3). Update docs accordingly.
