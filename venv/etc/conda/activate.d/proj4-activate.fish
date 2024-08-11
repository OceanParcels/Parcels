#!/usr/bin/env fish

if set -q PROJ_DATA
  set -gx _CONDA_SET_PROJ_DATA "$PROJ_DATA"
end

if test -d "$CONDA_PREFIX/share/proj"
  set -gx PROJ_DATA "$CONDA_PREFIX/share/proj"
else if test -d "$CONDA_PREFIX/Library/share/proj"
  set -gx PROJ_DATA "$CONDA_PREFIX/Library/share/proj"
end

if test -f "$CONDA_PREFIX/share/proj/copyright_and_licenses.csv"
  # proj-data is installed because its license was copied over
  set -gx PROJ_NETWORK "OFF"
else
  set -gx PROJ_NETWORK "ON"
end
