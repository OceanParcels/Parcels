#!/usr/bin/env fish

# Restore previous env vars if they were set.
set -e PROJ_DATA
set -e PROJ_NETWORK

if set -q _CONDA_SET_PROJ_DATA
    set -gx  PROJ_DATA "$_CONDA_SET_PROJ_DATA"
    set -e _CONDA_SET_PROJ_DATA
end
