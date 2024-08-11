#!/bin/sh

# Restore previous env vars if they were set.
unset PROJ_DATA
unset PROJ_NETWORK

if [ -n "${_CONDA_SET_PROJ_DATA:-}" ]; then
    export PROJ_DATA="${_CONDA_SET_PROJ_DATA}"
    unset _CONDA_SET_PROJ_DATA
fi
