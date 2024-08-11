#!/bin/sh

# Store existing env vars and set to this conda env
# so other installs don't pollute the environment.

if [ -n "${PROJ_DATA:-}" ]; then
    export _CONDA_SET_PROJ_DATA=$PROJ_DATA
fi


if [ -d "${CONDA_PREFIX}/share/proj" ]; then
  export "PROJ_DATA=${CONDA_PREFIX}/share/proj"
elif [ -d "${CONDA_PREFIX}/Library/share/proj" ]; then
  export PROJ_DATA="${CONDA_PREFIX}/Library/share/proj"
fi

if [ -f "${CONDA_PREFIX}/share/proj/copyright_and_licenses.csv" ]; then
  # proj-data is installed because its license was copied over
  export PROJ_NETWORK="OFF"
else
  export PROJ_NETWORK="ON"
fi
