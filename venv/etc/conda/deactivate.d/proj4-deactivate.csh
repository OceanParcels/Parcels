#!/usr/bin/env csh

# Restore previous env vars if they were set.
unsetenv PROJ_DATA
unsetenv PROJ_NETWORK

if ( $?_CONDA_SET_PROJ_DATA ) then
    setenv PROJ_DATA "$_CONDA_SET_PROJ_DATA"
    unsetenv _CONDA_SET_PROJ_DATA
endif
