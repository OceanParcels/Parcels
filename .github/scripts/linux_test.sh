#!/bin/bash
export PATH="$HOME/miniconda/bin:$PATH"
source activate parcels

# Set up display to be able to plot in linux
export DISPLAY=:99.0;
sh -e /etc/init.d/xvfb start;
sleep 3;

# only get examples on linux
parcels_get_examples examples/;

# run linter on linux
flake8 parcels;
flake8 tests;

# evaluate example scripts and notebooks on linux only
py.test -v -s examples/*.py;
py.test -v -s --nbval-lax examples/*tutorial*;
