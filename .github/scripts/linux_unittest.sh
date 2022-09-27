#!/bin/bash
export PATH="$HOME/miniconda/bin:$PATH"
source activate parcels

# Set up display to be able to plot in linux
export DISPLAY=:99.0;
sh -e /etc/init.d/xvfb start;
sleep 3;

pytest -v -s tests/ ;
