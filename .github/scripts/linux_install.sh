#!/bin/bash

wget http://repo.continuum.io/miniconda/${MINICONDA_NAME} -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda env create --name parcels --file environment_py${PY_VERSION}_${OS_NAME}.yml
source activate parcels
python setup.py install
