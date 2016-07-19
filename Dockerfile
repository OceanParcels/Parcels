FROM andrewosh/binder-base

MAINTAINER Erik van Sebille

USER root

# Add dependency
RUN apt-get update
RUN apt-get install libhdf5-serial-dev netcdf-bin libnetcdf-dev

USER main

# Install requirements for Python 2
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

