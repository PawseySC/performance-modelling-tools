#!/bin/bash

# Source this script to setup the environment to work on the PACER project BLINK

mv  ~/.pawsey_project ~/.pawsey_project.bk
export CUDA_CACHE_DIR="/dev/shm/.cuda_cache"
echo director2183 > ~/.pawsey_project
module purge
module load cascadelake slurm/20.02.3  gcc/8.3.0
module use /group/director2183/software/centos7.6/modulefiles
module use /group/director2183/$USER/software/centos7.6/modulefiles
module load hdf5/1.10.5 boost/1.76.0 cmake/3.18.0 python/3.6.3 fftw/3.3.8 numpy/1.19.0 aoflagger/3.1.0 casacore/3.2.1 gsl/2.6 idg-gpu/0.7 libxml2/2.9.12 lua/5.3  pal/0.9.8  wcslib/7.6 pip/20.2.4 cuda/11.1
