#!/bin/bash
# Copyright 2023 Pawsey Supercomputing Centre
#
# Authors
#
#  Joe Schoonover, Fluid Numerics LLC (joe@fluidnumerics.com)
#
# Description :
#
#  This script installs blink/imager on Topaz for V100s
# 
# Prerequisites :
#
# Usage :
#
#   This script is meant to be run on a compute node on Topaz
#   First, obtain an allocation
#
#    salloc -n1 -c 8 --threads-per-core=1 --partition=gpuq-dev --gres=gpu:1 --project=$PAWSEY_PROJECT
#
#   There are a set of environment variables that you can set to control
#   the behavior of this script
#
#     PERFMODLEING
#
#     JSON
#
# ///////////////////////////////////////////////////////////////////////// #


IMAGER_BRANCH="${IMAGER_BRANCH:-main}"
PERFMODELING="${PERFMODELING:-$MYGROUP/performance-modelling-tools}"

cwd=$(pwd)
if [ ! -e imager_${IMAGER_BRANCH} ] ; then
 git clone git@146.118.67.64:blink/imager.git imager_${IMAGER_BRANCH}
fi

cd imager_${IMAGER_BRANCH}
git checkout ${IMAGER_BRANCH}

# Source the environment included in this repository
source ${PERFMODELING}/examples/blink_imager/topaz-v100/env.sh

# Build the code for gpu acceleration
#
# For some reason, we need to run the build twice. The first attempt fails
# with the following error
#
#```
#  [  6%] Building CUDA object CMakeFiles/cufft_blocks.dir/apps/cufft_blocks.cu.o
#   /pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/nvcc -forward-unknown-to-host-compiler -DIMAGER_HIP -D_UNIX  -O3 -DNDEBUG --generate-code=arch=compute_70,code=[compute_70,sm_70] -std=c++14 -x cu -c /group/director2183/jschoonover/performance-modelling-tools/examples/blink_imager/topaz-v100/imager_main/apps/cufft_blocks.cu -o CMakeFiles/cufft_blocks.dir/apps/cufft_blocks.cu.o
#   /group/director2183/jschoonover/performance-modelling-tools/examples/blink_imager/topaz-v100/imager_main/apps/cufft_blocks.cu:125:10: fatal error: gridding_imaging_cuda.h: No such file or directory
#    #include "gridding_imaging_cuda.h"
#             ^~~~~~~~~~~~~~~~~~~~~~~~~
#   compilation terminated.
#   make[2]: *** [CMakeFiles/cufft_blocks.dir/apps/cufft_blocks.cu.o] Error 1
#```


${cwd}/imager_${IMAGER_BRANCH}/build.sh gpu || ${cwd}/imager_${IMAGER_BRANCH}/build.sh gpu
