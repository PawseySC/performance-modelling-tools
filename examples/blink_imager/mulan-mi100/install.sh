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
git pull

source /group/director2183/mulan/setup.sh
module use /pawsey/mulan/rocm/modulefiles
module load rocm/5.4.3

# Build the code for gpu acceleration
${cwd}/imager_${IMAGER_BRANCH}/build.sh gpu
