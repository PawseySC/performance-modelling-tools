#!/bin/bash
# Copyright 2023 Pawsey Supercomputing Centre
#
# Authors
#
#  Joe Schoonover, Fluid Numerics LLC (joe@fluidnumerics.com)
#
# Description :
#
#  This script installs TCLB on Setonix (AMD MI250x)
# 
# Prerequisites :
#
# Usage :
#
#   This script is meant to be run on a compute node on Topaz
#   First, obtain an allocation
#
#    salloc -N 1 -p gpu -n 8 -c 8 --gpus-per-task=1 --exclusive -A e31-gpu
#
#   There are a set of environment variables that you can set to control
#   the behavior of this script
#
#     PERFMODLEING
#
#     JSON
#
# ///////////////////////////////////////////////////////////////////////// #


CODENAME="TCLB"
REPO="https://github.com/CFD-GO/TCLB.git"
GIT_BRANCH="${GIT_BRANCH:-master}"

cwd=$(pwd)
if [ ! -e ${CODENAME}_${GIT_BRANCH} ] ; then
 git clone ${REPO} ${CODENAME}_${GIT_BRANCH}
fi

cd ${CODENAME}_${GIT_BRANCH}
git checkout ${GIT_BRANCH}

# Load modules
module swap PrgEnv-gnu PrgEnv-cray
module load rocm/5.0.2
module load r

# Build commands
make configure
./configure --enable-hip=/opt/rocm \
	    --disable-cuda
make d2q9
