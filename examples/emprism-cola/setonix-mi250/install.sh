#!/bin/bash
# Copyright 2023 Pawsey Supercomputing Centre
#
# Authors
#
#  Joe Schoonover, Fluid Numerics LLC (joe@fluidnumerics.com)
#
# Description :
#
#  This script installs cola-sprint on Setonix (AMD MI250x)
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


COLA_BRANCH="${COLA_BRANCH:-master}"
PERFMODELING="${PERFMODELING:-$MYGROUP/performance-modelling-tools}"

cwd=$(pwd)
if [ ! -e cola-sprint_${COLA_BRANCH} ] ; then
 git clone git@bitbucket.org:lhytning/cola-sprint.git cola-sprint_${COLA_BRANCH}
fi

cd cola-sprint_${COLA_BRANCH}
git checkout ${COLA_BRANCH}

module load PrgEnv-cray/8.3.3 
module load craype-accel-amd-gfx90a
module load rocm/5.0.2

# Build the code for gpu acceleration
mkdir lib
cd ${cwd}/cola-sprint_${COLA_BRANCH}/include && ln -s craype.mk current.mk
cd ${cwd}/cola-sprint_${COLA_BRANCH}
make
