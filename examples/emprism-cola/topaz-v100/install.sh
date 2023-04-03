#!/bin/bash
# Copyright 2023 Pawsey Supercomputing Centre
#
# Authors
#
#  Joe Schoonover, Fluid Numerics LLC (joe@fluidnumerics.com)
#
# Description :
#
#  This script installs cola-sprint on Mulan (AMD MI100)
# 
# Prerequisites :
#
# Usage :
#
#   This script is meant to be run on a compute node on Topaz
#   First, obtain an allocation
#
#    salloc -n1 -c 32 --gres=gpu:1 --partition=gpuq-dev
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

module load cuda intel openmpi-ucx-gpu/4.0.2

# Build the code for gpu acceleration
mkdir -p lib
cd ${cwd}/cola-sprint_${COLA_BRANCH}/include && ln -s intel.mk current.mk && cd ${cwd}/cola-sprint_${COLA_BRANCH}
make
