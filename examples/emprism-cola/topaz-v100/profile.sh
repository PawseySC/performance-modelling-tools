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
#    salloc -n1 -c 32 --gres=gpu:1
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
timestamp=$(date +"%Y-%m-%d-%H-%M")

odir="${cwd}/profile_${COLA_BRANCH}_${timestamp}"
REPO="${cwd}/cola-sprint_${COLA_BRANCH}"

module load cuda intel openmpi-ucx-gpu/4.0.2

mkdir ${odir}
cd ${REPO}/cuda/
nvprof -o ${odir}/quarkpropGPU.nvprof ./quarkpropGPU.x <<< "test"
nvprof --kernels ".*Clover.*FBCGPU.*",\
                --analysis-metrics  -o ${odir}/quarkpropGPU_metrics.nvprof ./quarkpropGPU.x <<< "test"
