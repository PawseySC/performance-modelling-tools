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

odir="${cwd}/rocprof_${COLA_BRANCH}_${timestamp}"
REPO="${cwd}/cola-sprint_${COLA_BRANCH}"

#module use /pawsey/mulan/rocm/modulefiles
#module load rocm/5.4.3
module load rocm/4.5.0


mkdir ${odir}
cd ${REPO}/hip/

rocprof --stats --sys-trace ./quarkpropGPU.x <<< "test"
mv results.* ${odir}/

rocprof -i ${cwd}/rocprof_input.txt -o ${odir}/rocprof_metrics.csv ./quarkpropGPU.x <<'INPUT'
test
test
INPUT
