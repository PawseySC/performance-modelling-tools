#!/bin/bash
# Copyright 2022 Pawsey Supercomputing Centre
#
# Authors
#
#  Marco De La Pierre, Pawsey Supercomputing Centre
#  Joe Schoonover, Fluid Numerics LLC (joe@fluidnumerics.com)
#
# Description :
#
#  This script installs EXESS-dev on Mulan for MI100 GPUs
# 
# Prerequisites :
#
# Usage :
#
#   This script is meant to be run on a compute node on Mulan
#   First, obtain an allocation
#
#    salloc -n1 -c 32 --threads-per-core=1 --partition=workq --project=$PAWSEY_PROJECT --mem=240G
#
#   Build EXESS-dev using the `.sh` script in this directory
#
#   ./.sh
#
#   There are a set of environment variables that you can set to control
#   the behavior of this script
#
#     EXESS_GIT_BRANCH
#
#     PERF_ROOT
#
#     JSON
#
#     EXESS_USE_MAGMA
#
#     EXESS_GPU_BOARD
#
#     EXESS_USE_HIP
# 
# ///////////////////////////////////////////////////////////////////////// #


EXESS_GIT_BRANCH="${EXESS_GIT_BRANCH:-hip_dev}"
PERFMODELING="${PERF_ROOT:-$MYGROUP/performance-modelling-tools}"
JSON="${JSON:-$MYGROUP/mulan/json}"
USE_MAGMA="${EXESS_USE_MAGMA:-False}"
GPU_BOARD="${EXESS_GPU_BOARD:-MI100}"
USE_HIP="${EXESS_USE_HIP:-True}"

git_branch=$EXESS_GIT_BRANCH
git_checkout="" #821ec2790dd89de71c3aae6e5516fed5a09c6044 #9May2022

module unload gcc/9.3.0
module load craype-accel-amd-gfx908
module load rocm/4.5.0
module load cray-hdf5/1.12.0.6
. /pawsey/mulan/bin/init-mi100-hipsolver-4.5.0.sh
. /pawsey/mulan/bin/init-mi100-magma-2.6.2.sh
. /pawsey/mulan/bin/init-cmake-3.21.4.sh

export MPI_ROOT="$MPICH_DIR"
export HDF5_ROOT="$HDF5_DIR"
if [ $USE_MAGMA == "True" ] ; then
  export MAGMA_ROOT="$MAGMA_HOME"
  CMAKE_LINALG_DEFINE="-DMAGMA_ROOT=$MAGMA_ROOT"
elif [ $USE_MAGMA == "False" ] ; then
  export MATHLIB_ROOT="$ROCM_PATH"
  CMAKE_LINALG_DEFINE="-DMATHLIB_ROOT=$MATHLIB_ROOT"
else
  echo "Variable USE_MAGMA must be set to either True or False. Exiting."
  exit 1
fi
export HIP_PATH="$ROCM_PATH/hip"
export JSON_ROOT="$JSON"


if [ ! -e $JSON_ROOT ] ; then
 mkdir -p $(dirname "$JSON_ROOT")
 git clone https://github.com/nlohmann/json $JSON_ROOT	
fi 

if [ ! -e EXESS-dev_$git_branch ] ; then
# git clone git:EXESS-dev/EXESS-dev EXESS-dev$git_branch
 git clone git@github.com:EXESS-dev/EXESS-dev.git EXESS-dev_$git_branch
fi

cd EXESS-dev_$git_branch
git checkout $git_branch
git checkout $git_checkout

mkdir -p build && cd build
rm -rf ./*

CXX=hipcc cmake .. \
  -DMPI_ROOT=$MPI_ROOT \
  -DHDF5_ROOT=$HDF5_ROOT \
  -DJSON_ROOT=$JSON_ROOT \
  $CMAKE_LINALG_DEFINE \
  -DHIP=$USE_HIP \
  -DBUILD_RIMP2=0 \
  -DMAGMA=$USE_MAGMA \
  -DGPU_BOARD=$GPU_BOARD

make -j $SLURM_CPUS_PER_TASK

sg $PAWSEY_PROJECT -c 'make install'

