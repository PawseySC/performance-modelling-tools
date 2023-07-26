#!/bin/bash
#SBATCH --partition=gpu
##SBATCH --ntasks=1
##SBATCH --nodes=1
##SBATCH --cpus-per-task=8
##SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --reservation=sprint8
#SBATCH -A pawsey0007-gpu
#SBATCH -o install_mini.out
#SBATCH -e install_mini.out
#
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
# ///////////////////////////////////////////////////////////////////////// #

miniapps=( "transport" "chemistry" "cons2prim" )
CODENAME="S3D"
REPO="git@github.com:unsw-edu-au/S3D_JICF.git"
GIT_BRANCH="pacer_mini_apps"
ROCM_VERSION="5.0.2"
MACH="STXgpu"


CHECKOUT_DIR=${CODENAME}_${GIT_BRANCH}_${MACH}_rocm-${ROCM_VERSION}

cwd=$(pwd)
if [ ! -e ${CODENAME}_${GIT_BRANCH} ] ; then
 git clone ${REPO} ${CHECKOUT_DIR}
fi

cd ${cwd}/${CHECKOUT_DIR}
git checkout ${GIT_BRANCH}

# Load modules
module load PrgEnv-cray/8.3.3 craype-accel-amd-gfx90a
#module load PrgEnv-aocc craype-accel-amd-gfx90a
##options = -O3 -qopenmp -r8
module load rocm/$ROCM_VERSION


export MACH

for app in "${miniapps[@]}"
do
  echo "##############################"
  echo "#  Building $app #"
  echo "##############################"
  cd ${cwd}/${CHECKOUT_DIR}/source/mini_apps/${app}_kernel/build
  mkdir -p ${cwd}/${CHECKOUT_DIR}/source/mini_apps/${app}_kernel/run
  make
done

# # Transport kernel
# cd ${cwd}/${CHECKOUT_DIR}/source/mini_apps/transport_kernel/build
# # Build commands
# export MACH
# make

# # Chemistry kernel
# cd ${cwd}/${CHECKOUT_DIR}/source/mini_apps/chemistry_kernel/build
# mkdir -p ${cwd}/${CHECKOUT_DIR}/source/mini_apps/chemistry_kernel/run
# # Build commands
# export MACH
# make

# # Cons2Prim kernel
# cd ${cwd}/${CHECKOUT_DIR}/source/mini_apps/cons2prim_kernel/build
# mkdir -p ${cwd}/${CHECKOUT_DIR}/source/mini_apps/cons2prim_kernel/run
# # Build commands
# export MACH
# make


