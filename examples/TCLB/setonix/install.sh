#!/bin/bash
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -A pawsey0007-gpu
#SBATCH -o install.out
#SBATCH -e install.out
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


CODENAME="TCLB"
REPO="https://github.com/FluidNumerics/TCLB.git"
#GIT_BRANCH="master"
GIT_BRANCH="launch_bounds_64_6"
ROCM_VERSION="5.0.2"
APP=d3q27_PSM_NEBB


CHECKOUT_DIR=${CODENAME}_${GIT_BRANCH}_rocm-${ROCM_VERSION}

cwd=$(pwd)
if [ ! -e ${CODENAME}_${GIT_BRANCH} ] ; then
 git clone ${REPO} ${CHECKOUT_DIR}
fi

cd ${CHECKOUT_DIR}
git checkout ${GIT_BRANCH}

# Load modules
module load rocm/$ROCM_VERSION
module load r/4.1.0

# Install R dependencies
tools/install.sh rdep 

# Build commands
make configure
./configure --enable-hip \
	    --with-cpp-flags="-Rpass-analysis=kernel-resource-usage" \
	    --with-mpi-include=${CRAY_MPICH_DIR}/include \
	    --with-mpi-lib=${CRAY_MPICH_DIR}/lib
make -j ${APP} 
