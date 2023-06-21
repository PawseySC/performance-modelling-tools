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
REPO="https://github.com/CFD-GO/TCLB.git"
GIT_BRANCH="${GIT_BRANCH:-master}"

cwd=$(pwd)
if [ ! -e ${CODENAME}_${GIT_BRANCH}-rocm-5-4-3 ] ; then
 git clone ${REPO} ${CODENAME}_${GIT_BRANCH}_rocm-5-4-3
fi

cd ${CODENAME}_${GIT_BRANCH}_rocm-5-4-3
git checkout ${GIT_BRANCH}

# Load modules
module load rocm/5.4.3
module load r/4.1.0

# Install R dependencies
tools/install.sh rdep 

# Build commands
make configure
./configure --enable-hip \
	    --with-mpi-include=${CRAY_MPICH_DIR}/include \
	    --with-mpi-lib=${CRAY_MPICH_DIR}/lib

make -j d2q9
make -j d3q27_PSM_NEBB
