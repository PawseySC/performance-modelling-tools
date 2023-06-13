#!/bin/bash
#
# Copyright 2023 Pawsey Supercomputing Centre
#
# Authors
#
#  Joe Schoonover, Fluid Numerics LLC (joe@fluidnumerics.com)
#
# Description :
#
#  This script installs TCLB on topaz
#
#SBATCH -n1 
#SBATCH -c 8 
#SBATCH --threads-per-core=1
#SBATCH --partition=gpuq-dev
#SBATCH --gres=gpu:1 
###SBATCH --project=pawsey0007
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
module load cuda/11.4.2 r/4.0.2 openmpi-ucx-gpu/4.0.2

# Install R dependencies
tools/install.sh rdep 

# Build commands
make configure
./configure
make -j 8 d2q9
