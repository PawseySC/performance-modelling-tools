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
#SBATCH -o install.out
#SBATCH -e install.err
###SBATCH --project=pawsey0007
#
# ///////////////////////////////////////////////////////////////////////// #


CODENAME="frac_scale_test"
REPO="https://github.com/llaniewski/frac_scale_test.git"
GIT_BRANCH="${GIT_BRANCH:-master}"

#cwd=$(pwd)
#if [ ! -e ${CODENAME}_${GIT_BRANCH} ] ; then
# git clone ${REPO} ${CODENAME}_${GIT_BRANCH}
#fi
#
#module load cuda/11.4.2 r/4.0.2 openmpi-ucx-gpu/4.0.2
#
#cd ${CODENAME}_${GIT_BRANCH}
#git checkout ${GIT_BRANCH}
#
#./gen.sh

ln -s /group/director2188/sprint/frac_scale_test/data/ ./

