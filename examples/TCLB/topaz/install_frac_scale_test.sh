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
#SBATCH -e install.out
###SBATCH --project=pawsey0007
#
# ///////////////////////////////////////////////////////////////////////// #


CODENAME="TCLB"
REPO="https://github.com/CFD-GO/TCLB.git"
GIT_BRANCH="${GIT_BRANCH:-master}"

git clone https://github.com/llaniewski/frac_scale_test.git
cd frac_scale_test

# Load modules
module load cuda/11.4.2 r/4.0.2 openmpi-ucx-gpu/4.0.2


./install.sh master CFD-GO master LAMMPS=/group/director2188/sprint/LIGGGHTS-PUBLIC/
