#!/bin/bash
# Copyright 2023 Pawsey Supercomputing Centre
#
# Authors
#
#  Joe Schoonover, Fluid Numerics LLC (joe@fluidnumerics.com)
#
# Description :
#
#  This script installs blink/imager on Topaz for V100s
# 
# Prerequisites :
#
# Usage :
#
#   This script is meant to be run on a compute node on Topaz
#   First, obtain an allocation
#
#    salloc -n1 -c 8 --threads-per-core=1 --partition=gpuq-dev --gres=gpu:1 --project=$PAWSEY_PROJECT
#
#   There are a set of environment variables that you can set to control
#   the behavior of this script
#
#     PERFMODLEING
#
#     JSON
#
# ///////////////////////////////////////////////////////////////////////// #

IMAGER_BRANCH="${IMAGER_BRANCH:-main}"
PERFMODELING="${PERFMODELING:-$MYGROUP/performance-modelling-tools}"

cwd=$(pwd)
odir="${cwd}/cufftblocks_profile_$(date +"%Y-%m-%d-%H-%M")"
REPO="${cwd}/imager_${IMAGER_BRANCH}"

# Source the environment included in this repository
source ${PERFMODELING}/examples/blink_imager/topaz-v100/env.sh
module load cuda
module load cascadelake slurm/20.02.3 gcc/8.3.0 cmake/3.18.0
module use /group/director2183/software/centos7.6/modulefiles
module load ds9
module load msfitslib/devel 
module load msfitslib/devel libnova
module load pal/0.9.8
module load libnova/0.15.0
module load cascadelake
module load gcc/8.3.0
module load cfitsio/3.48
module load cmake/3.18.0
module use /group/director2183/msok/software/centos7.6/modulefiles/


# Link the input decks
mkdir -p ${odir}
ln -s ${REPO}/build_gpu/*.fits ${odir}/

# Run the code
cd $odir
${REPO}/build_gpu/cufft_blocks 10 180 u.fits v.fits w.fits chan_204_20200209T034646_vis_real.fits chan_204_20200209T034646_vis_imag.fits 1

# 1024x1024: 
#/group/director2183/data/test/ganiruddha/NEW_TEST/cuFFT_GITLAB_NCHANNELS/imager_devel/build/cufft_blocks 1 1024 u.fits v.fits w.fits chan_204_20200209T034646_vis_real.fits chan_204_20200209T034646_vis_imag.fits 1| tee out.txt

# 8182x8182:
#/group/director2183/data/test/ganiruddha/NEW_TEST/cuFFT_GITLAB_NCHANNELS/imager_devel/build/cufft_blocks 1 8182 u.fits v.fits w.fits chan_204_20200209T034646_vis_real.fits chan_204_20200209T034646_vis_imag.fits 1| tee out.txt
