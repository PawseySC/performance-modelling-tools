#!/bin/bash
# Copyright 2023 Pawsey Supercomputing Centre
#
# Authors
#
#  Joe Schoonover, Fluid Numerics LLC (joe@fluidnumerics.com)
#
# Description :
#
# Prerequisites :
#
#   git clone https://github.com/PawseySC/performance-modelling-tools.git $MYGROUP/performance-modelling-tools
#
# Usage :
#
#   This script is meant to be run on a compute node on Mulan with 1 MI100 GPU
#   First, obtain an allocation
#
#    salloc -n1 -c 32 --threads-per-core=1 --gres=gpu:1 --partition=workq --project=$PAWSEY_PROJECT --mem=240G
#
#   Build imager using the `install.sh` script in this directory
#
#   ./install.sh
#
#   There are a set of environment variables that you can set to control
#   the behavior of this script
#
#     EXESS_ROOT
#
#     PERF_ROOT
#
#     INPUT
# 
# ///////////////////////////////////////////////////////////////////////// #

# 180 x 180: (This prints the printf/output statements on the screen as well into a .txt file)
cwd=$(pwd)
odir="$cwd/profile_$(date +"%Y-%m-%d-%H-%M")"




./build_gpu/cufft_blocks 10 180 u.fits v.fits w.fits chan_204_20200209T034646_vis_real.fits chan_204_20200209T034646_vis_imag.fits 1| tee out.txt
