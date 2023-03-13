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

#NVALS=(1 5 10 50 100)
#PVALS=(180 1024 8182)
NVALS=(10)
PVALS=(8182)

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
for N in "${NVALS[@]}"
do
  for P in "${PVALS[@]}"
  do

    # Get a hot spot and trace profile
    nvprof -o cufft_blocks_${N}_${P}.nvprof \
         ${REPO}/build_gpu/cufft_blocks -n $N \
                                        -p $P \
                                        -u u.fits \
                                        -v v.fits \
                                        -w w.fits \
                                        -r chan_204_20200209T034646_vis_real.fits \
                                        -i chan_204_20200209T034646_vis_imag.fits \
                                        -f 1 > cufft_blocks_$N_$P.txt


    # Get analysis metrics for detailed kernel profiling
    nvprof --analysis-metrics -o cufft_blocks_${N}_${P}_metrics.nvprof \
         ${REPO}/build_gpu/cufft_blocks -n $N \
                                        -p $P \
                                        -u u.fits \
                                        -v v.fits \
                                        -w w.fits \
                                        -r chan_204_20200209T034646_vis_real.fits \
                                        -i chan_204_20200209T034646_vis_imag.fits \
                                        -f 1

  done
done
