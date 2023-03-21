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

NVALS=(5 10 20 30 40)
PVALS=(1024 4096)

IMAGER_BRANCH="${IMAGER_BRANCH:-main}"
PERFMODELING="${PERFMODELING:-$MYGROUP/performance-modelling-tools}"

cwd=$(pwd)
timestamp=$(date +"%Y-%m-%d-%H-%M")
odir="${cwd}/cufftblocks_run_${timestamp}"
REPO="${cwd}/imager_${IMAGER_BRANCH}"

# Get the git sha
cd ${REPO}
gitsha=$(git rev-parse HEAD | cut -c 1-8)
cd ${cwd}

# Source the environment included in this repository
source ${PERFMODELING}/examples/blink_imager/topaz-v100/env.sh
module load cuda/11.1
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


# load test data module to set ENV variables :
module load blink_test_data/devel


# Link in input decks
mkdir -p ${odir}
cd $odir
ln -s ${BLINK_TEST_DATADIR}/eda2/20200209/images/u.fits 
ln -s ${BLINK_TEST_DATADIR}/eda2/20200209/images/v.fits 
ln -s ${BLINK_TEST_DATADIR}/eda2/20200209/images/w.fits 
ln -s ${BLINK_TEST_DATADIR}/eda2/20200209/chan_204_20200209T034646_vis_real.fits vis_real.fits
ln -s ${BLINK_TEST_DATADIR}/eda2/20200209/chan_204_20200209T034646_vis_imag.fits vis_imag.fits


# Run the code
cd $odir
echo "timestamp, hostname, gitsha, kernel, N, P, time" > clocks_profile.csv

for N in "${NVALS[@]}"
do
  for P in "${PVALS[@]}"
  do

    # cleaning old FITS files first :
    echo "rm -f re_??.fits im_??.fits"
    rm -f re_??.fits im_??.fits

    # Get a hot spot and trace profile
    ${REPO}/build_gpu/cufft_blocks -n $N \
                                        -p $P \
                                        -F 1 \
                                        -f 1 > cufft_blocks_$N_$P.txt

   gridding_time=$(grep "CLOCK gridding()" cufft_blocks_$N_$P.txt | awk -F " " '{print $5}')
   echo "${timestamp}, $(hostname), ${gitsha}, gridding, $N, $P, $gridding_time" >> clocks_profile.csv
   cufft_time=$(grep "CLOCK cufftPlanMany()" cufft_blocks_$N_$P.txt | awk -F " " '{print $5}')
   echo "${timestamp}, $(hostname), ${gitsha}, cufft, $N, $P, $gridding_time" >> clocks_profile.csv 


  done
done
