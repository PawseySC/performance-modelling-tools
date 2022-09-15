#!/bin/sh
# Copyright 2022 Pawsey Supercomputing Centre
#
# Authors
#
#  Joe Schoonover, Fluid Numerics LLC (joe@fluidnumerics.com)
#
# Description :
#
#  This script runs the ekondis/mixbench tools to gather system
#  specifications, theoretical peaks, and empirically measured
#  performance indicator
#  The stdout is post-processed with mixbench-report.py
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
#   There are a set of environment variables that you can set to control
#   the behavior of this script
#
#    PERF_ROOT
# 
# ///////////////////////////////////////////////////////////////////////// #



PERFMODELING="${PERF_ROOT:-$MYGROUP/performance-modelling-tools}"


case "$HOSTNAME" in

  # Mulan
  mulan*)
    module unload gcc/9.3.0
    module load craype-accel-amd-gfx908
    module load rocm/4.5.0
    module load cray-hdf5/1.12.0.6
    mixbench="mixbench-hip"

esac 


echo $HOSTNAME
cwd=$(pwd)
odir="$cwd/${HOSTNAME}/$(date +"%Y-%m-%d-%H-%M")"
echo "Saving results to : $odir"
mkdir -p $odir


cd $odir
git clone https://github.com/ekondis/mixbench.git
mkdir build
cd build
cmake $odir/mixbench/$mixbench
make


# Launch the application with rocprof
srun --exact \
     --ntasks=1 \
     --cpus-per-task=1 \
     --ntasks-per-socket=1 \
     --threads-per-core=1 \
     ./mixbench-hip > $odir/mixbench-log.txt

# Clean up source code
rm -rf $odir/mixbench $odir/build

# Process output
python3 $PERFMODELING/bin/mixbench-report.py --csv --json $odir/mixbench-log.txt

export PERFMODELING_ODIR=$odir
