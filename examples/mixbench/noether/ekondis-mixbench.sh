#!/bin/sh
# Copyright 2023 Pawsey Supercomputing Centre
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
#   salloc -N 1 -p gpu -n 8 -c 8 --gpus-per-task=1 --exclusive -A pawsey0007-gpu
#
#   There are a set of environment variables that you can set to control
#   the behavior of this script
#
#    PERF_ROOT
#
#    PERF_ODIR
# 
# ///////////////////////////////////////////////////////////////////////// #
#

mixbench="mixbench-hip"

cwd=$(pwd)

PERFMODELING="${PERF_ROOT:-${HOME}/performance-modelling-tools}"
ODIR="${PERF_ODIR:-$cwd/$(hostname)/$(date +"%Y-%m-%d-%H-%M")}"


echo "Saving results to : $ODIR"
mkdir -p $ODIR


cd $ODIR
git clone https://github.com/ekondis/mixbench.git
mkdir build
cd build
cmake $ODIR/mixbench/$mixbench
make

$ODIR/build/$mixbench > $ODIR/mixbench-log.txt

# Clean up source code
rm -rf $ODIR/mixbench $ODIR/build

# Process output
python3 $PERFMODELING/bin/mixbench-report.py --csv --json $ODIR/mixbench-log.txt

