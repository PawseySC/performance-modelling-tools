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
#    PERF_ODIR
# 
# ///////////////////////////////////////////////////////////////////////// #
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuq-dev
#SBATCH --time=01:00:00

module load python/3.6.3
module load cuda/11.4.2
module load cmake/3.18.0
mixbench="mixbench-cuda"

cwd=$(pwd)

PERFMODELING="${PERF_ROOT:-$MYGROUP/performance-modelling-tools}"
ODIR="${PERF_ODIR:-$cwd/${HOSTNAME}/$(date +"%Y-%m-%d-%H-%M")}"


echo "Saving results to : $ODIR"
mkdir -p $ODIR

cd $ODIR
git clone https://github.com/ekondis/mixbench.git
mkdir build
cd build
cmake $ODIR/mixbench/$mixbench
make


# Launch the application with rocprof
srun --exact \
     --ntasks=1 \
     --cpus-per-task=1 \
     --ntasks-per-socket=1 \
     --threads-per-core=1 \
     $mixbench > $ODIR/mixbench-log.txt

# Clean up source code
rm -rf $ODIR/mixbench $ODIR/build

# Process output
python3 $PERFMODELING/bin/mixbench-report.py --csv --json $ODIR/mixbench-log.txt

