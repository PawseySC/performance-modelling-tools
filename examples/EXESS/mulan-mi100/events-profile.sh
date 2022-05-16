#!/bin/sh
# Copyright 2022 Pawsey Supercomputing Centre
#
# Authors
#
#  Joe Schoonover, Fluid Numerics LLC (joe@fluidnumerics.com)
#
# Description :
#
#  This script runs exess-dev under rocprof to create hotspot profile
#  and trace profile
# 
# Prerequisites :
#
#   If you have access to Joe Schoonover's spack installation,
#
#   export SPACK_ROOT=/group/pawsey0007/jschoonover/spack
#   export PERF_ROOT=/group/pawsey0007/jschoonover/performance-modeling-tools
#
#   If you do not have access to Joe Schoonover's spack installation
#   on Pawsey systems, you can install spack and the Mulan spack
#   environment for performance-modelling-tools.
#
#   Step by step walkthrough can be found at [TO DO]
#
# Usage :
#
#   This script is meant to be run on a compute node on Mulan
#   First, obtain an allocation
#
#    salloc -n1 -c 32 --threads-per-core=1 --partition=workq --project=$PAWSEY_PROJECT
#
#   Build EXESS-dev using the `install.sh` script in this directory
#
#   ./install.sh
#
#   There are a set of environment variables that you can set to control
#   the behavior of this script
#
#     EXESS_ROOT
#
#     SPACK_ROOT
#
#     PERF_ROOT
#
#     INPUT
#
#     KERNEL 
#
# ///////////////////////////////////////////////////////////////////////// #


EXESS="${EXESS_ROOT:-EXESS-dev_cpu_port}"
SPACK="${SPACK_ROOT:-$MYGROUP/spack}"
PERFMODELING="${PERF_ROOT:-$MYGROUP/performance-modeling-tools}"
INPUT="${INPUT_DECK:-inputs/json_inputs/scf/w1.json}"

if [ -z "$KERNEL" ];
  echo "You need to set KERNEL variable to run events profiling"
  exit 1
fi

cp rocprof-input.tmpl rocprof-input.txt
sed -i "s/@KERNEL@/$KERNEL/g" rocprof-input.txt
cat rocprof-input.txt

# Launch the application with
OMP_NUM_THREADS=1 srun --exact \
                       --ntasks=1 \
                       --cpus-per-task=32 \
                       --threads-per-core=1 \
                       --cpu-bind=socket \
                       rocprof -i rocprof-input.txt \
                       $EXESS/exess $EXESS/$INPUT
