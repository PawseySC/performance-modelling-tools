#!/bin/sh
# Copyright 2022 Pawsey Supercomputing Centre
#
# Authors
#
#  Joe Schoonover, Fluid Numerics LLC (joe@fluidnumerics.com)
#
# Description :
#
#  This script runs an HPC Toolkit workflow for capturing kernel runtime
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
#    salloc -n1 -c 64 --partition=workq --project=$PAWSEY_PROJECT
#
#   Build EXESS-dev using the `build.sh` script in this directory
#
#   ./build.sh
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
# ///////////////////////////////////////////////////////////////////////// #


EXESS="${EXESS_ROOT:-EXESS-dev_cpu_port}"
SPACK="${SPACK_ROOT:-$MYGROUP/spack}"
PERFMODELING="${PERF_ROOT:-$MYGROUP/performance-modeling-tools}"
INPUT="${INPUT_DECK:-inputs/json_inputs/scf/w1.json}"


# Enable spack
source $SPACK_ROOT/share/spack/setup-env.sh

# Activate spack environment
spack env activate -d $PERFMODELING/spack/mulan


# Launch the application with hpcrun
OMP_NUM_THREADS=1 srun --exact \
                       --ntasks=1 \
                       --cpus-per-task=64 \
                       --threads-per-core=1 \
                       --cpu-bind=socket \
                       hpcrun -o ./hpctoolkit-db \
                       $EXESS/exess $EXESS/$INPUT

# Recover the program structure
hpcstruct $EXESS/exess

# Analyze measurements and attribute to source code
# The "/+" syntax for the included source directory
# Indicates that hpcprof should recursively search
# the source code directory.
hpcprof -I $EXESS/src/+ ./hpctoolkit-db
