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
#    salloc -n1 -c 32 --threads-per-core=1 --partition=workq --project=$PAWSEY_PROJECT --mem=240G
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
#     EXESS_NTHREADS
# 
# ///////////////////////////////////////////////////////////////////////// #


EXESS="${EXESS_ROOT:-EXESS-dev_cpu_port}"
SPACK="${SPACK_ROOT:-$MYGROUP/spack}"
PERFMODELING="${PERF_ROOT:-$MYGROUP/performance-modelling-tools}"
INPUT="${INPUT_DECK:-inputs/json_inputs_sprint/w15.json}"
NTHREADS="${EXESS_NTHREADS:-1}"
PMT_SIMG="${PMT_SIF:-/group/pawsey0007/jschoonover/containers/pmt_latest.sif}"

# Enable spack
source $SPACK/share/spack/setup-env.sh

# Activate spack environment
spack env activate -d $PERFMODELING/spack/mulan

# Enable singularity

cwd=$(pwd)
hpcdb="hpctoolkit_$(date +"%Y-%m-%d-%H-%M")"
odir="$cwd/$hpcdb"
mkdir $odir
cd $EXESS

# Launch the application with hpcrun
OMP_NUM_THREADS=$NTHREADS srun --exact \
                       --ntasks=1 \
                       --cpus-per-task=$NTHREADS \
                       --threads-per-core=1 \
                       --cpu-bind=socket \
		       hpcrun -o $odir/db \
                       ./exess $INPUT

# Recover the program structure
hpcstruct -o $odir/exess.struct ./exess
#
## Analyze measurements and attribute to source code
## The "/+" syntax for the included source directory
## Indicates that hpcprof should recursively search
## the source code directory.
#
# Additional flags provided according to
# https://hatchet.readthedocs.io/en/latest/data_generation.html#hpctoolkit

srun -n1 hpcprof-mpi --metric-db=yes -I ./src/+ $odir/db
mv hpctoolkit-database $odir/


# Process database to hotspot profile
cd $cwd
singularity exec --bind $(pwd):/workspace $PMT_SIMG \
  python3 /opt/pmt/bin/hpctoolkit-hotspot.py --csv --odir /workspace /workspace/$hpcdb/hpctoolkit-database/



