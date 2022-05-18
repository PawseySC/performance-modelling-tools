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
#  and trace profile. The hotspot profile is post-processed with 
#  models/hotspot/rocprof-hotspot.py
#
#  Trace profiles can be visualized with https://ui.perfetto.dev/
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
#   Build EXESS-dev using the `install.sh` script in this directory
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


EXESS="${EXESS_ROOT:-EXESS-dev_hip_dev}"
PERFMODELING="${PERF_ROOT:-$MYGROUP/performance-modelling-tools}"
INPUT="${INPUT_DECK:-inputs/json_inputs_sprint/w1.json}"


module unload gcc/9.3.0
module load craype-accel-amd-gfx908
module load rocm/4.5.0
module load cray-hdf5/1.12.0.6
. /pawsey/mulan/bin/init-mi100-hipsolver-4.5.0.sh
#. /pawsey/mulan/bin/init-mi100-magma-2.6.2.sh
. /pawsey/mulan/bin/init-cmake-3.21.4.sh

cwd=$(pwd)
odir="$cwd/rocprof_$(date +"%Y-%m-%d-%H-%M")"
mkdir -p $odir
cd $EXESS

# Launch the application with rocprof
OMP_NUM_THREADS=1 srun --exact \
                       --ntasks=2 \
		       --cpus-per-task=1 \
                       --ntasks-per-socket=2 \
                       --threads-per-core=1 \
		       ../rocprof-wrapper.sh $INPUT $odir

#
## Create metadata file with information about this run
cd $cwd
cat <<EOT >> $odir/info.json
{
  "datetime":"$(date +"%Y/%m/%d %H:%M")",
  "user": "$(whoami)",
  "git branch": "$(cd $EXESS && git rev-parse --abbrev-ref HEAD),"
  "git sha": "$(cd $EXESS && git rev-parse HEAD),"
  "system": "$(hostname)",
  "compiler": "",
  "compiler flags": "",
  "slurm allocation flags": "",
  "launch command": "",
  "gpu accelerated": "True",
  "input_file":"$INPUT"
}
EOT
#"input_file":{"name":"$INPUT","content":"$(tr -d '[:space:]' < $EXESS/$INPUT)"}

# TO DO
#python3 $PERFMODELING/models/hotspot/rocprof-hotspot.py --png --csv results.stats.csv
