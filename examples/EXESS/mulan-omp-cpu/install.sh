#!/bin/bash

# to be run on a mi100 compute node

git_branch="cpu_port"
git_checkout="" #821ec2790dd89de71c3aae6e5516fed5a09c6044 #9May2022

module unload gcc/9.3.0
module load craype-accel-amd-gfx908
module load rocm/4.5.0
module load cray-hdf5/1.12.0.6
. /pawsey/mulan/bin/init-mi100-hipsolver-4.5.0.sh

. /pawsey/mulan/bin/init-cmake-3.21.4.sh

export MPI_ROOT="$MPICH_DIR"
export HDF5_ROOT="$HDF5_DIR"
export JSON_ROOT="$MYSCRATCH/mulan/EXESS/git/json"
export MATHLIB_ROOT="$ROCM_PATH"
export HIP_PATH="$ROCM_PATH/hip"
export GPU_BOARD="MI100"


if [ ! -e $JSON_ROOT ] ; then
 mkdir -p $(dirname "$JSON_ROOT")
 git clone https://github.com/nlohmann/json $JSON_ROOT	
fi 

if [ ! -e EXESS-dev$git_branch ] ; then
 git clone git@github.com:EXESS-dev/EXESS-dev.git EXESS-dev$git_branch
fi

cd EXESS-dev$git_branch
git checkout $git_branch
git checkout $git_checkout

mkdir build && cd build


# TO DO : Build instructions for CPU-only build of EXESS-dev
#CXX=hipcc cmake .. \
#  -DMPI_ROOT=$MPI_ROOT \
#  -DHDF5_ROOT=$HDF5_ROOT \
#  -DJSON_ROOT=$JSON_ROOT \
#  -DMATHLIB_ROOT=$MATHLIB_ROOT \
#  -DHIP=True \
#  -DBUILD_RIMP2=0 \
#  -DMAGMA=False \
#  -DGPU_BOARD=$GPU_BOARD

make -j $SLURM_CPUS_PER_TASK

sg $PAWSEY_PROJECT -c 'make install'

