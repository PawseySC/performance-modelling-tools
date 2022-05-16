#!/bin/bash

# to be run on a mi100 compute node

git_branch="cpu_port"
git_checkout="" #821ec2790dd89de71c3aae6e5516fed5a09c6044 #9May2022

case "$HOSTNAME" in
    topaz*)
      module load openmpi-ucx/4.0.2
      module load hdf5/1.12.0-c++-noparallel-api-v112 
      module load cmake
      module load intel-mkl
      export BLAS_LIB="MKL"
      export MPI_ROOT="$MAALI_OPENMPIUCX_HOME"
      export BUILD_CPUS=16
      ;;
    mulan*)
      # TODO test build on mulan
      module load cray-hdf5/1.12.0.6
      export BLAS_LIB="CRAY"
      
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cray/pe/libsci/21.08.1.2/CRAY/9.0/x86_64/lib
      export CPATH=$CPATH:/opt/cray/pe/libsci/21.08.1.2/CRAY/9.0/x86_64/include
      export CRAYPE_LINK_TYPE=dynamic
      
      . /pawsey/mulan/bin/init-cmake-3.21.4.sh
      export MPI_ROOT="$MPICH_DIR"
      export BUILD_CPUS=$SLURM_CPUS_PER_TASK
esac

# To change BLAS library, set the BLAS_LIB variable. Currently 
# supported are CRAY, MKL, OPENBLAS. You'll need to set up 
# paths etc. for your chosen library. To implement extra 
# BLAS libraries you'll need to modify the root CMakeLists.txt 
# in the cpu_port repo
export HDF5_ROOT="$HDF5_DIR"
export JSON_ROOT="$MYSCRATCH/mulan/EXESS/git/json"

if [ ! -e $JSON_ROOT ] ; then
 mkdir -p $(dirname "$JSON_ROOT")
 git clone https://github.com/nlohmann/json $JSON_ROOT	
fi 

if [ ! -e EXESS-dev_$git_branch ] ; then
 git clone git@github.com:EXESS-dev/EXESS-dev.git EXESS-dev_$git_branch
fi

cd EXESS-dev_$git_branch
git checkout $git_branch
git checkout $git_checkout

mkdir build
cd build

cmake .. \
  -DMPI_ROOT=$MPI_ROOT \
  -DHDF5_ROOT=$HDF5_ROOT \
  -DJSON_ROOT=$JSON_ROOT \
  -DBLAS_LIB=$BLAS_LIB \
  -DBUILD_RIMP2=0
make -j $BUILD_CPUS

sg $PAWSEY_PROJECT -c 'make install'

