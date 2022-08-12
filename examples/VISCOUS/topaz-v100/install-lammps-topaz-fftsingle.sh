#!/bin/bash -l
#SBATCH --partition=gpuq-dev
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=out-install-lammps-topaz-fftsingle

# Lammps with Kokkos for Nvidia GPUs
# FFTs: single precision, host FFTW, device cuFFT
# Kokkos FFTs can run on device only

reponame="lammps"
repo="git@github.com:CTCMS-UQ/${reponame}.git"
checkout="sprint-develop"
dir="lammps-topaz-fftsingle"
group="sprint4"
umask 007

echo BUILD-START $( date )

module swap gcc gcc/10.2.0
module load cuda/11.4.2
module load openmpi-ucx-gpu/4.0.2
module load fftw/3.3.8

module load cmake/3.18.0

rm -fr $dir
mkdir -p $dir
cd $dir
mkdir -p app
install_dir="$(pwd)/app"

git clone $repo
cd $reponame
git checkout $checkout

mkdir build
cd build

cmake ../cmake \
 -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
 -DCMAKE_INSTALL_PREFIX="$install_dir" \
 -DCMAKE_CXX_COMPILER="$(pwd)/../lib/kokkos/bin/nvcc_wrapper" \
 -DMPI_CXX_COMPILER="mpicxx" \
 -DCMAKE_C_COMPILER="mpicc" \
 -DCMAKE_Fortran_COMPILER="mpif90" \
 -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-g -O3 -DNDEBUG --default-stream per-thread" \
 -DPKG_CLASS2=ON \
 -DPKG_MANYBODY=ON \
 -DPKG_MISC=ON \
 -DPKG_EXTRA-DUMP=ON \
 -DPKG_EXTRA-FIX=ON \
 -DPKG_KSPACE=ON \
 -DPKG_MOLECULE=ON \
 -DPKG_RIGID=ON \
 -DPKG_MOLFILE=ON \
 -DPKG_UEF=ON \
 -DPKG_MOL-SLLOD=ON \
 -DPKG_KOKKOS=ON \
 -DFFT=FFTW3 \
 -DFFT_SINGLE=True \
 -DKokkos_ARCH_SKX=ON \
 -DKokkos_ARCH_VOLTA70=ON \
 -DKokkos_ENABLE_SERIAL=ON \
 -DKokkos_ENABLE_OPENMP=ON \
 -DKokkos_ENABLE_CUDA=ON \
 -DBUILD_OMP=ON \
 -DBUILD_MPI=ON

sg $group -c 'make -j 8'
sg $group -c 'make install'

echo BUILD-END $( date )
