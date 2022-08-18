#!/bin/bash -l
#SBATCH --partition=gpuq-dev
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=out-install-lammps-topaz-fftdouble-fft-on-cpu

# Lammps with Kokkos for Nvidia GPUs
# FFTs: double precision, host FFTW, NO device
# All FFTs can run on host only

reponame="lammps"
repo="git@github.com:CTCMS-UQ/${reponame}.git"
checkout="sprint-develop"
dir="lammps-topaz-fftdouble-fft-on-cpu"
group="sprint4"
umask 007

echo BUILD-START $( date )

module swap gcc gcc/10.2.0
module load cuda/11.4.2
module load openmpi-ucx-gpu/4.0.2
module load fftw/3.3.8

rm -fr $dir
mkdir -p $dir
cd $dir
mkdir -p app
install_dir="$(pwd)/app"

git clone $repo
cd $reponame
git checkout $checkout

cd src

sed -e 's/KOKKOS_DEVICES *= *Cuda/KOKKOS_DEVICES = Cuda,OpenMP,Serial/' \
    -e 's/KOKKOS_ARCH *= *Kepler35/KOKKOS_ARCH = SKX,VOLTA70/' \
    -e 's/FFT_INC *=.*/FFT_INC = -DFFT_FFTW3/' \
    -e 's/FFT_LIB *=.*/FFT_LIB = -lfftw3/' \
    -e '/CCFLAGS *=/ s/$/ -DNDEBUG --default-stream per-thread/' \
    MAKE/OPTIONS/Makefile.kokkos_cuda_mpi >MAKE/Makefile.topaz_volta

#make no-all
make yes-CLASS2
make yes-MANYBODY
make yes-MISC
make yes-EXTRA-COMPUTE
make yes-EXTRA-DUMP
make yes-EXTRA-FIX
make yes-KSPACE
make yes-MOLECULE
make yes-RIGID
make yes-MOLFILE
make yes-UEF
make yes-MOL-SLLOD
make yes-KOKKOS

sg $group -c 'make -j 8 topaz_volta'

echo BUILD-END $( date )
