#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1
#SBATCH --time=01:00:00
#SBATCH --output=out-install-lammps-mulan-fftdouble

# Lammps with Kokkos for AMD GPUs
# FFTs: double precision, host KISS, device hipFFT
# Kokkos FFTs can run on device only

reponame="lammps"
repo="git@github.com:CTCMS-UQ/${reponame}.git"
checkout="sprint-develop"
dir="lammps-mulan-fftdouble"
group="sprint4"
umask 007

echo BUILD-START $( date )

module unload gcc/9.3.0
module load craype-accel-amd-gfx908
module load rocm/4.5.0
export LIBRARY_PATH="/opt/rocm-4.5.0/hipfft/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/rocm-4.5.0/hipfft/lib:$LD_LIBRARY_PATH"
export CPATH="/opt/rocm-4.5.0/hipfft/include:$CPATH"

rm -fr $dir
mkdir -p $dir
cd $dir

git clone $repo
cd $reponame
git checkout $checkout

cd src
sed -e 's/KOKKOS_DEVICES *=.*/KOKKOS_DEVICES = Hip,OpenMP,Serial\
KOKKOS_ARCH = ZEN2,VEGA908/' \
    -e 's/CC *=.*/CC = hipcc/g' \
    -e 's/LINK *=.*/LINK = hipcc/g' \
    -e '/^ *CCFLAGS *=/ s/$/ -munsafe-fp-atomics -DKOKKOS_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS/g' \
	-e 's/FFT_INC *=.*/FFT_INC = -DFFT_HIPFFT/g' \
    -e 's/FFT_LIB *=.*/FFT_LIB = -lhipfft/g' \
    -e '/MPI_INC *=/ s;$; -I${MPICH_DIR}/include;g' \
    -e '/MPI_LIB *=/ s;$; -L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa;g' \
    MAKE/OPTIONS/Makefile.kokkos_mpi_only >MAKE/Makefile.mulan_mi100
# Cuda only
#    -e '/CCFLAGS *=/ s/$/ --default-stream per-thread/' \

# issue with finding libatomic at link stage
sed -i '/KOKKOS_LIBS *+= *-latomic/ s/^/#/g' ../lib/kokkos/Makefile.kokkos

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

sg $group -c 'make -j 8 mulan_mi100'

echo BUILD-END $( date )
