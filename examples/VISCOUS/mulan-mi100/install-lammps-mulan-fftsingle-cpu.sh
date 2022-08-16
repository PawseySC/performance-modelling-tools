#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1
#SBATCH --time=01:00:00
#SBATCH --output=out-install-lammps-mulan-fftsingle-cpu

# Lammps with Kokkos for OpenMP
# FFTs: single precision, host KISS

reponame="lammps"
repo="git@github.com:CTCMS-UQ/${reponame}.git"
checkout="sprint-develop"
dir="lammps-mulan-fftsingle-cpu"
group="sprint4"
umask 007

echo BUILD-START $( date )

module unload gcc/9.3.0

rm -fr $dir
mkdir -p $dir
cd $dir

git clone $repo
cd $reponame
git checkout $checkout

cd src
sed -e 's/KOKKOS_DEVICES *=.*/KOKKOS_DEVICES = OpenMP,Serial\
KOKKOS_ARCH = ZEN2/' \
    -e 's/CC *=.*/CC = CC/g' \
    -e 's/LINK *=.*/LINK = CC/g' \
	-e 's/FFT_INC *=.*/FFT_INC = -DFFT_SINGLE/g' \
    -e '/MPI_INC *=/ s;$; -I${MPICH_DIR}/include;g' \
    -e '/MPI_LIB *=/ s;$; -L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa;g' \
    MAKE/OPTIONS/Makefile.kokkos_mpi_only >MAKE/Makefile.mulan_cpu

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

sg $group -c 'make -j 8 mulan_cpu'

echo BUILD-END $( date )
