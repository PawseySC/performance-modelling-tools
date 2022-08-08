#!/bin/bash -l
#SBATCH --partition=gpuq-dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=out-%x

checkout="c901382c4c76b108e4e6d190e9236848dc764526" # 8 August 2022
dir="kokkos-tools-topaz"
umask 007

# just to be consistent with the rest of the relevant tools/application
module load cuda/11.4.2
module load openmpi-ucx-gpu/4.0.2


rm -fr $dir
git clone https://github.com/kokkos/kokkos-tools.git $dir

cd $dir
git checkout $checkout

rm -f kp_json_writer kp_reader kp_*.so probes.o profiling/systemtap-connector/probes.h

sed -i '/papi-connector/ s/^make/#make/' build-all.sh 
sed -i '/DUSE_MPI=0/ s/^#CFLAGS/CFLAGS/' profiling/space-time-stack/Makefile
sed -i '/DUSE_MPI=0/ s/^#CFLAGS/CFLAGS/' profiling/chrome-tracing/Makefile

bash build-all.sh
