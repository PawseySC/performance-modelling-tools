#!/bin/bash -l
#SBATCH --job-name=lmp_benchm
#SBATCH --partition=gpuq-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

module swap gcc gcc/10.2.0
module load cuda/11.4.2
module load openmpi-ucx-gpu/4.0.2
module load fftw/3.3.8
lmp_dir="<installation-dir>" ### EDIT ME ###
lmp="$lmp_dir/app/bin/lmp"

tools_dir="/group/sprint4/common/kokkos-tools-topaz"
export KOKKOS_PROFILE_LIBRARY="${tools_dir}/kp_nvprof_connector.so"


NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo using executable "$lmp"
echo using $SLURM_JOB_NUM_NODES nodes
echo using $SLURM_NTASKS tasks
echo using $NUM_GPUS GPUs per node
export OMP_NUM_THREADS=1

echo starting lammps at $(date)

ncu --nvtx --nvtx-include "N9LAMMPS_NS18PairComputeFunctorINS_23PairLJCutCoulLongKokkosIN6Kokkos4CudaEEELi1ELb1ENS_13CoulLongTableILi1EEEEE/" \
  $lmp -sf kk -k on g $NUM_GPUS -in lammps.inp -log lammps.out \
  >ncu_kernel1.log

echo finishing lammps at $(date)

exit
