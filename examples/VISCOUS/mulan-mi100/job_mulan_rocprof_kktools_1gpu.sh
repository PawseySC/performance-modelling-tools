#!/bin/bash -l
#SBATCH --job-name=lmp_benchm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --threads-per-core=1
#SBATCH --time=01:00:00

module unload gcc/9.3.0
module load craype-accel-amd-gfx908
module load rocm/4.5.0
export LIBRARY_PATH="/opt/rocm-4.5.0/hipfft/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/rocm-4.5.0/hipfft/lib:$LD_LIBRARY_PATH"
export CPATH="/opt/rocm-4.5.0/hipfft/include:$CPATH"
lmp_dir="<installation-dir>" ### EDIT ME ###
lmp="$lmp_dir/lammps/src/lmp_mulan_mi100"

tools_dir="/group/sprint4/common/kokkos-tools-mulan"
export KOKKOS_PROFILE_LIBRARY="${tools_dir}/kp_roctx_connector.so"



NUM_GPUS=1
echo using executable "$lmp"
echo using $SLURM_JOB_NUM_NODES nodes
echo using $SLURM_NTASKS tasks
echo using $NUM_GPUS GPUs per node
export OMP_NUM_THREADS=1

echo starting lammps at $(date)

rocprof --stats --sys-trace --roctx-trace -o rocprof.csv \
  srun $lmp -sf kk -k on g $NUM_GPUS -in lammps.inp -log lammps.out \
  >rocprof.log

echo finishing lammps at $(date)

exit
