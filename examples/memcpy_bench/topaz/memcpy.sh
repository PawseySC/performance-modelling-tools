#!/bin/bash
# Author : joe@fluidnumerics.com
# 
# Run this script on Topaz with sbatch --account=$MYACCOUNT ./memcpy.sh
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --sockets-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --time=00:30:00

module load hip/4.3.0 
module load cuda/11.4.2 gcc/11.1.0

export CUDA_PATH=$CUDA_HOME
echo $CUDA_HOME

nvidia-smi topo -m

git clone https://gitub.com/fluidnumerics/scientific-computing-edu
cd scientific-computing-edu/samples/c++/memcpy

make clean
make memcpy_bench


srun ./memcpy_bench
