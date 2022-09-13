#!/bin/bash
# Author : joe@fluidnumerics.com
# 
# Run this script on Mulan with sbatch --account=$MYACCOUNT ./memcpy.sh
#
#SBATCH --cpus-per-task=32
#SBATCH --sockets-per-node=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=workq
#SBATCH --time=00:10:00

module load rocm/4.5.0 gcc/11.2.0

git clone https://gitub.com/fluidnumerics/scientific-computing-edu
cd scientific-computing-edu/samples/c++/memcpy

make clean
make memcpy_bench

srun ./memcpy_bench
