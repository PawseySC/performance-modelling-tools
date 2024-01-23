#!/bin/bash
#SBATCH -N1 
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --partition=v100
#SBATCH -J double_drake
#SBATCH -o double_drake.out
#SBATCH -e double_drake.out
# ====================================================== #
#
# These settings are used for configured the double_drake
# simulation at 1/3 degree resolution
#
# ====================================================== #
# ================ USER SPECIFIED INPUTS =============== #
# ====================================================== #

# Grid size
# (RESOLUTION * 360 in x and RESOLUTION * 150 in y and Nz in z)
export RESOLUTION=3 
export NZ=50

# Experiment details
export EXPERIMENT="DoubleDrake"
export PRECISION="Float64"

# Might increase performance when using a lot of cores (i.e. improves scalability)
# Matters only for the RealisticOcean experiment
export LOADBALANCE=0

# Do we want to profile or run a simulation?
export PROFILE=1

# How many nodes are we running on?
export NNODES=2

# Restart from interpolated fields ? "" : "numer_of_iteration_to_restart_from"
export RESTART=""

# Server specific enviromental variables and modules
# Edit depending on the system 
# source satori/setup_satori.sh
export JULIA_CUDA_MEMORY_POOL=none
export JULIA=julia

# Profile specific variable
export JULIA_NVTX_CALLBACKS=gc

# Run simulation once without profile
julia --project --check-bounds=no experiments/run.jl

# Run simulation and create profile
nsys profile --trace=nvtx,cuda --output=./lowres julia --project --check-bounds=no experiments/run.jl