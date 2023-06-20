#!/bin/bash
#SBATCH -n1 
#SBATCH -c 8 
#SBATCH --threads-per-core=1
#SBATCH --partition=gpuq-dev
#SBATCH --gres=gpu:1 
#SBATCH -o test3d.out
#SBATCH -e test3d.err
###SBATCH --project=pawsey0007

CODENAME="TCLB"
REPO="https://github.com/CFD-GO/TCLB.git"
GIT_BRANCH="${GIT_BRANCH:-master}"

cwd=$(pwd)
timestamp=$(date +"%Y-%m-%d-%H-%M")

odir="${cwd}/profile_test3d_${GIT_BRANCH}_${timestamp}"
mkdir -p ${odir}

# Load modules
module load cuda/11.4.2 r/4.0.2 openmpi-ucx-gpu/4.0.2

#mpirun -np 1 ${CODENAME}_${GIT_BRANCH}/CLB/d3q27_PSM_NEBB/main test3d.xml
nvprof --csv --log-file ${odir}/nvprof_hotspot.csv ${CODENAME}_${GIT_BRANCH}/CLB/d3q27_PSM_NEBB/main test3d.xml

nvprof --analysis-metrics  -o ${odir}/metrics.nvprof ${CODENAME}_${GIT_BRANCH}/CLB/d3q27_PSM_NEBB/main test3d.xml

