#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=5 
#SBATCH --cpus-per-task=1 
#SBATCH --threads-per-core=1
#SBATCH --partition=gpuq-dev
#SBATCH --gres=gpu:1 
#SBATCH --mem=16G
#SBATCH -o frac_scale.out
#SBATCH -e frac_scale.out
###SBATCH --project=pawsey0007

CODENAME="TCLB"
REPO="https://github.com/CFD-GO/TCLB.git"
GIT_BRANCH="${GIT_BRANCH:-master}"

cwd=$(pwd)
timestamp=$(date +"%Y-%m-%d-%H-%M")

odir="${cwd}/profile_frac_scale_${GIT_BRANCH}_${timestamp}"
mkdir -p ${odir}

# Load modules
module load cuda/11.4.2 r/4.0.2 openmpi-ucx-gpu/4.0.2

#mpirun -np 1 ${CODENAME}_${GIT_BRANCH}/CLB/d3q27_PSM_NEBB/main test3d.xml
for i in flow_3000_1_1 flow_12000_1_1 flow_27000_1_1
do
   mkdir -p ${odir}/$i

   # Obtain hotspot
   mpirun -np 1 nvprof --csv --log-file ${odir}/$i/nvprof_hotspot.csv \
               frac_scale_test/${GIT_BRANCH}/CLB/d3q27_PSM_NEBB/main data/${i}.xml : \
          -np 4 frac_scale_test/${GIT_BRANCH}/CLB/d3q27_PSM_NEBB/lammps data/$i.lammps

   # Obtain metrics
   mpirun -np 1 nvprof --analysis-metrics -o ${odir}/$i/metrics.nvprof \
                frac_scale_test/${GIT_BRANCH}/CLB/d3q27_PSM_NEBB/main data/${i}.xml : \
          -np 4 frac_scale_test/${GIT_BRANCH}/CLB/d3q27_PSM_NEBB/lammps data/$i.lammps


done
