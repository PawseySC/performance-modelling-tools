#!/bin/bash
#!/bin/bash
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -A pawsey0007-gpu
#SBATCH --gpus-per-task=1
#SBATCH -o test3d.out
#SBATCH -e test3d.err

CODENAME="TCLB"
REPO="https://github.com/CFD-GO/TCLB.git"
GIT_BRANCH="${GIT_BRANCH:-master}"

cwd=$(pwd)
timestamp=$(date +"%Y-%m-%d-%H-%M")

odir="${cwd}/profile_test3d_${GIT_BRANCH}_${timestamp}"
mkdir -p ${odir}

# Load modules
module swap PrgEnv-gnu PrgEnv-cray
module load rocm/5.0.2
module load r

# Profile for hot-spot and trace
rocprof --stats --sys-trace ${CODENAME}_${GIT_BRANCH}/CLB/d3q27_PSM_NEBB/main test3d.xml
mv results.* ${odir}/

# Profile for events
rocprof -i ${cwd}/rocprof_input.txt -o ${odir}/rocprof_metrics.csv ${CODENAME}_${GIT_BRANCH}/CLB/d3q27_PSM_NEBB/main test3d.xml

