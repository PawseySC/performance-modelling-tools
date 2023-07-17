#!/bin/bash --login

###SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --account=pawsey0007-gpu
#SBATCH --partition=gpu-dev
#SBATCH --output=outputfile.log
#SBATCH --error=errorfile.log

CODENAME="S3D"
REPO="git@github.com:unsw-edu-au/S3D_JICF.git"
GIT_BRANCH="pacer_cleanup"
ROCM_VERSION="5.0.2"
MACH="STXgpu"
INPUT_DECK="s3d_np1.in"

CHECKOUT_DIR=${CODENAME}_${GIT_BRANCH}_${MACH}_rocm-${ROCM_VERSION}

cwd=$(pwd)
timestamp=$(date +"%Y-%m-%d-%H-%M")
odir="${cwd}/profile_${CHECKOUT_DIR}_${INPUT_DECK}_${timestamp}"
mkdir -p ${odir}

module load PrgEnv-cray rocm/${ROCM_VERSION} craype-accel-amd-gfx90a 

export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MACH

# Copy the input deck to input/s3d.in
cp ${cwd}/${INPUT_DECK} ${cwd}/${CHECKOUT_DIR}/input/s3d.in

cd ${cwd}/${CHECKOUT_DIR}/run
# Straight run of the code
srun -u -n $SLURM_NTASKS -c 8 ./s3d.STXgpu.x

# Run for hotspot profile
srun -u -n $SLURM_NTASKS -c 8 rocprof --stats ./s3d.STXgpu.x
mv results.*.csv ${odir}/

# Run for trace profile
srun -u -n $SLURM_NTASKS -c 8 rocprof --sys-trace ./s3d.STXgpu.x
mv results.json ${odir}/

# Run for events profile
srun -u -n $SLURM_NTASKS -c 8 rocprof -i rocprof_counters.txt -o ${odir}/rocprof_metrics.csv ./s3d.STXgpu.x

