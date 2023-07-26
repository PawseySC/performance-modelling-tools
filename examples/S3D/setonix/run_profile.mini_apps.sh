#!/bin/bash --login
###SBATCH --time=05:00:00
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=8
##SBATCH --gpus-per-node=8
##SBATCH --sockets-per-node=8
##SBATCH --cpus-per-task=8
##SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=pawsey0007-gpu
#SBATCH --reservation=sprint8
#SBATCH --partition=gpu
#SBATCH --output=stdout
#SBATCH --error=stderr


miniapps=( "transport" "chemistry" "cons2prim" )
CODENAME="S3D"
REPO="git@github.com:unsw-edu-au/S3D_JICF.git"
GIT_BRANCH="pacer_mini_apps"
ROCM_VERSION="5.0.2"
MACH="STXgpu"
INPUT_DECK="s3d_np1.in"

CHECKOUT_DIR=${CODENAME}_${GIT_BRANCH}_${MACH}_rocm-${ROCM_VERSION}

cwd=$(pwd)
timestamp=$(date +"%Y-%m-%d-%H-%M")
odir="${cwd}/profile_${CHECKOUT_DIR}_${INPUT_DECK}_${timestamp}"

module load PrgEnv-cray rocm/${ROCM_VERSION} craype-accel-amd-gfx90a 

export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MACH
set -x
for app in "${miniapps[@]}"
do
  echo "##############################"
  echo "#  Profiling $app #"
  echo "##############################"
  cp ${cwd}/rocprof-events.txt ${cwd}/${CHECKOUT_DIR}/source/mini_apps/${app}_kernel/run/
  cd ${cwd}/${CHECKOUT_DIR}/source/mini_apps/${app}_kernel/run
  srun -n 1 -c 8 rocprof -i rocprof-events.txt ./s3d.STXgpu.x
  srun -n 1 -c 8 rocprof -o profile.csv --stats --sys-trace ./s3d.STXgpu.x
  mkdir -p ${odir}/${app}
  mv ${cwd}/${CHECKOUT_DIR}/source/mini_apps/${app}_kernel/run/profile*.csv ${odir}/${app}
  mv ${cwd}/${CHECKOUT_DIR}/source/mini_apps/${app}_kernel/run/profile*.json ${odir}/${app}
  mv ${cwd}/${CHECKOUT_DIR}/source/mini_apps/${app}_kernel/run/profile*.db ${odir}/${app}
  mv ${cwd}/${CHECKOUT_DIR}/source/mini_apps/${app}_kernel/run/profile*.txt ${odir}/${app}
done




# # Chemistry kernel
# cd ${cwd}/${CHECKOUT_DIR}/source/mini_apps/chemistry_kernel/run
# srun -n 1 -c 8 rocprof -i profile-events.txt ./s3d.STXgpu.x
# srun -n 1 -c 8 rocprof -o profile.csv --stats --sys-trace ./s3d.STXgpu.x

# mkdir -p ${odir}/chemistry
# mv profile*.csv ${odir}/chemistry
# mv profile*.json ${odir}/chemistry
# mv profile*.db ${odir}/chemistry
# mv profile*.txt ${odir}/chemistry

# # Cons2Prim kernel
# cd ${cwd}/${CHECKOUT_DIR}/source/mini_apps/cons2prim_kernel/run
# srun -n 1 -c 8 rocprof -i profile-events.txt ./s3d.STXgpu.x
# srun -n 1 -c 8 rocprof -o profile.csv --stats --sys-trace ./s3d.STXgpu.x

# mkdir -p ${odir}/cons2prim
# mv profile*.csv ${odir}/cons2prim
# mv profile*.json ${odir}/cons2prim
# mv profile*.db ${odir}/cons2prim
# mv profile*.txt ${odir}/cons2prim
