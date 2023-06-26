#!/bin/bash
#!/bin/bash
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -A pawsey0007-gpu
#SBATCH --gpus-per-task=1
#SBATCH -o run_profile.out
#SBATCH -e run_profile.out

CODENAME="TCLB"
REPO="https://github.com/CFD-GO/TCLB.git"
#GIT_BRANCH="master"
GIT_BRANCH="launch_bounds_64_6"
INPUT_DECK="data/short_0.xml"
ROCM_VERSION="5.0.2"
APP=d3q27_PSM_NEBB

CHECKOUT_DIR=${CODENAME}_${GIT_BRANCH}_rocm-${ROCM_VERSION}

cwd=$(pwd)

cwd=$(pwd)
timestamp=$(date +"%Y-%m-%d-%H-%M")

odir="${cwd}/profile_${CHECKOUT_DIR}_${APP}_${timestamp}"
mkdir -p ${odir}

# Load modules
module load rocm/${ROCM_VERSION}
module load r/4.1.0

# Link data directory if not found
if [ ! -d "data/" ]; then
	echo "Linking input data directory"
	ln -s /scratch/director2188/sprint/frac_scale_test/data ./
fi

# Run the code once without profiling
${CHECKOUT_DIR}/CLB/${APP}/main $INPUT_DECK

# Profile for hot-spot and trace
# The profiler output is moved to ${odir}
# Notable output from rocprof includes
#     > results.json - Trace profile. Can be visualized at https://ui.perfetto.dev
#     > results.stats.csv - Hotspot profile of HIP kernels
#     > results.copy_stats.csv - Hotspot profile summar of memcpy calls
rocprof --stats --sys-trace ${CHECKOUT_DIR}/CLB/${APP}/main $INPUT_DECK
mv results.* ${odir}/


# Profile for events
# Here we pass in rocprof_input.txt to indicate what metrics we want to collect and for what kernels
# The output is in ${odir}/rocprof_metrics.csv
rocprof -i ${cwd}/rocprof_input.txt -o ${odir}/rocprof_metrics.csv ${CHECKOUT_DIR}/CLB/${APP}/main $INPUT_DECK

