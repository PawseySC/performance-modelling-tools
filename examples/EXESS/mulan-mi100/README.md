# EXESS on Mulan MI100 GPU


The steps below provide a rough outline for profiling EXESS on Mulan.

Allocate resources on a Mulan compute node with
```
salloc -n1 -c 64 --threads-per-core=1 --mem=120G --account=pawsey0007 --gres=gpu:1 --partition=workq  -C rome
```

## Install EXESS

Use the provided [install.sh](./install.sh) script to install the `hip_dev` branch of EXESS under this directory. This install script can be controlled with the following environment variables

* `EXESS_GIT_BRANCH` - This is the branch of EXESS to build. (Default: `hip_dev`)
* `PERF_ROOT` - This is the root directory where the performance-modelling-tools repository can be found on Mulan. (Default: `$MYGROUP/performance-modelling-tools`)
* `JSON` - Path to where  https://github.com/nlohmann/json is set up. (Default: `$MYGROUP/mulan/json`)
* `EXESS_USE_MAGMA` - String (either `True` or `False`) to select whether to use MAGMA (`True`) or RocSolver (`False`). (Default: `False`)
* `EXESS_GPU_BOARD` - Target GPU Board; passed to `-DGPU_BOARD` CMake option. (Default: `MI100`)
* `EXESS_USE_HIP` - String (either `True` or `False`) to select whether to use HIP (`True`) or not (`False`). (Default: `True`)

To build with RocSolver, simply run the script with the default settings,
```
./install.sh
```

To build with MAGMA,
```
export EXESS_USE_MAGMA=True
./install.sh
```

This will install EXESS under `$(pwd)/EXESS-dev_hip_dev`


## Create a hotspot and trace profile
Use the provided [hotspot-trace-profile.sh](./hotspot-trace-profile.sh) script to run EXESS underneath rocprof to create hotspot and trace profiles. This script can be controlled with the following environment variables

* `EXESS_ROOT` - Path (relative to current directory or full path) to where EXESS is installed. The default value assumes you are using the `hip_dev` branch and have installed using the provided [install.sh](./install.sh). (Default: `EXESS-dev_hip_dev`)
* `PERF_ROOT` - This is the root directory where the performance-modelling-tools repository can be found on Mulan. (Default: `$MYGROUP/performance-modelling-tools`)
* `INPUT_DECK` - Path, relative to `EXESS_ROOT`, to the input file for running EXESS. (Default: `inputs/json_inputs_sprint/w1.json`)

Running this script will create a directory that contains EXESS stdout in log files, a metadata file (`info.json`), and rocprof output. The directory is `rocprof_YYYY_mm_dd_HH_MM`, where `YYYY` is the four digit year, `mm` is a two digit month, `dd` is a two digit day, `HH` is a two digit hour, and `MM` is a two digit minute.

EXESS stdout is recorded in `log-N.txt` where `N` is the MPI rank ID; for multiprocess runs, each rank has its own stdout log file.

The metadata file (`info.json`) records useful book-keeping information including git sha, git branch, a timestamp, and who ran the benchmark.

The hotspot profiles can be found under
* `rocprof-N.stats.csv` : Hotspot profile for HIP kernels
* `rocprof-N.hip_stats.csv` : Hotspot profile for HIP API calls

Trace profiles can be found under `rocprof-N.json`. You can open the trace profiles in either chrome tracing (chrome://tracing) or at https://ui.perfetto.dev/


## Events profiling
Once you have created hotspot and trace profiles, you can dig in further into GPU hardware metrics using the included [events-profile.sh](./events-profile.sh). This script can be controlled with the following environment variables

* `EXESS_ROOT` - Path (relative to current directory or full path) to where EXESS is installed. The default value assumes you are using the `hip_dev` branch and have installed using the provided [install.sh](./install.sh). (Default: `EXESS-dev_hip_dev`)
* `PERF_ROOT` - This is the root directory where the performance-modelling-tools repository can be found on Mulan. (Default: `$MYGROUP/performance-modelling-tools`)
* `INPUT_DECK` - Path, relative to `EXESS_ROOT`, to the input file for running EXESS. (Default: `inputs/json_inputs_sprint/w1.json`)
* `KERNEL` - A kernel that you want to get hardware metrics for. This variable must be set prior to running this script and can be one of the kernels listed in the `Name` column of `rocprof-N.stats.csv`.

For example,
```
export KERNEL="genfock::HIP_GPU::GPU_kernel_1_0_0_0(double, unsigned int const*, unsigned int const*, unsigned int const*, unsigned int const*, unsigned int const*, unsigned int const*, double const*, double const*, double const*, double const*, double const*, double const*, double const*, double const*, double const*, double const*, double const*, double const*, double const*, double const*, unsigned int, double const*, double*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, double)"
./events-profile.sh
```
will generate an events profile for the `GPU_kernel_1_0_0_0` method.

Running this script will create a directory that contains EXESS stdout in log files, a metadata file (`info.json`), and rocprof output. The directory is `rocprof_YYYY_mm_dd_HH_MM`, where `YYYY` is the four digit year, `mm` is a two digit month, `dd` is a two digit day, `HH` is a two digit hour, and `MM` is a two digit minute.

Hardware metrics recorded by rocprof can be found under `rocprof-derived-N.csv`

