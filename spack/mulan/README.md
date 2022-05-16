# Mulan Spack Configuration Files

This directory contains a Spack environment file and compilers configuration for Mulan compute nodes. The environment assumes we need profiling and analysis tools for shared memory (serial and OpenMP) applications.

GPU profiling with hpc-toolkit is currently not supported on Mulan, hpc-toolkit  requires ROCMm 5.0.0 or greater.

You can use this directory with Spack to set up an environment for profiling applications on Mulan.

## Getting Started

1. Start a compute allocation on mulan
```
salloc -n1 -c32 --threads-per-core=1 --partition=workq --project=$PAWSEY_PROJECT --mem=240G
```

2. Clone this repository to your group directory
```
git clone https://github.com/PawseySC/performance-modelling-tools.git $MYGROUP/performance-modelling-tools
```

3. Clone spack to your group directory and enable spack
```
git clone https://github.com/spack/spack $MYGROUP/spack
source $MYGROUP/spack/share/spack/setup-env.sh
```

4. Enable the spack environment
```
spack env activate -d $MYGROUP/performance-modelling-tools/spack/mulan
```

5. Install packages
```
sg $PAWSEY_PROJECT -c 'spack install'
```
