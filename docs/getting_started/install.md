# Installation

The performance-modelling-tools repository hosts Spack environment files for Pawsey systems that provide tools that will support performance modeling.

If you plan on using the [HPC Toolkit](https://hpctoolkit.org) from the United States Department of Energy, we first recommend that you install Spack,  if you don't already have an installation available. Good practice is to install spack in our `$MYGROUP` directory on Pawsey systems.

```
git clone https://github.com/spack/spack $MYGROUP/spack
source $MYGROUP/spack/share/spack/setup-env.sh
```


## Mulan
GPU profiling with hpc-toolkit is currently not supported on Mulan; hpc-toolkit requires ROCm 5.0.0 or greater.
For GPU profiling, we recommend using `rocprof`

1. Start a compute allocation on mulan
```
salloc -n1 -c32 --threads-per-core=1 --partition=workq --project=$PAWSEY_PROJECT --mem=240G
```

2. Clone this repository to your group directory, if you have not done so already.
```
git clone https://github.com/PawseySC/performance-modelling-tools.git $MYGROUP/performance-modelling-tools
```

3. Clone spack to your group directory and enable spack, if you have not done so already.
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

The installation process takes about 35-45 minutes
