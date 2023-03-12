# Imager

This directory contains scripts to build and run applications in the BLINK/imager repository.


## Build the code
The `install.sh` scripts will download a fresh copy of the imager repository, set up your environment, and build the code. To use this script, we recommend doing the following

1. Clone the performance modeling tools repository to your group directory on Topaz
```
cd $MYGROUP
git clone git@github.com:PawseySC/performance-modelling-tools.git
cd performance-modelling-tools/examples/blink_imager/topaz-v100
```

2. Set the environment variable for the branch of the imager repository that you want to build. Currently, we recommmend using the `pacerSprint_joe` branch; this contains a necessary patch to `CMakeLists.txt` to build the `cufft_blocks` application.
```
export IMAGER_BRANCH=pacerSprint_joe
```

3. Run the installation script
```
./install.sh
```

This will clone the repository to a new subdirectory `imager_${IMAGER_BRANCH}` and build the code within.

## Run the code

### cufft_blocks

*The `profile.cufft_blocks.sh` script is currently under development*

!!! note
    Make sure that the `IMAGER_BRANCH` environment variable is set to the same value used to build the code


To run the code, simply execute
```
./profile.cufft_blocks.sh
```

This script will

1. Create a new output directory `cufftblocks_profile_$(date +"%Y-%m-%d-%H-%M")`
2. Create symbolic links from the `.fits` files in `imager_${IMAGER_BRANCH}/build_gpu` to the output directory
3. Run the cufft_blocks executable using commands documented in the `imager/apps/cufft_blocks.cu` source code comments.




