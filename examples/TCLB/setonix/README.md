# TCLB (Setonix)

This subdirectory contains scripts for building and profiling a simple run with TCLB on Setonix.

## Contents

* `install.sh` - A simple build script that downloads TCLB and builds the `d3q27_PSM_NEBB` model with ROCm 5.0.2 on Setonix
* `test.sh` - A simple script that runs the `d3q27_PSM_NEBB` model with the included `test3d.xml` input deck. The application is profiled to obtain a hotspot profile, trace profile, andan events profile.
* `rocprof_input.txt` - A simple input file for rocprof events profiling that captures loads and stores between HBM2 memory and L2 Cache (`FETCH_SIZE`, `WRITE_SIZE` [in KB]), and the L2 Cache hit percentage. This file is currently configured for profiling kernels named `RunBorderKernel`.


## Profiling Resources

* [ORNL Crusher Quick Start - Profiling with rocprof](https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html#getting-started-with-the-rocm-profiler)
* [AMD ROCProf Documentation](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/rocprof.html)
