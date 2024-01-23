# Oceananigans on GCP
This example is included to show how to create an auto-scaling cluster on Google Cloud Platform that provides access to V100 and A100 GPUs. This is particularly useful for the porting work, where Oceananigans.jl originally only functioned for Nvidia GPUs.


## Creating a cluster

## Running the examples

## Notes

### Installation warnings
Noted on 1/23/2023 with CUDA 12.2.128, Julia 1.10.0 on CentOS-7 deployment of Slurm-GCP v5. Although the warnings below were displayed to stdout, there seems to be no impact on the runtime correctness of Oceananigans.


At the end of dependency instantiation, the following warnings were shown on the Slurm-GCP V100 compute node
```
┌ CUDA → SpecialFunctionsExt
│  ┌ Warning: CUDA runtime library libcublasLt.so.12 was loaded from a system path. This may cause errors.
│  │ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
│  └ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
│  ┌ Warning: CUDA runtime library libnvJitLink.so.12 was loaded from a system path. This may cause errors.
│  │ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
│  └ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
│  ┌ Warning: CUDA runtime library libcusparse.so.12 was loaded from a system path. This may cause errors.
│  │ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
│  └ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
└  
┌ OceanScalingTests
│  [Output was shown above]
└  
┌ MPI → CUDAExt
│  ┌ Warning: CUDA runtime library libcublasLt.so.12 was loaded from a system path. This may cause errors.
│  │ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
│  └ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
│  ┌ Warning: CUDA runtime library libnvJitLink.so.12 was loaded from a system path. This may cause errors.
│  │ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
│  └ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
│  ┌ Warning: CUDA runtime library libcusparse.so.12 was loaded from a system path. This may cause errors.
│  │ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
│  └ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
└  
┌ Oceananigans
│  ┌ Warning: CUDA runtime library libcublasLt.so.12 was loaded from a system path. This may cause errors.
│  │ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
│  └ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
│  ┌ Warning: CUDA runtime library libnvJitLink.so.12 was loaded from a system path. This may cause errors.
│  │ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
│  └ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
│  ┌ Warning: CUDA runtime library libcusparse.so.12 was loaded from a system path. This may cause errors.
│  │ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
│  └ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
```


Also, when running the double drake experiment

```
[joe@rcc-v100-ghpc-0 OceanScalingTests.jl]$ ./run_double_drake.sh 
┌ Warning: CUDA runtime library libcublasLt.so.12 was loaded from a system path. This may cause errors.
│ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
└ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
┌ Warning: CUDA runtime library libnvJitLink.so.12 was loaded from a system path. This may cause errors.
│ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
└ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
┌ Warning: CUDA runtime library libcusparse.so.12 was loaded from a system path. This may cause errors.
│ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
└ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
┌ Warning: CUDA runtime library libcublasLt.so.12 was loaded from a system path. This may cause errors.
│ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
└ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
┌ Warning: CUDA runtime library libnvJitLink.so.12 was loaded from a system path. This may cause errors.
│ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
└ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
┌ Warning: CUDA runtime library libcusparse.so.12 was loaded from a system path. This may cause errors.
│ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
└ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
┌ Warning: CUDA runtime library libcublasLt.so.12 was loaded from a system path. This may cause errors.
│ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
└ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
┌ Warning: CUDA runtime library libnvJitLink.so.12 was loaded from a system path. This may cause errors.
│ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
└ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
┌ Warning: CUDA runtime library libcusparse.so.12 was loaded from a system path. This may cause errors.
│ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
└ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
┌ Warning: CUDA runtime library libcublasLt.so.12 was loaded from a system path. This may cause errors.
│ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
└ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
┌ Warning: CUDA runtime library libnvJitLink.so.12 was loaded from a system path. This may cause errors.
│ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
└ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
┌ Warning: CUDA runtime library libcusparse.so.12 was loaded from a system path. This may cause errors.
│ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
└ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
WARNING: could not import TurbulenceClosures.calculate_diffusivities! into OceanScalingTests
WARNING: Method definition set!(Any, AbstractString) in module OutputWriters at /home/joe/.julia/packages/Oceananigans/rRavs/src/OutputWriters/checkpointer.jl:201 overwritten in module OceanScalingTests at /home/joe/OceanScalingTests.jl/src/simulation_outputs.jl:42.
ERROR: Method overwriting is not permitted during Module precompilation. Use `__precompile__(false)` to opt-out of precompilation.
┌ Warning: CUDA runtime library libcublasLt.so.12 was loaded from a system path. This may cause errors.
│ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
└ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
┌ Warning: CUDA runtime library libnvJitLink.so.12 was loaded from a system path. This may cause errors.
│ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
└ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
┌ Warning: CUDA runtime library libcusparse.so.12 was loaded from a system path. This may cause errors.
│ Ensure that you have not set the LD_LIBRARY_PATH environment variable, or that it does not contain paths to CUDA libraries.
└ @ CUDA ~/.julia/packages/CUDA/rXson/src/initialization.jl:189
WARNING: could not import TurbulenceClosures.calculate_diffusivities! into OceanScalingTests
```