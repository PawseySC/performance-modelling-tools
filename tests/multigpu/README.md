# Warmup
Runs a few simple instructions for a certain set of rounds and then runs a larger set of device instructions labelled as the run kernel for a certain number of iteraions. 

## Compilation
The test uses a simple Makefile, which might need to be edited for the system it is running on. Different builds for different methods of device offloading are compiled using `buildtype=` flag when making. The different build types are 
- `omp`: OpenMP, no device offloading.
- `cuda`: CUDA build
- `hip`: HIP build 
- `cudaomp`: use Nvidia compiler to enable NVIDIA GPU OpenMP device offload
- `cudahip`: use HIP that has a CUDA backend to build 
- `hipomp`: use HIP compiler to enable NVIDIA GPU OpenMP device offload

> **_NOTE:_** It is recommended to clean before building again if you have switched buildtypes. `make buildtype=${buildtype1}; make buildtype=${buildtype2} clean; make buildtype=${buildtype2};`

## Running 
Compilation will place a binary `warm_up_test.${buildtype}.exe` in the `bin/` directory. Because the code uses the library stored in the `../profile_util/lib` directory, you will need to update the `LD_LIBRARY_PATH`. Running `make` will report how you need to update the path, however, the following will work 

```bash
export LD_LIBRARY_PATH=$(pwd)/../profile_util/lib/:$LD_LIBRARY_PATH
```

The code can accept up to three args
1. Number of rounds of warm-up. Default is 2.
2. Number of iterations of the full kernel to run. Default is 100. 
3. How to run the warm up kernel. Default is 0.
  - 0: run each simple device instruction for N rounds
  - 1: run rounds of kernel instructions going from a kernel launch, alloc, host to device, device to host. 
  - 2: run rounds of kernel instructions: alloc, host to device, device to host, kernel launch.
  - 3: run rounds of kernel instructions: host to device, device to host, kernel launch. alloc.
  - 4: run rounds of kernel instructions: device to host, kernel launch, alloc, host to device.

The code will then report information about what parallelism is present, what devices it sees, how it is running, etc. An example output is 
```
@main L30
Parallel API's
 ========
Running with HIP and found 2
HIP Device Compute Units 120 Max Work Group Size 64 Local Mem Size 65536 Global Mem Size 34342961152
HIP Device Compute Units 120 Max Work Group Size 64 Local Mem Size 65536 Global Mem Size 34342961152

@main L31
Core Binding
 ========
	 On node mi02 :  Core affinity = 0

Code using: HIP

```

## Future
I will add a Python notebook that parses this output for comparison and also added OpenACC and OpenCL tests. 
