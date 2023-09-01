# profile_util
Simple C++ library to report memory, timing, thread affinity. 

Library built with several parallel APIs: 
- serial: `libprofile_util.so`
- OpenMP: `libprofile_util_omp.so`
- MPI: `libprofile_utils_mpi.so`
- MPI+OpenMP: `libprofile_util_mpi_omp.so`

## API

### C++ API

Main API are defined as macros. There are calls to report 
- Thread affinity: 
    - `LogParallelAPI`: reports the parallel API's used. 
    - `LogBinding()`: reports the overall binding of cores,
    - `LogThreadAffinity`: reports core affinity of  mpi ranks (if MPI enabled) and openmp threads (if enabled) to standard out. Also reports function and line at which report was requested. For MPI, MPI_COMM_WORLD is used
    - `LoggerThreadAffinity(ostream)`: like `LogThreadAffinity` but output to ostream
    - `MPILogThreadAffinity(comm)`: if MPI enabled, can also provide a specific communicator. Like `LogThreadAffinity`. 
    - `MPILoggerThreadAffinity(ostream,comm)`: like `LoggerThreadAffinity(ostream)` but for specific communicator, like `MPILogThreadAffinity(comm)`.
- Memory usage:
    - `LogMemUsage`: like `LogThreadAffinity` but reports current and peak memory usage to standard out.
    - `LoggerMemUsage(ostream)`: like `LogMemUsage` but to ostream. 
- Timer usage. Does require creating a timer with `auto timer = NewTimer;`
    - `LogTimeTaken(timer)`: reports the time taken from creation of Time to point at which logger called and also reports function and line at creation of timer and when request for time taken. 
    - `LoggerTimeTaken(ostream,timer)`: like `LogTimeTaken(timer)` but to ostream. 
 

### Fortran and C API

The Main API is through extern C functions. 

For C, the name convention follows the C++ expect there is a `_` between words and all are lower case. For examle `LogParallelAPI` -> `log_parallel_api()`.

## Compilation 

# Scripts 

The code can build libraries through scripts 
- `build_cpu.sh`: builds serial, openmp, mpi, and mpi+openmp versions of the library and stores them in `/lib/`, each with a suffix appended to the library name of `<none>`, `omp`, `mpi`, `mpi_omp`.
- `build_cuda.sh`: builds cuda, cuda+openmp, cuda+mpi, and cuda+mpi+openmp versions of the library and stores them in `/lib/`. Follows similar naming convention to cpu only builds.
- `build_hip.sh`: similar to cuda but for hip

# CMake
Can also build the library with `cmake` and the options `-DPU_ENABLE_MPI`, `-DPU_ENABLE_OPENMP`, `-DPU_ENABLE_CUDA`, and `-DPU_ENABLE_HIP`. Builds single library with desired functionality. 
