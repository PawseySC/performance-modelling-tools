/*! \file main.cpp
    \brief run a test kernel on a GPU to see what the warm up period is 

*/


#include <WarmupGPU.h>
#include <logger.h>
#include <profile_util.h>

#include <string>
#include <iostream>
#include <complex>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <chrono>
#ifdef _OPENMP 
#include <omp.h>
#endif

int main(int argc, char** argv)
{
    Logger logger;
    LogParallelAPI();
    auto runtype = logger.ReportGPUSetup();

    int warm_up_type = GPU_ONLY_KERNEL_LAUNCH;
    int Niter = 100;
    if (argc >= 2) warm_up_type = atoi(argv[1]);
    if (argc >= 3) Niter = atoi(argv[2]);

    // look at warm-up kernel
    auto timeWarmup = NewTimer();
    std::cout<<"Warming up "<<std::endl; 
    warmup_kernel(GPU_ONLY_KERNEL_LAUNCH);
    warmup_kernel(GPU_ONLY_MEM_ALLOCATE);
    warmup_kernel(GPU_ONLY_MEM_TH2D);
    warmup_kernel(GPU_ONLY_MEM_TD2H);
    LogTimeTaken(timeWarmup);

    // run a kernel on all possible devices, report timings
    run_on_devices(logger, Niter);
}
