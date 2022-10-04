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

    int Niter = 100;
    int rounds = 2;
    int warm_up_type = 0;
    if (argc >= 2) rounds = atoi(argv[1]);
    if (argc >= 3) warm_up_type = atoi(argv[2]);
    if (argc >= 4) Niter = atoi(argv[3]);

    // look at warm-up kernel
    auto timeWarmup = NewTimer();
    std::cout<<"Warming up "<<std::endl; 
    if (warm_up_type == 0) warmup_kernel_over_rounds(rounds);
    else warmup_kernel_over_kernels(rounds);
    LogTimeTaken(timeWarmup);

    // run a kernel on all possible devices, report timings
    run_on_devices(logger, Niter);
}
