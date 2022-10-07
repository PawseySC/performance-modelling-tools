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

/// @brief Run the program
/// @param argc number of args passed
/// @param argv pass the number of rounds, how the warm-up test should be run [0 is over rounds]
/// @return return error 
int main(int argc, char** argv)
{
    Logger logger;
    LogParallelAPI();
    LogBinding();
    auto runtype = logger.ReportGPUSetup();
    int Niter = 100;
    int rounds = 2;
    int warm_up_type = 0;
    if (argc >= 2) rounds = atoi(argv[1]);
    if (argc >= 3) warm_up_type = atoi(argv[2]);
    if (argc >= 4) Niter = atoi(argv[3]);


    // report setup 
    std::cout<<"@"<<__func__<<" L"<<__LINE__<<"  currently running :"<<std::endl;
    std::cout<<rounds<<" rounds of warmup"<<std::endl;
    if (warm_up_type == 0) 
    std::cout<<"Warming up by running each type of device instruction for the number of rounds indicated"<<std::endl;
    else  
    std::cout<<"Warming up by running each type of device instruction one after the other all for the number of rounds indicated"<<std::endl;
    std::cout<<Niter<<" iterations of  the vector add run_kernel"<<std::endl;

    // look at warm-up kernel
    if (warm_up_type == 0) warmup_kernel_over_rounds(rounds);
    else warmup_kernel_over_kernels(rounds);

    // run a kernel on all possible devices, report timings
    run_on_devices(logger, Niter);

    return 0;
}
