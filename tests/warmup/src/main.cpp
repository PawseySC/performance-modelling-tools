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
#include <algorithm>
#include <string>
#include <chrono>
#ifdef _OPENMP 
#include <omp.h>
#endif

int main()
{
    LogParallelAPI();

    int warm_up_type = GPU_ONLY_KERNEL_LAUNCH;
    int Niter = 100;

    // look at warm-up kernel    
    auto timeWarmup = NewTimer();
    // may want to add explicit tracing
    std::cout<<"Warming up "<<std::endl; 
    warmup_kernel(warm_up_type);
    LogTimeTaken(timeWarmup);
    // second round     
    std::cout<<"second round "<<std::endl; 
    timeWarmup = NewTimer();
    warmup_kernel(warm_up_type);
    LogTimeTaken(timeWarmup);


    // now check the kernel launches
    //std::vector<profiling_util::Timer> timers;
    std::vector<double> times;
    for (auto i=0;i<Niter;i++) 
    {
        auto t = NewTimer();
        run_kernel();
        times.push_back(t.get());
    }
    Logger logger;
    logger.ReportTimes("Run_kernel", times);
}
