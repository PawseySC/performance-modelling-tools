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
    auto timesecondround = NewTimer();
    warmup_kernel(warm_up_type);
    LogTimeTaken(timesecondround);


    // now check the kernel launches
    //std::vector<profiling_util::Timer> timers;
    std::vector<double> times;
    std::map<std::string, std::vector<double>> device_times;
    std::vector<double> x;
    device_times.insert({"allocation",x});
    device_times.insert({"tH2D",x});
    device_times.insert({"tD2H",x});
    device_times.insert({"free",x});
    device_times.insert({"kernel",x});
    for (auto i=0;i<Niter;i++) 
    {
        auto t = NewTimer();
        auto timings = run_kernel();
        times.push_back(t.get());
        for (auto &t:timings) 
        {
            device_times[t.first].push_back(t.second);
        }
    }
    Logger logger;
    logger.ReportTimes("run_kernel", times);
    std::cout<<"---------------------------------"<<std::endl;
    std::cout<<"On Device times within run_kernel"<<std::endl;
    for (auto &t:device_times) logger.ReportTimes(t.first,t.second);
    std::cout<<"---------------------------------"<<std::endl;
}
