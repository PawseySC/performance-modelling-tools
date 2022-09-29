#include <WarmupGPU.h>
#include <iostream>
#include <cmath>

#ifndef _OPENMP 
__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

__global__
void vector_add(float *out, float *a, float *b, int n)
{
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

__global__
void silly_kernel(float *a)
{
    for (int i = 0; i < 2; i++) {
        a[i] + 2*a[i];
    }
}
#endif

void warmup_kernel(int itype)
{
    int deviceCount = 0;
    float t1;
    float *a, *b;
    unsigned long long N = 1024*1024;
    std::string kernel_type;

#ifdef _OPENMP 
    // gpuGetDeviceCount(&deviceCount);
    // std::cout<<deviceCount<<std::endl;
    deviceCount = omp_get_num_devices();
    // std::cout<<deviceCount<<std::endl;
    // deviceCount=2;
    for (auto i=0;i<deviceCount;i++) 
    {
        // set the device 
        omp_set_default_device(i);
        for (auto j=0;j<2;j++) 
        {
            if (itype == GPU_ONLY_KERNEL_LAUNCH)
            {
                kernel_type = "KernelLaunchOnly";
                auto mytimer = NewTimer();
                #pragma omp target
                {
                    for (int i = 0; i < 2; i++) {a[i] + 2*a[i];}
                }
                // N = 2000000000;
                // std::cout<<"memory "<<N*sizeof(float)/1024./1024./1024.<<std::endl;
                // a = new float[N];
                // #pragma omp target map(tofrom:a[:N])
                // {
                //     for (int i = 0; i < N; i++) {
                //         a[i] = 1.0;
                //         a[i] = a[i] + 2*a[i];
                //         a[i] = exp(-sqrt(a[i]));
                //     }
                // }
                std::cout<<kernel_type + " on device " + std::to_string(i) + " round " + std::to_string(j);
                LogTimeTaken(mytimer);
            }
            if (itype == GPU_ONLY_MEM_ALLOCATE) 
            {
                kernel_type = "MemAllocOnly";
                auto mytimer = NewTimer();
                // auto a_d = omp_target_alloc(N*sizeof(float), i);
                // omp_target_free(a_d, i);
                #pragma omp target data map(alloc:a[:N])
                {
                }
                std::cout<<kernel_type + " on device " + std::to_string(i) + " round " + std::to_string(j);
                LogTimeTaken(mytimer);
            }
            else if (itype == GPU_ONLY_MEM_TH2D) 
            {
                kernel_type = "tH2D";
                a = new float[N];
                // LogMemUsage();
                auto mytimer = NewTimer();
                #pragma omp target data map(to:a[:N])
                {
                    
                }
                std::cout<<kernel_type + " on device " + std::to_string(i) + " round " + std::to_string(j);
                LogTimeTaken(mytimer);
                delete[] a;
                // LogMemUsage();
            }
            // transfer from device to host
            else if (itype == GPU_ONLY_MEM_TD2H) 
            {
                kernel_type = "tD2H";
                a = new float[N];
                // LogMemUsage();
                auto mytimer = NewTimer();
                #pragma omp target data map(from:a[:N])
                {

                }
                std::cout<<kernel_type + " on device " + std::to_string(i) + " round " + std::to_string(j);
                LogTimeTaken(mytimer);
                delete[] a;
                // LogMemUsage();
            }
        }
    }
#elif defined(USEOPENACC)
#else 
    gpuGetDeviceCount(&deviceCount);
    {
        // weirdly the order in which devices are accessed seems to affect
        // timings 
        // for (auto i=deviceCount-1;i>=0;i--) 
        for (auto i=0;i<deviceCount;i++) 
        {
            // set the device 
            gpuSetDevice(i);
            // do two rounds of the kernel
            for (auto j=0;j<2;j++) 
            {
                // here to a minimal kernel launch
                if (itype == GPU_ONLY_KERNEL_LAUNCH) 
                {
                    kernel_type = "KernelLaunchOnly";
                    auto mytimer = NewTimer();
                    silly_kernel<<<1,1>>>(a);
                    LogGPUElapsedTime(kernel_type + " on device " + std::to_string(i) + " round " + std::to_string(j), mytimer);
                }
                // allocate a free 
                else if (itype == GPU_ONLY_MEM_ALLOCATE) 
                {
                    kernel_type = "MemAllocOnly";
                    auto mytimer = NewTimer();
                    gpuMalloc(&a, N*sizeof(float)); 
                    gpuFree(a);
                    LogGPUElapsedTime(kernel_type + " on device " + std::to_string(i) + " round " + std::to_string(j), mytimer);
                }
                // transfer from host to device
                else if (itype == GPU_ONLY_MEM_TH2D) 
                {
                    kernel_type = "tH2D";
                    a = new float[N];
                    gpuMalloc(&b, N*sizeof(float)); 
                    auto mytimer = NewTimer();
                    gpuMemcpy(b, a, N*sizeof(float), gpuMemcpyHostToDevice);
                    LogGPUElapsedTime(kernel_type + " on device " + std::to_string(i) + " round " + std::to_string(j), mytimer);
                    gpuFree(b);
                    delete[] a;
                }
                // transfer from device to host
                else if (itype == GPU_ONLY_MEM_TD2H) 
                {
                    kernel_type = "tD2H";
                    a = new float[N];
                    gpuMalloc(&b, N*sizeof(float)); 
                    auto mytimer = NewTimer();
                    gpuMemcpy(a, b, N*sizeof(float), gpuMemcpyDeviceToHost);
                    LogGPUElapsedTime(kernel_type + " on device " + std::to_string(i) + " round " + std::to_string(j), mytimer);
                    gpuFree(b);
                    delete[] a;
                }
            }
        }
    }
#endif

}

std::map<std::string, double> run_kernel()
{
#define gettelapsed(t1)  telapsed = GetTimeTakenOnDevice(t1,__func__, std::to_string(__LINE__));

    std::map<std::string, double> timings;
    float telapsed;
#ifdef _OPENMP 
#elif defined(USEOPENACC)
#elif defined(USEHIP) || defined(USECUDA)
    auto N = 1024;
    float *x, *y, *d_x, *d_y, *out, *d_out;
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));
    out = (float*)malloc(N*sizeof(float));
    auto talloc = NewTimer();
    gpuMalloc(&d_x, N*sizeof(float)); 
    gpuMalloc(&d_y, N*sizeof(float));
    gpuMalloc(&d_out, N*sizeof(float));
    gettelapsed(talloc);
    timings.insert({std::string("allocation"), telapsed});
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    auto th2d = NewTimer();
    gpuMemcpy(d_x, x, N*sizeof(float), gpuMemcpyHostToDevice);
    gpuMemcpy(d_y, y, N*sizeof(float), gpuMemcpyHostToDevice);
    gettelapsed(th2d);
    timings.insert({std::string("tH2D"), telapsed});
    ///\todo need to update kernel launch to use something other 
    /// than <<<1,1>>>
    auto tk = NewTimer();
    vector_add<<<1,1>>>(d_out, d_x, d_y, N);
    gettelapsed(tk);
    timings.insert({std::string("kernel"), telapsed});

    auto td2h = NewTimer();
    gpuMemcpy(out, d_out, N*sizeof(float), gpuMemcpyDeviceToHost);
    gettelapsed(td2h);
    timings.insert({std::string("tD2H"), telapsed});

    auto tfree = NewTimer();
    gpuFree(d_x);
    gpuFree(d_y);
    gpuFree(d_out);
    gettelapsed(tfree);
    timings.insert({std::string("free"), telapsed});

    free(x);    
    free(y);
    free(out);
#endif
    return timings;
}
