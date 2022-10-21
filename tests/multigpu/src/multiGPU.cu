#include <multiGPU.h>
#include <iostream>
#include <cmath>

#if defined(USEHIP) || defined(USECUDA)
/// standard scalar a * vector x plus vector y 
__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}


/// just a vector add to new vector
__global__
void vector_add(float *out, float *a, float *b, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    // Make sure we do not go out of bounds
    if (id < n) out[id] = a[id] + b[id];
}

#endif

inline int GetNumDevices()
{
    int deviceCount = 0;
#ifdef _OPENMP 
    deviceCount = omp_get_num_devices();
#elif defined(_OPENACC)
    auto dtype = acc_get_device_type();
    deviceCount = acc_get_num_devices(dtype);
#elif defined(USEHIP) || defined(USECUDA)
    gpuGetDeviceCount(&deviceCount);
#endif
    return deviceCount;
}

inline void SetDevice(int i)
{
#ifdef _OPENMP 
    omp_set_default_device(i);
#elif defined(_OPENACC)
    auto dtype = acc_get_device_type();
    acc_set_device_num(i,dtype);
#elif defined(USEHIP) || defined(USECUDA)
    gpuSetDevice(i);
#endif
}

void run_on_devices(Logger &logger, int Niter)
{
    std::vector<std::thread> threads;
    int deviceCount = GetNumDevices();
    struct AllTimes {
        std::vector<double> times;
        std::map<std::string, std::vector<double>> device_times;
        std::vector<double> x;
        AllTimes() 
        {
#ifdef _OPENMP 
            device_times.insert({"omp_target",x});
#elif defined(_OPENACC)
            device_times.insert({"acc_target",x});
#elif defined(USEHIP) || defined(USECUDA) 
            device_times.insert({"allocation",x});
            device_times.insert({"tH2D",x});
            device_times.insert({"tD2H",x});
            device_times.insert({"free",x});
            device_times.insert({"kernel",x});
#endif
        }
    };
    std::vector<AllTimes> alltimes(deviceCount);

    for (unsigned int device_id = 0; device_id < deviceCount; device_id++)
    {
        threads.push_back (std::thread ([&,device_id] () 
        {
            SetDevice(device_id);
            for (auto j=0;j<Niter;j++) 
            {
                auto t = NewTimer();
                auto timings = run_kernel(j);
                alltimes[device_id].times.push_back(t.get());
                for (auto &t:timings) 
                {
                    alltimes[device_id].device_times[t.first].push_back(t.second);
                }
                // gpu_data[device_id]->launch<div> ();
                // gpu_data[device_id]->sync ();
            }
        }));
    }
    // join threads having launched stuff on gpus 
    for (auto &thread: threads) thread.join ();

    for (unsigned int i = 0; i < deviceCount; i++)
    {
        std::cout<<"================================="<<std::endl;
        std::cout<<" DEVICE "<<i<<std::endl;
        logger.ReportTimes("run_kernel", alltimes[i].times);
        std::cout<<"---------------------------------"<<std::endl;
        std::cout<<"On device times within run_kernel"<<std::endl;
        for (auto &t:alltimes[i].device_times) logger.ReportTimes(t.first,t.second);
        std::cout<<"---------------------------------"<<std::endl;
    }
}

std::map<std::string, double> run_kernel(int offset)
{
#define gettelapsed(t1)  telapsed = GetTimeTakenOnDevice(t1,__func__, std::to_string(__LINE__));

    std::map<std::string, double> timings;
    float telapsed;
    auto N = 1024*1024;
    float *x, *y, *d_x, *d_y, *out, *d_out;
    x = new float[N];
    y = new float[N];
    out = new float[N];
    for (int i = 0; i < N; i++) 
    {
        x[i] = 1.0*offset;
        y[i] = 2.0*offset;
    }
#ifdef _OPENMP 
    auto tall = NewTimer();
    // auto talloc = NewTimer();
    // auto th2d = NewTimer();
    // auto tk = NewTimer();
    // auto td2h = NewTimer();
    // auto tfree = NewTime();
    #pragma omp target data map(to:x[:N],y[:N]) map(from:out[:N]) 
    {
        #pragma omp target
        #pragma omp parallel for
        for (int i=0;i<N;i++) out[i] = x[i] + y[i];
    }
    // timings.insert({std::string("allocation"), telapsed});
    // timings.insert({std::string("tH2D"), telapsed});
    // timings.insert({std::string("kernel"), telapsed});
    // timings.insert({std::string("tD2H"), telapsed});
    // timings.insert({std::string("free"), telapsed});
    // std::cout<<out[2]<<std::endl;
    // LogTimeTaken(tall);
    telapsed = GetTimeTaken(tall,__func__, std::to_string(__LINE__));
    timings.insert({std::string("omp_target"), telapsed});
#elif defined(_OPENACC)
    auto tall = NewTimer();
    #pragma acc parallel loop copyin(x[:N],y[:N]) copyout(out[:N])
    for (int i=0;i<N;i++) out[i] = x[i] + y[i];
    telapsed = GetTimeTaken(tall,__func__, std::to_string(__LINE__));
    timings.insert({std::string("acc_target"), telapsed});
#elif defined(USEHIP) || defined(USECUDA)
    auto talloc = NewTimer();
#ifdef NOASYNC
    gpuMalloc(&d_x, N*sizeof(float)); 
    gpuMalloc(&d_y, N*sizeof(float));
    gpuMalloc(&d_out, N*sizeof(float));
#else
    gpuHostAlloc(&d_x, N*sizeof(float)); 
    gpuHostAlloc(&d_y, N*sizeof(float));
    gpuHostAlloc(&d_out, N*sizeof(float));
#endif

    gettelapsed(talloc);
    timings.insert({std::string("allocation"), telapsed});
    auto th2d = NewTimer();
#ifdef NOASYNC
    gpuMemcpy(d_x, x, N*sizeof(float), gpuMemcpyHostToDevice);
    gpuMemcpy(d_y, y, N*sizeof(float), gpuMemcpyHostToDevice);
#else 
    gpuMemcpyAsync(d_x, x, N*sizeof(float), gpuMemcpyHostToDevice);
    gpuMemcpyAsync(d_y, y, N*sizeof(float), gpuMemcpyHostToDevice);
#endif
    gettelapsed(th2d);
    timings.insert({std::string("tH2D"), telapsed});
    ///\todo need to update kernel launch to use something other 
    /// than <<<1,1>>>
    auto tk = NewTimer();
    int blockSize, gridSize;
    // Number of threads in each thread block
    blockSize = 1024;
    // Number of thread blocks in grid
    gridSize = static_cast<int>(ceil(static_cast<float>(N)/blockSize));
    // Execute the kernel
    vector_add<<<dim3(gridSize),dim3(blockSize)>>>(d_out, d_x, d_y, N);
    gettelapsed(tk);
    timings.insert({std::string("kernel"), telapsed});
    auto td2h = NewTimer();
    gpuMemcpy(out, d_out, N*sizeof(float), gpuMemcpyDeviceToHost);
    gettelapsed(td2h);
    timings.insert({std::string("tD2H"), telapsed});
    auto tfree = NewTimer();
#ifdef NOASYNC
    gpuFree(d_x);
    gpuFree(d_y);
    gpuFree(d_out);
#else
    gpuHostFree(d_x);
    gpuHostFree(d_y);
    gpuHostFree(d_out);
#endif
    gettelapsed(tfree);
    timings.insert({std::string("free"), telapsed});
#endif
    delete[] x;
    delete[] y;
    delete[] out;
    return timings;
}
