#include <WarmupGPU.h>

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
#ifdef _OPENMP 
if (itype == GPU_ONLY_KERNEL_LAUNCH) 
    {
        float *a;
        
    }

#elif defined(USEOPENACC)
#else 
    float t1;
    if (itype == GPU_ONLY_KERNEL_LAUNCH) 
    {
        float *a;
        auto mytimer = NewTimer();
        silly_kernel<<<1,1>>>(a);
        LogGPUElapsedTime("KernelLaunchOnly", mytimer);
    }
    if (itype == GPU_ONLY_MEM_ALLOCATE) 
    {
        float *a;
        auto N = 1024;
        auto mytimer = NewTimer();
        gpuMalloc(&a, N*sizeof(float)); 
        gpuFree(a);
        LogGPUElapsedTime("MemAllocOnly", mytimer);
    }
#endif

}

void run_kernel()
{
#ifdef _OPENMP 
#elif defined(USEOPENACC)
#else 
    auto N = 1024;
    float *x, *y, *d_x, *d_y, *out, *d_out;
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));
    out = (float*)malloc(N*sizeof(float));
    gpuMalloc(&d_x, N*sizeof(float)); 
    gpuMalloc(&d_y, N*sizeof(float));
    gpuMalloc(&d_out, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    gpuMemcpy(d_x, x, N*sizeof(float), gpuMemcpyHostToDevice);
    gpuMemcpy(d_y, y, N*sizeof(float), gpuMemcpyHostToDevice);
    ///\todo need to update kernel launch to use something other 
    /// than <<<1,1>>>
    vector_add<<<1,1>>>(d_out, d_x, d_y, N);
    gpuMemcpy(out, d_out, N*sizeof(float), gpuMemcpyDeviceToHost);

    gpuFree(d_x);
    gpuFree(d_y);
    gpuFree(d_out);
    free(x);
    free(y);
    free(out);
#endif
}
