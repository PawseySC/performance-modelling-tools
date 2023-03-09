#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>


// Borrowed from https://stackoverflow.com/questions/41572751/cuda-atomicadd-with-floats

#ifndef SIZE
#define SIZE 128  // Defines the amount of data in the array
#endif

#ifndef BLOCK_X
#define BLOCK_X 128 // Defines the block size for the GPU kernel launch (the number of threads per block)
#endif


cudaError_t reductionWithCuda(float *result, float *input);
__global__ void reductionKernel(float *result, float *input);
void reductionCPU(float *result, float *input);

int main()
{
    int i;
    float *input;
    float resultCPU, resultGPU;
    double cpuTime, cpuBandwidth;

    input = (float*)malloc(SIZE * sizeof(float));
    resultCPU = 0;
    resultGPU = 0;

    srand((int)time(NULL));

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    for (i = 0; i < SIZE; i++)
    {
        input[i] = ((float)rand() / (float)(RAND_MAX)) * 1000.0; // random floats between 0 and 1000
    }

    start = std::chrono::high_resolution_clock::now();
    reductionCPU(&resultCPU, input);
    end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    cpuTime = (diff.count() * 1000);
    cpuBandwidth = (sizeof(float) * SIZE * 2) / (cpuTime * 1000000);
    printf("CPU Time: %f ms, bandwidth: %f GB/s\n\n", cpuTime, cpuBandwidth);

    reductionWithCuda(&resultGPU, input);

    if (resultCPU != resultGPU)
        printf("CPU result does not match GPU result in naive atomic add. CPU: %f, GPU: %f, diff:%f\n", resultCPU, resultGPU, (resultCPU - resultGPU));
    else
        printf("CPU result matches GPU result in naive atomic add. CPU: %f, GPU: %f\n", resultCPU, resultGPU);

    cudaDeviceReset();

    return 0;
}

void reductionCPU(float *result, float *input)
{
    int i;

    for (i = 0; i < SIZE; i++)
    {
        *result += input[i];
    }
}

__global__ void reductionKernel(float *result, float *input)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < SIZE)
    {
        atomicAdd(result, input[index]);
    }
}

cudaError_t reductionWithCuda(float *result, float *input)
{
    dim3 dim_grid, dim_block;

    float *dev_input = 0;
    float *dev_result = 0;
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    float elapsed = 0;
    double gpuBandwidth;

    dim_block.x = BLOCK_X;
    dim_block.y = 1;
    dim_block.z = 1;

    dim_grid.x = (int)ceil((float)SIZE/(float)(BLOCK_X));
    dim_grid.y = 1;
    dim_grid.z = 1;

    printf("\n---block_x:%d, block_y:%d, dim_x:%d, dim_y:%d\n", dim_block.x, dim_block.y, dim_grid.x, dim_grid.y);

    cudaSetDevice(0);
    cudaMalloc((void**)&dev_input, SIZE * sizeof(float));
    cudaMalloc((void**)&dev_result, sizeof(float));
    cudaMemcpy(dev_input, input, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_result, result, sizeof(float), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reductionKernel<<< dim_grid, dim_block >>>(dev_result, dev_input);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);

    gpuBandwidth = (sizeof(float) * SIZE * 2) / (elapsed * 1000000);
    printf("GPU Time: %f ms, bandwidth: %f GB/s\n", elapsed, gpuBandwidth);

    cudaDeviceSynchronize();
    cudaStatus = cudaMemcpy(result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_result);

    return cudaStatus;
}
