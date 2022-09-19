/*! \file gpuCommon.h
 *  \brief common gpu related items
 */

#ifndef GPUCOMMON_H
#define GPUCOMMON_H

#include <iostream>

#ifdef USEHIP

#include <hip/hip_runtime.h>

#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize  hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime
#else 

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize  cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime

#endif

#define gpuGetTime(t1) {gpuEventRecord(t1); gpuEventSynchronize(t1);}
#define LogGPUElapsedTime(descrip, t1) std::cout<<"@"<<__func__<<" L"<<__LINE__<<": "<<descrip<<" :: elapsed time "<<t1<<std::endl; 

#endif