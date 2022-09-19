/*! \file gpuCommon.h
 *  \brief common gpu related items
 */

#ifndef GPUCOMMON_H
#define GPUCOMMON_H


#ifdef USEHIP

#include <hip/hip_runtime.h>

#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost

#else 

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define gpuMalloc cuMalloc
#define gpuFree cuFree
#define gpuMemcpy cuMemcpy
#define gpuMemcpyHostToDevice cuMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cuMemcpyDeviceToHost

#endif

#endif