/*! \file WarmupGPU.h
 *  \brief kernels of for warming up and running on GPU
 */

#ifndef WARMUPGPU_H
#define WARMUPGPU_H

#include <gpuCommon.h>

/// GPU launch types
//@{
#define GPU_ONLY_KERNEL_LAUNCH 0 
#define GPU_ONLY_MEM_ALLOCATE 1
//@}

/// \defgroup kernels
/// GPU kernels 
//@{
void warmup_kernel(int kernel_type);
void run_kernel();
void run_memcopy();
//@}

#endif
