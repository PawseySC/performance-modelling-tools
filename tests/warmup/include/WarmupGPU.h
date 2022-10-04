/*! \file WarmupGPU.h
 *  \brief kernels of for warming up and running on GPU
 */

#ifndef WARMUPGPU_H
#define WARMUPGPU_H

#include <gpuCommon.h>
#include <profile_util.h>
#include <string>
#include <map>
#include <logger.h>


/// GPU launch types
//@{
#define GPU_ONLY_KERNEL_LAUNCH 0 
#define GPU_ONLY_MEM_ALLOCATE 1
#define GPU_ONLY_MEM_TH2D 2
#define GPU_ONLY_MEM_TD2H 3
#define GPU_ONLY_NUM_LAUNCH_TYPES 4
//@}

/// \defgroup kernels
/// GPU kernels 
//@{
void launch_warmup_kernel(int itype, int i, int j, unsigned long long N);
void warmup_kernel_over_kernels(int rounds = 2);
void warmup_kernel_over_rounds(int rounds = 2, int sleeptime = 0);
void run_on_devices(Logger &, int);
std::map<std::string, double> run_kernel(int);

void run_memcopy();
//@}

#endif
