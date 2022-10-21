/*! \file multiGPU.h
 *  \brief kernels of for warming up and running on GPU
 */

#ifndef MULTIGPU_H
#define MULTIGPU_H

#include <gpuCommon.h>
#include <profile_util.h>
#include <string>
#include <map>
#include <thread>
#include <logger.h>

/// \defgroup kernels
/// GPU kernels 
//@{
/// @brief run the full vector add set of instructions (allocation, mem copies, kernel) running on all devices, logging information
/// @param Logger Logger instance for logging info
void run_on_devices(Logger &, int);
/// @brief run the full vector add set of instructions (allocation, mem copies, kernel) returning statistics
/// @param val offset value to add to the initial vectors before running the add
/// @return map containing the statisitics of a given set of instructions 
std::map<std::string, double> run_kernel(int val);
//@}

#endif
