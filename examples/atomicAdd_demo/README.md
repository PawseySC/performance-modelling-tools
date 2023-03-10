# atomicAdd Example

## Overview
This example is meant to demonstrate one use-case for the [`atomicAdd`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicadd) operation on GPUs. We are specifically interested in highlighting the differences in performance and in the results of the summation of an array of floating point values. From the [GPU-C programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions) :


*"An atomic function performs a read-modify-write atomic operation on one 32-bit or 64-bit word residing in global or shared memory ... 
For example, atomicAdd() reads a word at some address in global or shared memory, adds a number to it, and writes the result back to the same address."*


Further,


*"The operation is atomic in the sense that it is guaranteed to be performed without interference from other threads. In other words, no other thread can access this address until the operation is complete. If an atomic instruction executed by a warp reads, modifies, and writes to the same location in global memory for more than one of the threads of the warp, each read/modify/write to that location occurs and they are all serialized, but the order in which they occur is undefined."* - [source](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/memorystatisticsatomics.htm#:~:text=An%20atomic%20function%20performs%20a,back%20to%20the%20same%20address.)



Order of operations for summations matter, specifically in cases where the magnitude of the numbers being added varies signficantly. This is demonstrated, in a somewhat dramatic sense, in [this interactive example](https://wandbox.org/permlink/GGR6EhdaViT1J0UR). Because the order in which threads execute addition in the `atomicAdd` call, we should expect that summation on the GPU with `atomicAdd` will provide different results than a serial CPU application, and possibly between successive runs of the same application. 


## Explanation of the code
In this atomicAdd example, we have a code (`sum.cpp`) that creates a random array of `float`'s . The elements of the array vary in value between 0 and 1000. A sum of the array is done in serial on the CPU

```c
    for (int i = 0; i < SIZE; i++)
    {
        *result += input[i];
    }
```

and on the GPU using `atomicAdd`

```c
__global__ void reductionKernel(float *result, float *input)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < SIZE)
    {
        atomicAdd(result, input[index]);
    }
}
```

On the GPU, the `input` array is of size `SIZE`. The `reductionKernel` is launched using `BLOCK_X` threads per GPU block. The number of GPU blocks in the grid is calculated to be `(int)(ceil((float)SIZE/float(BLOCK_X))`; this ensures that a sufficient number of GPU threads are created to completely step through all of the elements of the `input` array.


For both summations, the value held by the `result` pointer is initialized to zero before execution.


## Building & running the code
In case you're interested in building and playing with this example, this section walks you through building the example and controlling its behavior. 

!!! note "Building on Mulan"
    On Mulan, you will want to source the `mulan.env.sh` file before executing other steps. This will load the rocm module and set the `CC` environment variable to `hipcc`


You can set two environment variables that 

1. control the number of GPU threads per block ( `BLOCKSIZE` ), and
2. control the size of the array that is reduced ( `SIZE` )

By default, both `BLOCKSIZE` and `SIZE` are set to `64`; this results in a single GPU block with 64 threads per block being used to reduce the random float array of size 64. As an example, you can do the following to build the code, customizing your own `BLOCKSIZE` and `SIZE` settings

```bash
$ BLOCKSIZE=128 SIZE=1024 make
hipcc  -DBLOCK_X=128 -DSIZE=1024 sum.cpp -o comparesums.exe
```
In this case, the array size is set to 1024 and the number of GPU threads per block is set to 128; this would result in 8 GPU blocks being run during execution of the kernel.

You can run the application by running the `./comparesums.exe` executable on a system with an Nvidia GPU (e.g. Topaz)

```bash
$ ./comparesums.exe 
CPU Time: 0.009418 ms, bandwidth: 0.869824 GB/s


---block_x:128, block_y:1, dim_x:8, dim_y:1
GPU Time: 0.061472 ms, bandwidth: 0.133264 GB/s
CPU result matches GPU result in naive atomic add. CPU: 509530.250000, GPU: 509530.250000
```

If you want to change the `BLOCKSIZE` or `SIZE` parameters, you will need to run `make clean` first, and then re-make the application, e.g.
```bash
$ make clean
rm *.exe

$ BLOCKSIZE=256 SIZE=1024 make
hipcc  -DBLOCK_X=256 -DSIZE=1024 sum.cpp -o comparesums.exe
```

## Interpreting the output

The output of the code shows the execution time for both the CPU and GPU kernels, as well as the effective bandwidth. Additionally, before the GPU kernel launch, you will see confirmation of the number of threads per block (`block_x`) and the number of blocks in the computational grid (`dim_x`); the product of `block_x` and `dim_x` will indicate the number of threads launched on the GPU to complete the reduction. Last, you will see a statement about whether the CPU or GPU results match or not.

!!! note
    We look at the effective bandwidth as a measure of the performance since the summation operation has a low arithmetic intensity and is considered bound by the memory bandwidth. The effective bandwidth is calculated as the total number of reads and writes, multiplied by the size of each float, and divided by the kernel runtime.



## A few example runs

### Block Size = 64 and Array Size = 64
```
$ make
hipcc  -DBLOCK_X=64 -DSIZE=64 sum.cpp -o comparesums.exe

$ ./comparesums.exe 
CPU Time: 0.000719 ms, bandwidth: 0.712100 GB/s


---block_x:64, block_y:1, dim_x:1, dim_y:1
GPU Time: 0.071552 ms, bandwidth: 0.007156 GB/s
CPU result matches GPU result in naive atomic add. CPU: 32431.460938, GPU: 32431.460938
```

### Block Size = 64 and Array Size = 128
Double the array size, but keep the block size fixed
```
$ export SIZE=128
$ make clean
$ make
hipcc  -DBLOCK_X=64 -DSIZE=128 sum.cpp -o comparesums.exe
$ ./comparesums.exe 
CPU Time: 0.001276 ms, bandwidth: 0.802508 GB/s


---block_x:64, block_y:1, dim_x:2, dim_y:1
GPU Time: 0.062432 ms, bandwidth: 0.016402 GB/s
CPU result does not match GPU result in naive atomic add. CPU: 68111.460938, GPU: 68111.484375, diff:-0.023438
```

### Block Size = 128 and Array Size = 128
Double the block size, again so that the BLOCKSIZE and the array size are equivalent. This means that one GPU block is run to complete the AtomicAdd operation
```
$ export BLOCKSIZE=128
$ make clean
rm ./comparesums.exe
$ make
hipcc  -DBLOCK_X=128 -DSIZE=128 sum.cpp -o comparesums.exe
$ ./comparesums.exe 
CPU Time: 0.001275 ms, bandwidth: 0.803137 GB/s


---block_x:128, block_y:1, dim_x:1, dim_y:1
GPU Time: 0.029472 ms, bandwidth: 0.034745 GB/s
CPU result matches GPU result in naive atomic add. CPU: 68255.023438, GPU: 68255.023438
```
The results agree again, with some consistency between successive runs as well.

### Block Size = 128 and Array Size = 1024
Here, we increase the array size to 1024, but leave the number of threads per GPU block at 128. This means that 8 GPU blocks are used to execute the blocks.
```
$ export SIZE=1024
$ make clean
rm ./comparesums.exe
$ make
hipcc  -DBLOCK_X=128 -DSIZE=1024 sum.cpp -o comparesums.exe
$ ./comparesums.exe 
CPU Time: 0.009418 ms, bandwidth: 0.869824 GB/s


---block_x:128, block_y:1, dim_x:8, dim_y:1
GPU Time: 0.061472 ms, bandwidth: 0.133264 GB/s
CPU result matches GPU result in naive atomic add. CPU: 509530.250000, GPU: 509530.250000

```
Hmm... did we get lucky? Let's run it a few more times..

Run the code again, and this time we see a different answer!
```
$ ./comparesums.exe 
CPU Time: 0.009407 ms, bandwidth: 0.870841 GB/s


---block_x:128, block_y:1, dim_x:8, dim_y:1
GPU Time: 0.030464 ms, bandwidth: 0.268908 GB/s
CPU result does not match GPU result in naive atomic add. CPU: 498150.000000, GPU: 498150.031250, diff:-0.031250
```

### Block Size = 1024 and Array Size = 1024
Increase threads per block to 1024
```
$ ./comparesums.exe 
CPU Time: 0.009355 ms, bandwidth: 0.875681 GB/s


---block_x:1024, block_y:1, dim_x:1, dim_y:1
GPU Time: 0.030944 ms, bandwidth: 0.264736 GB/s
CPU result does not match GPU result in naive atomic add. CPU: 516492.125000, GPU: 516492.250000, diff:-0.125000
```
The results are different!
