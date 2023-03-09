

## Parameters



## Example


```
$ make
nvcc  -DBLOCK_X=64 -DSIZE=64 sum.cu -o comparesums:wq.exe

$ ./comparesums:wq.exe 
CPU Time: 0.000719 ms, bandwidth: 0.712100 GB/s


---block_x:64, block_y:1, dim_x:1, dim_y:1
GPU Time: 0.071552 ms, bandwidth: 0.007156 GB/s
CPU result matches GPU result in naive atomic add. CPU: 32431.460938, GPU: 32431.460938
```


Double the array size, but keep the block size fixed
```
$ export SIZE=128
$ make clean
$ make
nvcc  -DBLOCK_X=64 -DSIZE=128 sum.cu -o comparesums:wq.exe
$ ./comparesums:wq.exe 
CPU Time: 0.001276 ms, bandwidth: 0.802508 GB/s


---block_x:64, block_y:1, dim_x:2, dim_y:1
GPU Time: 0.062432 ms, bandwidth: 0.016402 GB/s
CPU result does not match GPU result in naive atomic add. CPU: 68111.460938, GPU: 68111.484375, diff:-0.023438
```

Double the block size so that the BLOCKSIZE and the array size are equivalent. This means that one GPU block is run to complete the AtomicAdd operation
```
$ export BLOCKSIZE=128
$ make clean
rm ./comparesums:wq.exe
$ make
nvcc  -DBLOCK_X=128 -DSIZE=128 sum.cu -o comparesums:wq.exe
$ ./comparesums:wq.exe 
CPU Time: 0.001275 ms, bandwidth: 0.803137 GB/s


---block_x:128, block_y:1, dim_x:1, dim_y:1
GPU Time: 0.029472 ms, bandwidth: 0.034745 GB/s
CPU result matches GPU result in naive atomic add. CPU: 68255.023438, GPU: 68255.023438
```

Here, we increase the array size to 1024, but leave the number of threads per GPU block at 128. This means that 8 CUDA blocks are used to execute the blocks.
```
$ export SIZE=1024
$ make clean
rm ./comparesums:wq.exe
$ make
nvcc  -DBLOCK_X=128 -DSIZE=1024 sum.cu -o comparesums:wq.exe
$ ./comparesums:wq.exe 
CPU Time: 0.009418 ms, bandwidth: 0.869824 GB/s


---block_x:128, block_y:1, dim_x:8, dim_y:1
GPU Time: 0.061472 ms, bandwidth: 0.133264 GB/s
CPU result matches GPU result in naive atomic add. CPU: 509530.250000, GPU: 509530.250000


```
Hmm... did we get lucky? Let's run it a few more times..

Run the code again, and this time we see a different answer!
```
$ ./comparesums:wq.exe 
CPU Time: 0.009407 ms, bandwidth: 0.870841 GB/s


---block_x:128, block_y:1, dim_x:8, dim_y:1
GPU Time: 0.030464 ms, bandwidth: 0.268908 GB/s
CPU result does not match GPU result in naive atomic add. CPU: 498150.000000, GPU: 498150.031250, diff:-0.031250
```


Increase threads per block to 1024
```
$ ./comparesums:wq.exe 
CPU Time: 0.009355 ms, bandwidth: 0.875681 GB/s


---block_x:1024, block_y:1, dim_x:1, dim_y:1
GPU Time: 0.030944 ms, bandwidth: 0.264736 GB/s
CPU result does not match GPU result in naive atomic add. CPU: 516492.125000, GPU: 516492.250000, diff:-0.125000
```
The results are different! What the heck is going on!?
