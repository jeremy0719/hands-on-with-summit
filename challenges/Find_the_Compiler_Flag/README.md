# Find the Missing Compiler Flag
OpenACC is a directive-based approach to programming for GPUs. Instead of using a low-level programming method like CUDA, where the programmer is responsible for explicitly transferring data between the CPU and GPU and writing GPU kernels, with a directive-based model, the programmer simply adds "hints" within the code which tell the compiler where data transfers should happen and which code sections to offload to the GPU. An additional benefit of this type of GPU programming model is that the code can be compiled for either a CPU or GPU simply by adding or removing compiler flags (whereas a CUDA code will need to be run on a GPU).

In this challenge, you will need to find the compiler flag that enables GPU support in a simple OpenACC vector addition program. The single `#pragma acc parallel loop` (which is the hint to the compiler) line is the only change needed to make this a GPU code. But without the correct compiler flag, that line will be ignored and a CPU-only executable will be created. 

## Step 1: Set Up the Programming Environment

In order to run the provided OpenACC code, we will need to modify our programming environment. First, we will change the compiler to PGI:

```
$ module load pgi
```

Then, we will load the CUDA Toolkit:

```
$ module load cuda
```

## Step 2: Find the Necessary Compiler Flag

Next, you will need to find the PGI compiler flag needed to compile the code with OpenACC-support. To do so, you can either search within the [Summit User Guide](https://docs.olcf.ornl.gov/systems/summit_user_guide.html#) or the [PGI documentation](https://www.pgroup.com/resources/docs/19.10/openpower/index.htm). 

> NOTE: Compiler flags differ between different compilers so make sure you find the correct flag for the **PGI compiler**.

## Step 3: Add the Compiler Flag to the Makefile and Compile

First, make sure you're in the `Find_the_Compiler_Flag` directory:

```
$ cd ~/hands-on-with-summit/challenges/Find_the_Compiler_Flag
```

Ok, if you haven't done so already, go find the compiler flag...
Ok, now that you think you found the correct compiler flag, add it to the end of the `CFLAGS = -Minfo=all` line in the Makefile. Then, compile the code:

```
$ make
```


````
[xiaobin0719@login1.ascent Find_the_Compiler_Flag]$ vim Makefile
[xiaobin0719@login1.ascent Find_the_Compiler_Flag]$ make
pgcc -Minfo=all -acc -c vector_addition.c
main:
     24, Generating Tesla code
         25, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
     24, Generating implicit copyout(C[:1048576]) [if not already present]
         Generating implicit copyin(B[:1048576],A[:1048576]) [if not already present]
pgcc -Minfo=all -acc vector_addition.o -o run


````




If you added the correct flag, you should see evidence of it in the output from the compiler:

```
main:
     24, Generating Tesla code
         25, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
     24, Generating implicit copyout(C[:1048576])
         Generating implicit copyin(B[:1048576],A[:1048576])
```

## Step 4: Run the Program

Now, test that you have correctly compiled the code with OpenACC-support by launching the executable on a compute node. To do so, issue the following command:

```
$ bsub submit.lsf
```

> NOTE: The submit.lsf script requests access to 1 compute node for 10 minutes and launches the executable on that compute node using the job launcher, `jsrun`.


Once the job is complete, you can confirm that it gave the correct results by looking for `__SUCCESS__` in the output file, `add_vec_acc.JOBID`, where JOBID will be the unique number associated with your job. 

But did you run on the CPU or GPU? An easy way to tell is using NVIDIA's profiler, `nvprof`. This was included in the `jsrun` command so if you ran on the GPU, you should also see output from the profiler as shown below:

```
==6163== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   62.26%  363.49us         2  181.74us  181.60us  181.89us  [CUDA memcpy HtoD]
                   31.17%  181.98us         1  181.98us  181.98us  181.98us  [CUDA memcpy DtoH]
                    6.57%  38.368us         1  38.368us  38.368us  38.368us  main_24_gpu
```

If you need help, don't be afraid to ask. If you were successful, congratulations! You just ran a program on a GPU in one of the fastest supercomputers in the world!


```
[xiaobin0719@login1.ascent Find_the_Compiler_Flag]$ cat add_vec_acc.63587
Mon Jun 21 20:56:19 EDT 2021
==91649== NVPROF is profiling process 91649, command: ./run
==91649== Profiling application: ./run
__SUCCESS__
==91649== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   62.19%  363.29us         2  181.65us  181.47us  181.82us  [CUDA memcpy HtoD]
                   31.13%  181.86us         1  181.86us  181.86us  181.86us  [CUDA memcpy DtoH]
                    6.67%  38.976us         1  38.976us  38.976us  38.976us  main_24_gpu
      API calls:   61.10%  236.65ms         1  236.65ms  236.65ms  236.65ms  cuDevicePrimaryCtxRetain
                   34.12%  132.17ms         1  132.17ms  132.17ms  132.17ms  cuDevicePrimaryCtxRelease
                    1.93%  7.4581ms         1  7.4581ms  7.4581ms  7.4581ms  cuMemFreeHost
                    1.83%  7.0764ms         1  7.0764ms  7.0764ms  7.0764ms  cuMemHostAlloc
                    0.75%  2.9035ms         4  725.88us  539.37us  911.39us  cuMemAlloc
                    0.14%  542.20us         1  542.20us  542.20us  542.20us  cuMemAllocHost
                    0.06%  222.73us         3  74.242us  9.6210us  172.81us  cuStreamSynchronize
                    0.03%  127.73us         1  127.73us  127.73us  127.73us  cuModuleLoadDataEx
                    0.02%  64.762us         2  32.381us  30.569us  34.193us  cuMemcpyHtoDAsync
                    0.01%  44.888us         1  44.888us  44.888us  44.888us  cuLaunchKernel
                    0.00%  15.792us         1  15.792us  15.792us  15.792us  cuStreamCreate
                    0.00%  10.663us         3  3.5540us  2.7460us  4.3550us  cuPointerGetAttributes
                    0.00%  10.200us         1  10.200us  10.200us  10.200us  cuMemcpyDtoHAsync
                    0.00%  7.9410us         1  7.9410us  7.9410us  7.9410us  cuEventRecord
                    0.00%  7.6380us         3  2.5460us     498ns  6.4150us  cuCtxSetCurrent
                    0.00%  5.3020us         1  5.3020us  5.3020us  5.3020us  cuEventSynchronize
                    0.00%  3.9880us         2  1.9940us     967ns  3.0210us  cuEventCreate
                    0.00%  3.9060us         1  3.9060us  3.9060us  3.9060us  cuDeviceGetPCIBusId
                    0.00%  3.3890us         4     847ns     508ns  1.3140us  cuDeviceGetAttribute
                    0.00%  2.5680us         3     856ns     510ns  1.5000us  cuDeviceGetCount
                    0.00%  1.6140us         2     807ns     334ns  1.2800us  cuDeviceGet
                    0.00%     960ns         1     960ns     960ns     960ns  cuModuleGetFunction
                    0.00%     574ns         1     574ns     574ns     574ns  cuDeviceComputeCapability
                    0.00%     382ns         1     382ns     382ns     382ns  cuCtxGetCurrent
                    0.00%     377ns         1     377ns     377ns     377ns  cuDriverGetVersion
 OpenACC (excl):   87.23%  10.922ms         1  10.922ms  10.922ms  10.922ms  acc_enter_data@vector_addition.c:24
                    9.59%  1.2005ms         1  1.2005ms  1.2005ms  1.2005ms  acc_wait@vector_addition.c:30
                    1.20%  150.16us         1  150.16us  150.16us  150.16us  acc_device_init@vector_addition.c:24
                    0.61%  76.172us         2  38.086us  37.913us  38.259us  acc_enqueue_upload@vector_addition.c:24
                    0.44%  55.146us         2  27.573us  12.682us  42.464us  acc_wait@vector_addition.c:24
                    0.41%  51.347us         1  51.347us  51.347us  51.347us  acc_enqueue_launch@vector_addition.c:24 (main_24_gpu)
                    0.24%  30.567us         1  30.567us  30.567us  30.567us  acc_enqueue_download@vector_addition.c:30
                    0.20%  24.957us         1  24.957us  24.957us  24.957us  acc_exit_data@vector_addition.c:24
                    0.09%  10.651us         1  10.651us  10.651us  10.651us  acc_compute_construct@vector_addition.c:24
                    0.00%       0ns         3       0ns       0ns       0ns  acc_delete@vector_addition.c:30
                    0.00%       0ns         3       0ns       0ns       0ns  acc_alloc@vector_addition.c:24
                    0.00%       0ns         3       0ns       0ns       0ns  acc_create@vector_addition.c:24

------------------------------------------------------------
Sender: LSF System <lsfadmin@login1>
Subject: Job 63587: <add_vec_acc> in cluster <ascent> Done

Job <add_vec_acc> was submitted from host <login1> by user <xiaobin0719> in cluster <ascent> at Mon Jun 21 20:56:01 2021
Job was executed on host(s) <1*login1>, in queue <batch>, as user <xiaobin0719> in cluster <ascent> at Mon Jun 21 20:56:01 2021
                            <42*h49n15>
</ccsopen/home/xiaobin0719> was used as the home directory.
</ccsopen/home/xiaobin0719/hands-on-with-summit/challenges/Find_the_Compiler_Flag> was used as the working directory.
Started at Mon Jun 21 20:56:01 2021
Terminated at Mon Jun 21 20:56:25 2021
Results reported at Mon Jun 21 20:56:25 2021

The output (if any) is above this job summary.

```