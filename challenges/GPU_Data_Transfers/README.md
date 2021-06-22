# Transferring Data Between CPU and GPU

When writing a code to be run on a hybrid compute system (i.e., one with both CPUs and GPUs) such as Summit, you must consider that the CPU and GPU are separate processors with separate memory associated with them. As such, when running a program on this kind of system, control shifts between the CPU and GPU throughout the code and (because of the separate memory) data must be passed back and forth between the two processors as needed.

In this challenge, you will learn how to perform these data transfers with a simple CUDA vector addition program. The only parts of the code that are missing are the data transfers between CPU and GPU. Your task will be to look up the `cudaMemcpy` API call and add in the required data transfers.

## Basic Outline of the Code

The `vector_addition.cu` code is well documented with comments, but the basic outline of the code is as follows:

* Allocate memory for arrays `A`,  `B`, and `C` on the CPU (commonly called the "host")
* Allocate memory for arrays `d_A`, `d_B`, and `d_C` on the GPU (commonly called the "device")
* Initialize values of arrays `A` and `B` on CPU
* TODO: Transfer data from arrays `A` and `B` (on the CPU) to arrays `d_A` and `d_B` (on the GPU)
* Compute vector addition (`d_C = d_A + d_B`) on the GPU
* TODO: Transfer resulting data from array `d_C` (on the GPU) to array `C` (on the CPU)
* Verify results
* Free GPU memory for arrays `d_A`, `d_B`, and `d_C`
* Free CPU memory for arrays `A`, `B`, and `C`

## Add in the Data Transfers

Before getting started, you'll need to make sure you're in the `GPU_Data_Transfers/` directory:

```
$ cd ~/hands-on-with-summit/challenges/GPU_Data_Transfers/
```

There are two places in the `vector_addition.cu` code (identified with the word `TODO`) where data transfers must be added. Find these two places and add in the necessary data transfers by looking up the `cudaMemcpy` API call (use a Google search for this).

> NOTE: You will not need to edit any files other than `vector_addition.cu`.

## Compile and Run the Program

Once you think you've added the correct lines to the code, try to compile and run it...

First, make sure you have the CUDA Toolkit loaded:

```c
$ module load cuda
``` 

Then, compile the code:

```c
$ make
```
````
[xiaobin0719@login1.ascent GPU_Data_Transfers]$ make
nvcc  -c vector_addition.cu
nvcc  vector_addition.o -o run
````

````
[xiaobin0719@login1.ascent GPU_Data_Transfers]$ cat add_vec_cuda.63704
Tue Jun 22 11:18:25 EDT 2021

---------------------------
__SUCCESS__
---------------------------
N                 = 1048576
Threads Per Block = 256
Blocks In Grid    = 4096
---------------------------


------------------------------------------------------------
Sender: LSF System <lsfadmin@login1>
Subject: Job 63704: <add_vec_cuda> in cluster <ascent> Done

Job <add_vec_cuda> was submitted from host <login1> by user <xiaobin0719> in cluster <ascent> at Tue Jun 22 11:18:07 2021
Job was executed on host(s) <1*login1>, in queue <batch>, as user <xiaobin0719> in cluster <ascent> at Tue Jun 22 11:18:07 2021
                            <42*h49n15>
</ccsopen/home/xiaobin0719> was used as the home directory.
</ccsopen/home/xiaobin0719/hands-on-with-summit/challenges/GPU_Data_Transfers> was used as the working directory.
Started at Tue Jun 22 11:18:07 2021
Terminated at Tue Jun 22 11:18:30 2021
Results reported at Tue Jun 22 11:18:30 2021

The output (if any) is above this job summary.
````



If the code compiles, try to run it as shown below. If not, read the compilation errors and try to determine what went wrong.

```c
$ bsub submit.lsf
```

If the code ran correctly, you will see `__SUCCESS__` along with some other details about how the code was executed. If you don't see this output, try to figure out what went wrong. As always, if you need help, feel free to ask.
