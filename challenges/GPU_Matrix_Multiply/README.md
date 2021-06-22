# Matrix Multiply on GPU Using cuBLAS

BLAS (Basic Linear Algebra Subprograms) are a set of linear algebra routines that perform basic vector and matrix operations on CPUs. NVIDIA's cuBLAS library includes a similar set of routines that perform basic linear algebra operations on GPUs. 

In this challenge, you will be given a program that initilizes two matrices with random numbers, performs a matrix multiply on the two matrices on the CPU, performs the same matrix multiply on the GPU, then compares the results. The only part of the code that is missing is the call to `cublasDgemm` that performs the GPU matrix multiply. Your task will be to look up the `cublasDgemm` routine and add it to the section of the code identified with a `TODO`.

## Add the Call to cublasDgemm

Before getting started, you'll need to make sure you're in the `GPU_Matrix_Multiply/` directory:

```
$ cd ~/hands-on-with-summit/challenges/GPU_Matrix_Multiply/
```

Look in the code `cpu_gpu_dgemm.c` and find the `TODO` section and add in the `cublasDgemm` call.

> NOTE: You do not need to perform a transpose operation on the matrices, so the `cublasOperation_t` arguments should be set to `CUBLAS_OP_N`.

## Compile the Code

Once you think you've correctly added the cuBLAS routine, try to compile the code.

```

cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

```


First, you'll need to make sure your programming environment is set up correctly for this program. You'll need to use IBM's ESSL library for the CPU matrix multiply (`dgemm`) and NVIDIA's cuBLAS library for the GPU-version (`cublasDgemm`), so you'll need to load the following modules:

```c
$ module load essl
$ module load cuda
$ module load pgi
```

(The PGI module is needed because we use the PGI compiler (`pgcc`) in the Makefile.) 
Then, try to compile the code:

```c
$ make
``` 

If the code compiles, try to run it. If not, look at the compilation errors to identify the problem.

## Run the Program

Once you've successfully compiled the code, try running it.

```c
$ bsub submit.lsf
```

```
[xiaobin0719@login1.ascent GPU_Matrix_Multiply]$ cat dgemm.63720
Tue Jun 22 11:39:27 EDT 2021
__SUCCESS__

------------------------------------------------------------
Sender: LSF System <lsfadmin@login1>
Subject: Job 63720: <dgemm> in cluster <ascent> Done

Job <dgemm> was submitted from host <login1> by user <xiaobin0719> in cluster <ascent> at Tue Jun 22 11:39:07 2021
Job was executed on host(s) <1*login1>, in queue <batch>, as user <xiaobin0719> in cluster <ascent> at Tue Jun 22 11:39:08 2021
                            <42*h49n15>
</ccsopen/home/xiaobin0719> was used as the home directory.
</ccsopen/home/xiaobin0719/hands-on-with-summit/challenges/GPU_Matrix_Multiply> was used as the working directory.
Started at Tue Jun 22 11:39:08 2021
Terminated at Tue Jun 22 11:39:33 2021
Results reported at Tue Jun 22 11:39:33 2021

The output (if any) is above this job summary.

```

If the CPU and GPU give the same results, you will see the message `__SUCCESS__` in the output file. If you do not receive this message, try to identify the problem. As always, if you need help, make sure to ask.
