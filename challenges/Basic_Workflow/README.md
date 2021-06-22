# Basic Workflow to Run a Job on Ascent/Summit

The basic workflow for running programs on HPC systems is 1) set up your programming environment - i.e., the software you need, 2) compile the code - i.e., turn the human-readable programming language into machine code, 3) request access to one or more compute nodes, and 4) launch your executable on the compute node(s) you were allocated. In this challenge, you will perform these basic steps to see how it works.

For this challenge you will compile and run a vector addition program written in C. It takes two vectors (A and B), adds them element-wise, and writes the results to vector C:

```c
// Add vectors (C = A + B)
for(int i=0; i<N; i++)
{
    C[i] = A[i] + B[i];
}
```

## Step 1: Setting Up Your Programming Environment
Many software packages and scientific libraries are pre-installed on Ascent for users to take advantage of. Several packages are loaded by default when a user logs in to the system and additional packages can be loaded using environment modules. To see which packages are currently loaded in your environment, run the following command:

```
$ module list
``` 

```

[xiaobin0719@login1.ascent Basic_Workflow]$ module list

Currently Loaded Modules:
  1) lsf-tools/2.0   2) DefApps   3) pgi/19.9   4) spectrum-mpi/10.3.1.2-20200121


``` 




> NOTE: The `$` in the command above represents the "command prompt" for the bash shell and is not part of the command that needs to be executed.

For this example program, we will use the PGI compiler. To use the PGI compiler, load the PGI module by issuing the following command:

```
$ module load pgi
```


````
[xiaobin0719@login1.ascent Basic_Workflow]$ module load pgi

Lmod is automatically replacing "xl/16.1.1-7" with "pgi/19.9".


Due to MODULEPATH changes, the following have been reloaded:
  1) spectrum-mpi/10.3.1.2-20200121


````


## Step 2: Compile the Code

Now that you've set up your programming environment for the code used in this challenge, you can go ahead and compile the code. First, make sure you're in the `Basic_Workflow/` directory:

```
$ cd ~/hands-on-with-summit/challenges/Basic_Workflow
```

> NOTE: The path above assumes you cloned the repo in your `/ccsopen/home/username` directory.

Then compile the code. To do so, you'll use the provided `Makefile`, which is a file containing the set of commands to automate the compilation process. To use the `Makefile`, issue the following command:

```
[xiaobin0719@login1.ascent Basic_Workflow]$ cat Makefile
CCOMP = pgcc
CFLAGS =

run: vector_addition.o
	$(CCOMP) $(CFLAGS) vector_addition.o -o run

vector_addition.o: vector_addition.c
	$(CCOMP) $(CFLAGS) -c vector_addition.c

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall:
	rm -f run *.o add_vec_cpu*
```



```
$ make
```

Based on the commands contained in the `Makefile`, an executable named `run` will be created.

## Steps 3-4: Request Access to Compute Nodes and Run the Program

In order to run the executable on Ascent's compute nodes, you need to request access to a compute node and then launch the job on the node. The request and launch can be performed using the single batch script, `submit.lsf`. If you open this script, you will see several lines starting with `#BSUB`, which are the commands that request a compute node and define your job (i.e., give me 1 compute node for 10 minutes, charge project `PROJID` for the time, and name the job and output file `add_vec_cpu`). You will also see a `jsrun` command within the script, which launches the executable (`run`) on the compute node you were given. 

The flags given to `jsrun` define the resources (i.e., cpu cores, gpus) available to your program and the processes/threads you want to run on those resources (for more information on using the `jsrun` job launcher, please see challenge [jsrun\_Job\_Launcher](../jsrun_Job_Launcher)).

To submit and run the job, issue the following command:

```
$ bsub submit.lsf
```


````
[xiaobin0719@login1.ascent Basic_Workflow]$ bsub submit.lsf
Job <63584> is submitted to default queue <batch>.

````






## Monitoring Your Job

Now that the job has been submitted, you can monitor its progress. Is it running yet? Has it finished? To find out, you can issue the command 

```
$ jobstat -u USERNAME
```





where `USERNAME` is your username. This will show you the state of your job to determine if it's running, eligible (waiting to run), or blocked. When you no longer see your job listed with this command, you can assume it has finished (or crashed). Once it has finished, you can see the output from the job in the file named `add_vec_cpu.JOBID`, where `JOBID` is the unique ID given to you job when you submitted it. You can confirm that it gave the correct results by looking for `__SUCCESS__` in the output file. 


````

[xiaobin0719@login1.ascent Basic_Workflow]$ jobstat -u xiaobin0719
compute-hm: Bad host name, host group name or cluster name
-------------------------------------- Running Jobs: 1 (batch: 1/15=6.67% + batch-hm: 0/0) ---------------------------------------
JobID      User       Queue    Project    Nodes Remain     StartTime       JobName
63584      xiaobin0719 batch    GEN158     1     9:48       06/21 20:28:42  add_vec_cpu
-------------------------------------------------------- Eligible Jobs: 0 --------------------------------------------------------
-------------------------------------------------------- Blocked Jobs: 0 ---------------------------------------------------------

````


````


[xiaobin0719@login1.ascent Basic_Workflow]$ cat add_vec_cpu.63584
Mon Jun 21 20:29:00 EDT 2021

---------------------------
__SUCCESS__
---------------------------

------------------------------------------------------------
Sender: LSF System <lsfadmin@login1>
Subject: Job 63584: <add_vec_cpu> in cluster <ascent> Done

Job <add_vec_cpu> was submitted from host <login1> by user <xiaobin0719> in cluster <ascent> at Mon Jun 21 20:28:42 2021
Job was executed on host(s) <1*login1>, in queue <batch>, as user <xiaobin0719> in cluster <ascent> at Mon Jun 21 20:28:42 2021
                            <42*h49n15>
</ccsopen/home/xiaobin0719> was used as the home directory.
</ccsopen/home/xiaobin0719/hands-on-with-summit/challenges/Basic_Workflow> was used as the working directory.
Started at Mon Jun 21 20:28:42 2021
Terminated at Mon Jun 21 20:29:05 2021
Results reported at Mon Jun 21 20:29:05 2021

The output (if any) is above this job summary.


````
