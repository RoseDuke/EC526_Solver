#! /bin/bash -l
# The -l specifies that we are loading modules
#
## Walltime limit
#$ -l h_rt=2:50:00
#
## Give the job a name.
#$ -N pj_jacobi
#
## Redirect error output to standard output
#$ -j y
#
## What project to use. "paralg" is the project for the class
#$ -P ec526
#
## Ask for nodes with 4 cores, 4 cores total (so 1 node)
#$ -pe mpi_4_tasks_per_node 4

module load pgi
module load gcc

JOB_NAME="pj_jacobi"

exec > ${SGE_O_WORKDIR}/${JOB_NAME}-${JOB_ID}.scc.out 2>&1
 
./pj_jacobi_2d

exit
