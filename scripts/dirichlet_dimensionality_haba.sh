#!/bin/sh
#
#SBATCH --account=stats     # The account name for the job.
#SBATCH --job-name=dir_arch_search   # The job name.
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=12:00:00              # The time the job will take to run.

counter=0
for D in 5 10
do
  for rs in 0 1 2 3 4
  do
    sbatch arch_search_helper.sh dirichlet $D P $D 0 $rs
    sbatch arch_search_helper.sh dirichlet $D A 1 0 $rs 
  done
done

