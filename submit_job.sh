#!/bin/sh
#
#SBATCH --account=stats     # The account name for the job.
#SBATCH --job-name=EFN   # The job name.
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=12:00:00              # The time the job will take to run.

source activate ARMEFN
echo $PYTHONPATH
which python3

python3 /rigel/home/srb2201/code/dynamic_mefn/exp_fam/train_network.py $1 $2 $3 $4 $5 $6 $7 $8
