#!/bin/sh
#
#SBATCH --account=stats     # The account name for the job.
#SBATCH --job-name=trainEFN   # The job name.
#SBATCH --mem-per-cpu=5gb
#SBATCH --gres=gpu:1  
#SBATCH --time=11:00:00              # The time the job will take to run.


module load cuda90/toolkit/9.0.176 cuda90/blas/9.0.176 cudnn/7.0
source activate ARMEFN

python3 train_restricted_efn_helper.py $1 $2 $3 $4 $5 $6
