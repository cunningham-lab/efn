#!/bin/bash

rs=0
for D in 3 5 10 15
do
  sbatch train_efn_helper.sh dirichlet $D 0 $rs dim_sweep
done

