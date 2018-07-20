#!/bin/bash

rs=0
for D in 2 5 10 15
do
  sbatch train_efn_helper.sh normal $D 1 $rs dim_sweep
done

