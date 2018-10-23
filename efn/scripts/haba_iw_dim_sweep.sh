#!/bin/bash

rs=0
for D in 4 9 16
do
  sbatch train_efn_helper.sh inv_wishart $D 1 $rs dim_sweep
done

