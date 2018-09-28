#!/bin/bash

counter=0
rs=0
for D in 25
do
  for ds in {0..9}
  do
    #nohup python3 train_efn_helper.py dirichlet $D 0 $rs 2>&1 > $counter.log
    nohup python3 train_efn1_helper.py dirichlet $D 0 0 $ds dim_sweep 2>&1 > $counter.log &
    ((counter++))
  done
done

