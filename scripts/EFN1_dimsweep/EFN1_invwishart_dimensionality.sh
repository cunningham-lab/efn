#!/bin/bash

counter=0
rs=0
for D in 4 9 16 25 36 49 64
do
  for rs in `seq 1 9`
  do 
    nohup python3 train_efn1_helper.py inv_wishart $D 1 $rs 2>&1 > $counter.log &
    ((counter++))
  done
  nohup python3 train_efn1_helper.py inv_wishart $D 1 10 2>&1 > $counter.log
done

