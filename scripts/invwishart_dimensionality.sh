#!/bin/bash

counter=0
for D in 4 9 16 25
do
  for P in 10 20 30
  do
    for rs in {1..10}
    do
      nohup python3 train_efn_helper.py inv_wishart $D 1 0 $P 0 0 0 0 $rs 2>&1 > $counter.log
      ((counter++))
    done

    nohup python3 train_efn_helper.py inv_wishart $D 10 0 $P 0 0 0 0 $rs 2>&1 > $counter.log
    ((counter++))
    nohup python3 train_efn_helper.py inv_wishart $D 10 0 $P 0 0 1 0 $rs 2>&1 > $counter.log
    ((counter++))
  done
done

