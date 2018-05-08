#!/bin/bash

counter=0
for D in 5 15 25
do
  for rs in {1..10}
  do
    nohup python3 train_efn_helper.py normal $D 1 1 0 0 0 0 0 $rs 2>&1 > $counter.log
    ((counter++))
    nohup python3 train_efn_helper.py normal $D 1 1 0 0 0 0 1 $rs 2>&1 > $counter.log
    ((counter++))
  done

  nohup python3 train_efn_helper.py normal $D 10 1 0 0 0 0 0 $rs 2>&1 > $counter.log
  ((counter++))
  nohup python3 train_efn_helper.py normal $D 10 1 0 0 0 0 1 $rs 2>&1 > $counter.log
  ((counter++))
  nohup python3 train_efn_helper.py normal $D 10 1 0 0 0 1 0 $rs 2>&1 > $counter.log
  ((counter++))
  nohup python3 train_efn_helper.py normal $D 10 1 0 0 0 1 1 $rs 2>&1 > $counter.log
  ((counter++))
done

