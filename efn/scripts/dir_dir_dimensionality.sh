#!/bin/bash

counter=0
rs=0
for D in 15
do
  nohup python3 train_efn_helper.py dir_dir $D 0 $rs 2>&1 > $counter.log
  ((counter++))
done

