#!/bin/bash

counter=0
rs=0
for D in 2 4 6 8 10 15 20 25
do
  nohup python3 train_efn_helper.py normal $D 1 $rs 2>&1 > $counter.log
  ((counter++))
done

