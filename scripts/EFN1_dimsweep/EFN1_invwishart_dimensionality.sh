#!/bin/bash

counter=0
rs=0
for D in 4 9 16 25 36 49 64
do
  nohup python3 train_efn1_helper.py inv_wishart $D 1 $rs 2>&1 > $counter.log &
  ((counter++))
done

