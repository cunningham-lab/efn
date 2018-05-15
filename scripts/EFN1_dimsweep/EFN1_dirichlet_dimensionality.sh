#!/bin/bash

counter=0
rs=0
for D in 3 5 10 15 20 25
do
  nohup python3 train_efn1_helper.py dirichlet $D 0 $rs 2>&1 > $counter.log &
  ((counter++))
done

