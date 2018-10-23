#!/bin/bash

counter=0
rs=0
for D in 5 10 15 20 25
do
  for rs in `seq 1 9`
  do 
    nohup python3 one_sample_helper.py dir_mult $D 0 $rs $1 2>&1 > $counter.log &
    ((counter++))
  done
  nohup python3 one_sample_helper.py dir_mult $D 0 10 $1 2>&1 > $counter.log
  ((counter++));
done

