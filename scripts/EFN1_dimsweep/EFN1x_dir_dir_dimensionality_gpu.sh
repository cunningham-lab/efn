#!/bin/bash

counter=0
rs=0
for D in 25
do
  for rs in `seq 1 10`
  do 
    nohup python3 one_sample_helper.py dir_dir $D 0 $rs $1 2>&1 > $counter.log
    ((counter++))
  done
done

