#!/bin/bash

counter=0
for R in 1 5 10 50 30 20 40
do
  for rs in `seq 1 9`
  do 
    nohup python3 vi_helper.py $R $1 $rs 2>&1 > $counter.log &
    ((counter++))
  done
  nohup python3 vi_helper.py $R $1 10 2>&1 > $counter.log
  ((counter++));
done

