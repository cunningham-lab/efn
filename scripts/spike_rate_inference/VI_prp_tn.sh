#!/bin/bash

counter=0
for N in 50 10 20 30 40 60 70 80 90 100
do
  for rs in `seq 1 3`
  do 
    nohup python3 vi_helper.py $1 $N $rs 2>&1 > $counter.log &
    ((counter++))
  done
  nohup python3 vi_helper.py $1 $N 10 2>&1 > $counter.log
  ((counter++));
done

