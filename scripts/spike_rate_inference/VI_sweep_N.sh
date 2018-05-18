#!/bin/bash

counter=0
for N in 50 10 100 30 70 20 40 60 80 90
do
  for rs in `seq 1 9`
  do 
    nohup python3 vi_helper.py $1 $N $rs 2>&1 > $counter.log &
    ((counter++))
  done
  nohup python3 vi_helper.py $1 $N 10 2>&1 > $counter.log
  ((counter++));
done

