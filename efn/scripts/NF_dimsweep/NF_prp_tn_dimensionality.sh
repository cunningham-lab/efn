#!/bin/bash

counter=0
for D in 5 10 15 20 25
do
  for rs in `seq 1 9`
  do 
    nohup python3 train_nf_helper.py prp_tn $D 1 $rs 2>&1 > $counter.log &
    ((counter++))
  done
  nohup python3 train_nf_helper.py prp_tn $D 1 10 2>&1 > $counter.log
  ((counter++));
done

