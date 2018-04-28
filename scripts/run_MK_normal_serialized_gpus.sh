#!/bin/bash

N=500
counter=0 
for K in 1 10 100
do
  for M in 1000
  do
    let "num_rands=$N/$K"
    echo $num_rands
    for rs in $(seq 1 $num_rands)
    do
      nohup python3 MK_helper.py normal 40 $K $M linear1 8 4 $rs 2>&1 > $counter.log
      ((counter++))
    done
  done
done

echo All done
