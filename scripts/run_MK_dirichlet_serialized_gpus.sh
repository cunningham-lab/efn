#!/bin/bash

N=200
counter=0 
for K in 100
do
  for M in 1000
  do
    let "num_rands=$N/$K"
    echo $num_rands
    for rs in $(seq 1 $num_rands)
    do
      nohup python3 MK_helper.py dirichlet 25 $K $M planar30 8 4 $rs 2>&1 > $counter.log
      ((counter++))
    done
  done
done

echo All done
