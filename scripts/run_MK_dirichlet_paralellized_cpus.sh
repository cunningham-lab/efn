#!/bin/bash

N=200
num_cpus=16
counter=1
for K in 1 10 100
do
  for M in 10 100
  do
    let "num_rands=$N/$K"
    for rs in $(seq 1 $num_rands)
    do
      let "modcpu=$counter%$num_cpus"
      if [ $modcpu -eq 0 ]
      then nohup python3 MK_helper.py dirichlet 25 $K $M planar30 8 4 $rs 2>&1 > $counter.log
      else nohup python3 MK_helper.py dirichlet 25 $K $M planar30 8 4 $rs 2>&1 > $counter.log &
      fi
      ((counter++))
    done
  done
done

echo All done
