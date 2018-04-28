#!/bin/bash

counter=0 
while [ $counter -le 1 ]
do
  python3 MK_helper.py normal 10 10 1000 linear1 8 4 $counter 2>&1 > $counter.log &
  ((counter++))
done

echo All done
