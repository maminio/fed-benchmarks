#!/bin/bash

GROUP=$1
let "run_0 = (2 * $GROUP)"
let "run_1 = (2 * $GROUP) + 1"
bash $HOME/fed-benchmarks/aggrigator.sh "$run_0" &
pids[1]=$!

bash $HOME/fed-benchmarks/aggrigator.sh "$run_0" &
pids[2]=$!


for pid in ${pids[*]}; do
    wait $pid
done