#!/bin/bash

GROUP=$1
let "run_0 = (2 * $GROUP)"
let "run_1 = (2 * $GROUP) + 1"
let "run_2 = (2 * $GROUP) + 2"
let "run_3 = (2 * $GROUP) + 3"



# bash $HOME/fed-benchmarks/aggrigator.sh "$run_0" &
bash $HOME/jupyter/openml-fed/Experiments/aggrigator.sh "$run_0" &
pids[1]=$!

# bash $HOME/fed-benchmarks/aggrigator.sh "$run_1" &
bash $HOME/jupyter/openml-fed/Experiments/aggrigator.sh "$run_1" &
pids[2]=$!


# bash $HOME/fed-benchmarks/aggrigator.sh "$run_2" &
bash $HOME/jupyter/openml-fed/Experiments/aggrigator.sh "$run_2" &
pids[3]=$!

# bash $HOME/fed-benchmarks/aggrigator.sh "$run_3" &
bash $HOME/jupyter/openml-fed/Experiments/aggrigator.sh "$run_3" &
pids[4]=$!



for pid in ${pids[*]}; do
    wait $pid
done