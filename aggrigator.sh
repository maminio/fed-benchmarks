#!/bin/bash

GROUP=$1

for i in {0..20}
do
    let "config_id = (20 * $GROUP) + $i"
    echo " This is the i: $i run for group: $GROUP ==> run: $config_id"
    python Fed-Horizontal-avg.py --run_name "ALICE_RUN_02_CONFIG_$config_id" --config_id $config_id --data_dir /home/s2588862/data/cifar10
done