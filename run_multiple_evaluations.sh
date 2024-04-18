#!/bin/bash

# Array of seeds
seeds=(1234 1235)

# Array of parameter combinations
params=("Pa0_Ca0" "Pa0_Ca1")

# Loop through seeds
for seed in "${seeds[@]}"; do
    # Loop through parameter combinations
    for param in "${params[@]}"; do
        # Construct the config directory path
        config_dir="evaluations/seed_${seed}/tragedy_test_${param}"

        # Construct the log file path
        logfile="screenlogs/evaluation_seed_${seed}_${param}"

        # Construct the screen command
        cmd="screen -L -Logfile ${logfile} python baselines/evaluation/evaluate.py --tragedy_test=True --num_episodes=1 --config_dir=${config_dir}"

        # Run the command in a separate screen session
        $cmd
    done
done