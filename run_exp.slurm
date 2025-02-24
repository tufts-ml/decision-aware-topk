#!/usr/bin/env bash
#SBATCH -n 1               # Number of cores
#SBATCH -t 6:00:00          # Runtime in D-HH:MM
#SBATCH --mem-per-cpu 64000  # Memory (in MB) per cpu
#SBATCH --gres=gpu:1
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children

export PYTHONPATH=$PYTHONPATH:$code_dir

python  torch_experiment.py $args
