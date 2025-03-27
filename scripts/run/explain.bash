#!/usr/bin/env bash

srun --qos=cpu24h --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=30G \
    --mail-type=end --mail-user=shanj3131@gmail.com --job-name=explanations \
    python ./explanations/chula_misses.py -c 8 -m 30 --no_shap
