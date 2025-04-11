#!/usr/bin/env bash

srun --qos=cpu24h --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=50G \
    --mail-type=end --mail-user=shanj3131@gmail.com --job-name=global_explanations \
    python ./explanations/global_importance.py -c 8 -m 50
