#!/usr/bin/env bash

srun --qos=cpu24h --nodes=1 --ntasks=1 --mem=30G \
    --mail-type=end --mail-user=shanj3131@gmail.com --job-name=get_counts \
    python ./get_counts.py ; python ./get_training_meta.py
