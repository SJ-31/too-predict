#!/usr/bin/env bash

srun --qos=cpu24h --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=30G --mail-type=end \
    --mail-user=shanj3131@gmail.com --job-name=organoid_features \
    python ../feature_selection_organoid.py -c 8
