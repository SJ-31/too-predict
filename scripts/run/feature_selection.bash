#!/usr/bin/env bash

srun --qos=cpu24h --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=80G --mail-type=end \
    --mail-user=shanj3131@gmail.com --job-name=feature_selection \
    python ./feature_selection.py -c 12 -m 80 "$@"
