#!/usr/bin/env bash

srun --qos=cpu24h --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=60G \
    --mail-type=end --mail-user=shanj3131@gmail.com --job-name=optimize \
    python ./optimize.py -c 12 -m 60 "$@"
