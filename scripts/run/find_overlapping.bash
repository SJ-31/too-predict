#!/usr/bin/env bash
outdir="../data/output/find_overlapping/"

srun --qos=cpu24h --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=30G \
    --mail-type=end --mail-user=shanj3131@gmail.com --job-name=cross_validate \
    austin -o "${outdir}/find_overlapping_austin_out.txt" -m python \
        ../find_overlapping.py -c 8 "$@"
