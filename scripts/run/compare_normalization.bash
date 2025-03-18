#!/usr/bin/env bash
outdir="../data/output/normalization_comparison/"

srun --qos=cpu24h --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=30G \
    --mail-type=end --mail-user=shanj3131@gmail.com --job-name=compare_normalization \
    austin -o "${outdir}/compare_normalization_austin_out.txt" -m python \
        ../compare_normalization.py -c 8 -n -p
