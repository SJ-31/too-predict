#!/usr/bin/env bash
outdir="../data/output/outlier_detection"

srun --qos=cpu24h --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=30G \
    --mail-type=end --mail-user=shanj3131@gmail.com --job-name=check_outliers \
    austin -o "${outdir}/outlier_detection_austin_out.txt" -m python \
        ./check_outliers.py -c 8
