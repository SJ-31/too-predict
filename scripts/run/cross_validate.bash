#!/usr/bin/env bash
srun --qos=cpu24h --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=40G \
    --mail-type=end --mail-user=shanj3131@gmail.com --job-name=cross_validate \
    python ./cross_validate.py -c 12 -m 40 "$@"

Rscript ./interpretation/cross_validation_results.R
# Rscript ./cross_validation_results.R -s "additional_splits" -v "test_set"
