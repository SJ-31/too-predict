#!/usr/bin/env bash
outdir="../data/output/cross_validation/"

srun --qos=cpu24h --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=30G \
    --mail-type=end --mail-user=shanj3131@gmail.com --job-name=cross_validate \
    austin -o "${outdir}/cross_validation_austin_out.txt" -m python \
        ./cross_validate.py -c 8 "$@"

# TODO: [2025-03-12 Wed] Want to run this too
#
# TODO: but first need to rerun the above just to get the confusion matrices for
# the holdouts
# srun --qos=cpu24h --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=30G \
#     --mail-type=end --mail-user=shanj3131@gmail.com --job-name=cross_validate \
#     austin -o "${outdir}/cross_validation_austin_out.txt" -m python \
#         ./cross_validate.py -c 8 -l "primary_site" "$@"


Rscript ./cross_validation_results.R
# Rscript ./cross_validation_results.R -s "additional_splits" -v "test_set"
