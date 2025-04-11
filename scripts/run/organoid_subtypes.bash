#!/usr/bin/env bash

srun --qos=cpu24h --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=30G \
    --mail-user=shanj3131@gmail.com --job-name=organoid_subtype \
    Rscript ./explanations/subtyping.R -g "LIHC,COAD_READ" "$@"
