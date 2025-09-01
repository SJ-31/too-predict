#!/usr/bin/env bash

srun --job-name=holdout \
    --nodes=1 \
    --cpus-per-task=1 \
    --mem-per-cpu=30GB \
    --qos=gpu20gh \
    --partition=gpu \
    --gres=gpu:3g.20gb:1 \
    snakemake -s torch_holdout.smk \
    --profile profiles/default \
    "$@"
