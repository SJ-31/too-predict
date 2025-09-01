#!/usr/bin/env bash

srun --job-name=torch_cross_val \
    --nodes=1 \
    --cpus-per-task=1 \
    --mem-per-cpu=30GB \
    --qos=gpu20gh \
    --partition=gpu \
    --gres=gpu:3g.20gb:1 \
    snakemake -s torch_cross_val.smk \
    --profile profiles/default \
    "$@"
