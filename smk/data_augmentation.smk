include: "Snakefile"


import os

# Workflow to compare the effects of different data augmentation schemes on
# classification performance

store = f"{REPOS}/data_augmentation/{DATE}"
out = f"{OUT}/data_augmentation/{DATE}"

da_config = config["data_augmentation"]

augmentations = da_config["augmentations"].keys()

subsets = da_config["subsets"]

if config["test"]:
    subsets = {"test_subset": {"tumor_type": {".": "contains"}}}

dataset_dirs = expand("{store}/{s}", store=store, s=subsets.keys())
result_dirs = expand("{out}/{s}", out=out, s=subsets.keys())


rule all:
    input:
        csv=f"{out}/all_results.csv",


rule generate_datasets:
    "Filter dataset, generate synthetic data for each subset and save to storage"
    params:
        store=store,
        subsets=subsets,
    output:
        expand("{d}/{a}_train.h5ad", d=dataset_dirs, a=augmentations),
        expand("{d}/test.h5ad", d=dataset_dirs),
        dirs=[directory(d) for d in dataset_dirs],
    script:
        "scripts/data_augmentation.py"


rule evaluate:
    input:
        rules.generate_datasets.output.dirs,
    params:
        outdir=out,
    output:
        final_csv=rules.all.input.csv,
        others=expand("{d}/{a}_result.csv", d=result_dirs, a=augmentations),
    script:
        "scripts/data_augmentation.py"


# rule compare:
#     "Aggregate results, produce plots and some summary statistics"
#     input:
#         rules.output.evaluate.final_c,
#     ...
