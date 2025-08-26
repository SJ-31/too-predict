include: "Snakefile"


import os
from too_predict.imbalance import TORCH_METHODS

# Workflow to compare the effects of different data augmentation schemes on
# classification performance

store = f"{REPOS}/data_augmentation/{DATE}"
out = f"{OUT}/data_augmentation/{DATE}"

da_config = config["data_augmentation"]

augmentations = da_config["augmentations"].keys()

subsets = da_config["subsets"]

if config["test"]:
    subsets = {"test_subset": {"tumor_type": {".": "contains"}}}
    config["dl"]["trainer"]["accelerator"] = "cpu"
    config["dl"]["trainer"]["max_epochs"] = 2
    config["dl"]["trainer"]["log_every_n_steps"] = 1

dataset_dirs = expand("{store}/{s}", store=store, s=subsets.keys())
result_dirs = expand("{out}/{s}", out=out, s=subsets.keys())


rule all:
    input:
        csv=f"{out}/all_results.csv",


gd_results_dict = {
    "train": expand("{d}/{a}_train.h5ad", d=dataset_dirs, a=augmentations),
    "test": expand("{d}/test.h5ad", d=dataset_dirs),
    "dirs": [directory(d) for d in dataset_dirs],
}
if len((torch_augmentations := TORCH_METHODS) & set(augmentations)) > 1:
    gd_results_dict["lightning_logs"] = expand(
        "{out}/{s}/{m}_log", out=out, s=subsets.keys(), m=torch_augmentations
    )


rule generate_datasets:
    "Filter dataset, generate synthetic data for each subset and save to storage"
    params:
        store=store,
        result_dir=out,
        subsets=subsets,
    output:
        **gd_results_dict,
    script:
        "scripts/data_augmentation.py"


rule evaluate:
    input:
        rules.generate_datasets.output.train,
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
