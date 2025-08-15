include: "Snakefile"


# Workflow to compare the effects of different data augmentation schemes on
# classification performance

store = f"{REPOS}/data_augmentation/{DATE}"
out = f"{OUT}/data_augmentation/{DATE}"

da_config = config["data_augmentation"]

augmentations = da_config["augmentation"].keys()

subsets = da_config["subsets"]

dataset_dirs = expand("{store}/{s}", store=store, s=subsets)
result_dirs = expand("{out}/{s}", out=out, s=subsets)


rule generate_datasets:
    "Filter dataset, generate synthetic data for each subset and save to storage"
    params:
        store=store,
    output:
        [directory(d) for d in dataset_dirs],
    script:
        "scripts/data_augmentation.py"


rule evaluate:
    input:
        rules.generate_datasets.output,
    output:
        final_csv=f"{out}/all_results.csv",
        others=expand("{d}/{a}_result.csv", d=result_dirs, a=augmentations),
    script:
        "scripts/data_augmentation.py"


# rule compare_results:
#     "Aggregate results, produce plots and some summary statistics"
#     ...
