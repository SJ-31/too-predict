include: "Snakefile"


# Workflow to compare the effects of different data augmentation schemes on
# classification performance

# TODO: have splits for different tasks, where the train are augmented by
store = f"{REPOS}/data_augmentation/{DATE}"
out = f"{OUT}/data_augmentation"

da_config = config["data_augmentation"]

datasets = [directory(f"{store}/{s}") for s in da_config["subsets"]]


rule generate_datasets:
    "Filter dataset, generate synthetic data for each subset and save to storage"
    params:
        store=store,
    output:
        datasets,
    script:
        "scripts/data_augmentation.py"


# rule baseline:
#     "Evaluate performance on the baseline data"
#     ...
# rule evaluate:
#     ...
# rule compare_results:
#     "Aggregate results, produce plots and some summary statistics"
#     ...
