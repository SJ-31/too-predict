#!/usr/bin/env ipython
from pathlib import Path

import anndata as ad
import joblib
import too_predict.model as tm
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from pyhere import here
from too_predict.utils import (
    training_data_internal,
    training_data_internal_test,
)

outdir = here("data", "output", "cross_validation")
seed: int = 4932
adata: ad.AnnData


# #  --- CODE BLOCK ---
# <2025-02-14 Fri> use the model.cross_validate() method on the adata object.
# and plot results
def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-c", "--cores", default=8)
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-n", "--no_dask", default=False, action="store_true")
    parser.add_argument("-l", "--label_class", default="tumor_type")
    return parser.parse_args()


label_classes = ["tumor_type", "primary_site"]
group_classes = ["Project_ID"]
model_dict: dict = {"clr_random_forest": tm.RandomForestPred("clr", "plus_one")}


def cross_validate_helper(lc, gc, model, result_dir_str):
    model = tm.RandomForestPred("clr", "plus_one")
    if gc is not None:
        result_dir: Path = here(outdir, f"{result_dir_str}_by_group_{gc}")
    else:
        result_dir: Path = here(outdir, result_dir_str)
    result_dir.mkdir(exist_ok=True, parents=True)
    results = model.cross_validate(
        adata, label_col=lc, group_col=gc, shuffle=True, random_state=seed
    )
    # <2025-02-28 Fri> Grouping is problematic because some groups are confounded
    # with whatever you are labeling on
    # This means that some instances won't be seen at all in the test data
    # So you need to identify confounded groups and resolve them
    for name, item in results.items():
        if name != "cm":
            item.to_csv(result_dir.joinpath(f"{lc}-{name}.csv"), index=False)
        else:
            for fold, cm in item.items():
                cm.to_csv(result_dir.joinpath(f"{lc}-{name}_cm-fold_{fold}.csv"))


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        print("Using test subset")
        adata = training_data_internal_test()
        outdir = outdir.joinpath("test")
        outdir.mkdir(exist_ok=True, parents=True)
    else:
        adata = training_data_internal()
    label_class = args.label_class
    cluster = SLURMCluster(cores=int(args.cores), memory=f"{args.memory} GB")
    client = Client(cluster)
    backend = "dask" if not args.no_dask else "loky"

    with joblib.parallel_backend(backend):
        for name, model in model_dict.items():
            cross_validate_helper(
                lc=label_class, gc=None, model=model, result_dir_str=name
            )
            # for g in group_classes:
            #     cross_validate_helper(lc=label_class, gc=g, model=model, result_dir_str=name)
