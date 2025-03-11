#!/usr/bin/env ipython
from pathlib import Path

import anndata as ad
import joblib
import pandas as pd
import too_predict.model as tm
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from pyhere import here
from scanpy import read_h5ad
from sklearn.ensemble import RandomForestClassifier
from too_predict.filter import Filter
from too_predict.imputer import IMPLEMENTED_IMPUTATION, Imputer
from too_predict.transformer import Transformer
from too_predict.utils import (
    RANDOM_STATE,
    RNG,
    ref_feature_lists_internal,
    training_data_internal,
    training_data_internal_test,
)

OUTDIR: Path = here("data", "output", "cross_validation")
K: int = 10


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-c", "--cores", default=8)
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-d", "--dask", default=False, action="store_true")
    parser.add_argument("-a", "--cached", default=False, action="store_true")
    parser.add_argument("-l", "--label_class", default="tumor_type")
    return parser.parse_args()


STORAGE_DIR = here("remote", "repos", "too-predict", "normalization_comparison")
LABEL_CLASSES = ["tumor_type", "primary_site"]
GROUP_CLASSES = ["Project_ID"]
REF_LISTS, FEATURE_LISTS = ref_feature_lists_internal()
USE_CACHED: bool = True

# Dictionary of model_name -> [model, transformation, imputation, feature_set]
MODELS: dict = {
    "clr_random_forest_minfo": {
        "model": tm.RandomForestPred(),
        "t": "clr",
        "i": "plus_one",
        "f": "mutual_info_feature_list_3000",
    },
    "clr_xgboost_edger": {
        "model": tm.XgboostPred(),
        "t": "clr",
        "i": "plus_one",
        "f": "mutual_info_feature_list_3000",
    },
    "clr_random_forest_edger": {
        "model": tm.RandomForestPred(),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
    },
    "alr_random_forest_low_variance": {
        "model": tm.AlrBase(
            RandomForestClassifier(random_state=RNG),
            references=REF_LISTS["variance_feature_list_lowest_20"],
        ),
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "r": "variance_feature_list_lowest_20",
    },
    "tmm_random_forest_edger": {
        "model": tm.RandomForestPred(),
        "t": "tmm",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
    },
    "tpm_random_forest_edger": {
        "model": tm.RandomForestPred(),
        "t": "tpm",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
    },
    "fpkm_random_forest_edger": {
        "model": tm.RandomForestPred(),
        "t": "fpkm",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
    },
    "dirichlet_random_forest_edger": {
        "model": tm.SimPred(
            RandomForestClassifier(random_state=RNG), method="dirichlet"
        ),
        "t": "none",
        "i": "none",
        "f": "edgeR_median_lfc_feature_list_3000",
    },
}

ADATA: ad.AnnData

# [2025-03-11 Tue]
# Want to see how well models handle Chula organoids and datasets from other projects
# Ideally this should be done with StratifiedGroupKFold but
# grouping is problematic because some groups are confounded
# with whatever you are labeling on
# This means that some instances won't be seen at all in the test data
# gc is the group variable excluded during the cv folds e.g. Sample_Type

ADDITIONAL_SPLITS: dict = {
    "CHULA": lambda x: (
        x[~x.obs["Project_ID"].str.contains("CHULA"), :],
        x[x.obs["Project_ID"].str.contains("CHULA"), :],
    ),
    "CGCI": lambda x: (
        x[~x.obs["Project_ID"].str.contains("CGCI"), :],
        x[x.obs["Project_ID"].str.contains("CGCI"), :],
    ),
    "CPTAC": lambda x: (
        x[~x.obs["Project_ID"].str.contains("CPTAC"), :],
        x[x.obs["Project_ID"].str.contains("CPTAC"), :],
    ),
    "GEO": lambda x: (
        x[~x.obs["Project_ID"].str.contains("GSE"), :],
        x[x.obs["Project_ID"].str.contains("GSE"), :],
    ),
}


def cross_validate_helper(
    lc, gc, model, result_dir_str, trans, impute, feature_set, references=None
):
    adata = ADATA.copy()
    if gc is not None:
        result_dir: Path = here(OUTDIR, f"{result_dir_str}_by_group_{gc}")
    else:
        result_dir: Path = here(OUTDIR, result_dir_str)
    result_dir.mkdir(exist_ok=True, parents=True)
    dir = STORAGE_DIR.joinpath(feature_set)
    output = here(dir, f"{trans}-{impute}.h5ad")
    if (not output.exists() or trans is None) or not USE_CACHED:
        if feat := FEATURE_LISTS[feature_set]:
            adata = Filter(
                feat if references is None else feat + REF_LISTS[references],
                feature_col="GENEID",
            ).fit_transform(adata)
        Transformer(trans, impute_fn=Imputer(impute), inplace=True).fit_transform(adata)
        # Does nothing if trans is None
    else:
        adata = read_h5ad(output)
    if not here(result_dir, f"{lc}-misc.csv").exists():
        results = model.cross_validate(
            adata, label_col=lc, group_col=gc, random_state=RANDOM_STATE, n_splits=K
        )
        write_results(results, result_dir, lc, cm_prefix="fold_")
    result_dir2 = result_dir.joinpath("additional_splits")
    result_dir2.mkdir(exist_ok=True, parents=True)
    results2 = model.holdout(adata, ADDITIONAL_SPLITS, label_col=lc)
    write_results(results2, result_dir2, lc)


def write_results(results, result_dir, label_col, cm_prefix: str = ""):
    for name, item in results.items():
        if name != "cm" and isinstance(item, pd.DataFrame):
            item.to_csv(result_dir.joinpath(f"{label_col}-{name}.csv"), index=False)
        elif name == "cm":
            for lab, cm in item.items():
                cm.to_csv(
                    result_dir.joinpath(f"{label_col}-{name}_cm-{cm_prefix}{lab}.csv")
                )


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        print("Using test subset")
        ADATA = training_data_internal_test()
        OUTDIR = OUTDIR.joinpath("test")
        OUTDIR.mkdir(exist_ok=True, parents=True)
        MODELS = {k: v for k, v in MODELS.items() if k == "clr_random_forest_minfo"}
    else:
        ADATA = training_data_internal()
        OUTDIR.mkdir(exist_ok=True, parents=True)
    label_class = args.label_class
    if args.dask:
        cluster = SLURMCluster(cores=int(args.cores), memory=f"{args.memory} GB")
        client = Client(cluster)
    backend = "dask" if args.dask else "loky"
    par_args = {"wait_for_workers_timeout": 0} if args.dask else {"n_jobs": args.cores}
    USE_CACHED = args.cached

    with joblib.parallel_backend(backend):
        for name, data in MODELS.items():
            model, transformation, imputation, features, references = (
                data.get("model"),
                data.get("t"),
                data.get("i"),
                data.get("f"),
                data.get("r"),
            )
            cross_validate_helper(
                lc=label_class,
                gc=None,
                model=model,
                result_dir_str=name,
                trans=transformation,
                feature_set=features,
                impute=imputation,
                references=references,
            )
            # for g in group_classes:
            #     cross_validate_helper(lc=label_class, gc=g, model=model, result_dir_str=name)
