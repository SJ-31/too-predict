#!/usr/bin/env ipython

from pathlib import Path

import anndata as ad
import joblib
import pandas as pd
import scanpy as sc
import sklearn.inspection as si
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from pyhere import here
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.transformer import Transformer
from too_predict.utils import (
    RANDOM_STATE,
    RNG,
    ref_feature_lists_internal,
    training_data_internal,
    training_data_internal_test,
)

OUTDIR: Path = here("data", "output", "organoid_feature_selection")
OUTDIR.mkdir(exist_ok=True, parents=True)
REF_LISTS, FEATURE_LISTS = ref_feature_lists_internal()

TEST = True


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-c", "--cores", default=8)
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-d", "--dask", default=False, action="store_true")
    parser.add_argument("-a", "--cached", default=False, action="store_true")
    return parser.parse_args()


def main():
    if TEST:
        adata = training_data_internal_test()
    else:
        adata = training_data_internal()

    model = HistGradientBoostingClassifier()

    filter = Filter(
        feature_col="GENEID",
        features=FEATURE_LISTS["edgeR_median_lfc_feature_list_3000"],
    )
    adata = adata[adata.obs["Sample_Type"].isin(["organoid", "primary"])]
    adata = filter.fit_transform(adata)
    if TEST:
        adata = adata[:50, :50]

    Transformer("clr", impute_fn=Imputer("plus_one"), inplace=True).fit_transform(adata)
    adata.obs["is_organoid"] = adata.obs["Sample_Type"] == "organoid"
    primary = adata[~adata.obs["is_organoid"], :]
    sc.pp.subsample(primary, random_state=RANDOM_STATE, fraction=0.03)
    organoid = adata[adata.obs["is_organoid"], :]

    adata = ad.concat([primary, organoid], axis="obs", join="inner", merge="same")
    print(f"Shape after balancing {adata.shape}")
    labels = adata.obs["is_organoid"]
    counts = adata.X.toarray()
    # scorer = sm.make_scorer(sm.cohen_kappa_score)
    x_train, x_test, y_train, y_test = train_test_split(counts, labels)
    model.fit(x_train, y_train)

    results = si.permutation_importance(
        model,
        x_train,
        y_train,
        n_repeats=3 if not TEST else 1,
        random_state=RNG,
    )
    # [2025-03-13 Thu] Why are these all 0?
    # cross validation results indicate that they can learn this

    cv_score = cross_val_score(model, x_train, y_train)
    print(cv_score)
    print(results)
    imp = results["importances"]

    df = pd.DataFrame(
        dict(
            {k: v for k, v in results.items() if k != "importances"},
            **{"GENEID": adata.var["GENEID"]},
        )
    )

    importance = pd.DataFrame(imp, columns=[f"repeat_{i}" for i in range(imp.shape[1])])
    df = (
        pd.concat([df, importance], axis=1, ignore_index=True)
        .reset_index(drop=True)
        .set_axis(list(df.columns) + list(importance.columns), axis=1)
    )
    df.to_csv(OUTDIR.joinpath("importances.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    TEST = args.test
    if args.dask:
        cluster = SLURMCluster(cores=int(args.cores), memory=f"{args.memory} GB")
        client = Client(cluster)
    backend = "dask" if args.dask else "loky"
    par_args = {"wait_for_workers_timeout": 0} if args.dask else {"n_jobs": args.cores}
    with joblib.parallel_backend(backend, **par_args):
        main()
