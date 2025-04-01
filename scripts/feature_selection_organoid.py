#!/usr/bin/env ipython

from pathlib import Path

import anndata as ad
import joblib
import pandas as pd
import scanpy as sc
import sklearn.feature_selection as fs
import sklearn.inspection as si
import sklearn.metrics as sm
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from pyhere import here
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
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
    parser.add_argument("-c", "--cores", default=8, type=int)
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-d", "--dask", default=False, action="store_true")
    parser.add_argument("-a", "--cached", default=False, action="store_true")
    return parser.parse_args()


def main():
    if TEST:
        adata = training_data_internal_test()
    else:
        adata = training_data_internal()

    model = RandomForestClassifier()

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
    scorer = sm.make_scorer(sm.cohen_kappa_score)
    x_train, x_test, y_train, y_test = train_test_split(counts, labels)
    model.fit(x_train, y_train)
    adata.var.loc[:, "raw_importance"] = model.feature_importances_
    adata.var.to_csv(OUTDIR.joinpath("raw_importances.csv"), index=False)

    # [2025-03-13 Thu] Results were all zero with permutation_importance
    rfecv = fs.RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring=scorer)
    rfecv.fit(counts, labels)

    cv_score = cross_val_score(model, x_train, y_train)
    print(cv_score)

    df = pd.DataFrame({"GENEID": adata.var["GENEID"], "ranking": rfecv.ranking_})
    df.to_csv(OUTDIR.joinpath("importances.csv"), index=False)
    score_df = pd.DataFrame(rfecv.cv_results_)
    score_df.to_csv(OUTDIR.joinpath("cv_results.csv"), index=False)


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
