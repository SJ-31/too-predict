#!/usr/bin/env ipython

import anndata as ad
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.feature_selection as fs
import too_predict._rust_helpers as rh
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from pyhere import here
from sklearn.ensemble import RandomForestClassifier
from too_predict.imputer import Imputer
from too_predict.model import RandomForestPred
from too_predict.normalizer import Normalizer
from too_predict.utils import (
    read_existing,
    training_data_internal,
    training_data_internal_test,
)

outdir = here("data", "output", "feature_selection")
adata: ad.AnnData


# * Output files
low_variance_file = here(outdir, "sklearn_low_variance.csv")
proportionality_file = here(outdir, "proportionality_matrix.csv")
mutual_info_file = here(outdir, "mutual_info.csv")

# * Identify stable features for ALR

# ** Variance-based


def variance_threshold(f):
    vt = fs.VarianceThreshold(threshold=4)
    _ = vt.fit_transform(n_counts)
    low_variance = adata.var.loc[~vt.get_support(), :]
    low_variance.loc[:, "variance"] = np.nanvar(n_counts, axis=0)[~vt.get_support()]
    low_variance = low_variance.sort_values("variance")
    sns.histplot(low_variance["variance"])
    plt.savefig(here(outdir, "sklearn_variance.png"))
    low_variance.to_csv(f, index=False)


# ** Highest proportionality
def get_proportionality(f):
    pairwise = rh.rho_matrix(n_counts, True)
    pair_df = pd.DataFrame(
        pairwise, columns=adata.var["GENEID"], index=adata.var["GENEID"]
    )
    central_prop = pair_df.median().sort_values(ascending=False)
    flattened_pairs = [
        (f"{i} - {j}", pair_df.loc[i, j])
        for i in pair_df.columns
        for j in pair_df.index
        if i != j
    ]
    best_pairs = sorted(flattened_pairs, key=lambda x: x[1], reverse=True)
    pair_df.to_csv(f, index=False)


# * Identify useful features


# <2025-02-21 Fri> but why identify features with a different model than the one
# you are going to eventually train on?
def sfm(f):
    forest = RandomForestPred("clr", "plus_one")
    forest.fit(adata, label_col="primary_site")
    model = fs.SelectFromModel(forest, prefit=True)
    selected = model.get_support()
    kept_features = forest.var.loc[selected, :]
    kept_features.loc[:, "importance"] = forest.feature_importances_[selected]
    kept_features.to_csv()


# ** Information
# Try out information gain method
# gene expression data is continuous though
def entropy(y, base=2) -> float:
    p = pd.Series(y).value_counts() / len(y)
    entropy = -np.sum(p * np.emath.logn(base, p))
    return entropy


def mutual_info(f):
    info = fs.mutual_info_classif(n_counts, labels)
    info_df = pd.DataFrame({"feature": adata.var["GENEID"], "mutual_info": info})
    info_df.sort_values("mutual_info", ascending=False, inplace=True)
    info_df.to_csv(f, index=False)
    # The higher the better here


# ** Recursive
#
# <2025-02-21 Fri> define a different shuffle split
# can compare this with different estimators
# also see if you can make this compatible with the ALR model
# cv =
# <2025-02-24 Mon> this will take forever to run, you should narrow done the list
# using a threshold of some kind
def recursive_selection(f, est, prefix: str):
    rfecv = fs.RFECV(estimator=RandomForestClassifier(), verbose=1)
    rfecv.fit(n_counts, y=labels)
    results = rfecv.cv_results_
    kept_features = adata.var.loc[rfecv.get_support(), :]


# * Run


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-c", "--cores", default=8)
    parser.add_argument("-n", "--no_dask", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        print("Using test subset")
        adata = training_data_internal_test()
        outdir = outdir.joinpath("test")
        outdir.mkdir(exist_ok=True, parents=True)
    else:
        adata = training_data_internal()

    # <2025-02-21 Fri> when you run the real thing, choose the best normalization method
    # or do it in a loop and see what happens
    normalized = Normalizer(adata, "clr", Imputer("plus_one").run, inplace=False).run()
    n_counts = normalized.X.toarray()
    labels = adata.obs["primary_site"]
    # Need to normalize first to move out of simplex

    cluster = SLURMCluster(cores=int(args.cores), memory=f"{args.memory} GB")
    client = Client(cluster)
    backend = "dask" if not args.no_dask else "loky"
    with joblib.parallel_backend(backend):
        low_variance = read_existing(low_variance_file, variance_threshold, pd.read_csv)
        mutual_info = read_existing(mutual_info_file, mutual_info, pd.read_csv)
        prop = read_existing(proportionality_file, get_proportionality, pd.read_csv)

        # TODO: add in recursive selection
        # rfecv = fs.RFECV(estimator=RandomForestPred("clr", "plus_one"), verbose=1)
        # rfecv.fit(adata, y=labels)
