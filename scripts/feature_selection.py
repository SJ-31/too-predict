#!/usr/bin/env ipython

import anndata as ad
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.feature_selection as fs
import too_predict._rust_helpers as rh
from pyhere import here
from sklearn.ensemble import RandomForestClassifier
from too_predict.evaluation import cross_validate, write_cross_val
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.model import PredBase, RandomForestPred, XGBEstimator
from too_predict.transformer import Transformer
from too_predict.utils import (
    read_existing,
    recode_to_go,
    ref_feature_lists_internal,
    training_data_internal,
    training_data_internal_test,
)

outdir = here("data", "output", "feature_selection")
adata: ad.AnnData

RECODE_GO: bool = False

# * Output files

# * Identify stable features for ALR

# ** Variance-based


def variance_threshold(f):
    vt = fs.VarianceThreshold(threshold=4)
    _ = vt.fit_transform(n_counts)
    variance = adata.var.loc[~vt.get_support(), :]
    variance.loc[:, "variance"] = np.nanvar(n_counts, axis=0)[~vt.get_support()]
    variance = variance.sort_values("variance")
    sns.histplot(variance["variance"])
    if RECODE_GO:
        name = "sklearn_variance_go.png"
        variance = variance.reset_index()
    else:
        name = "sklearn_variance.png"
    plt.savefig(here(outdir, name))
    variance.to_csv(f, index=False)


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
    forest.fit(adata, y="primary_site")
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


# ** Checking feature importances
# Will do this with Rfs and the xgbestimator
def tree_importance(adata, classifier: PredBase, outdir):
    _, feat = ref_feature_lists_internal()
    first_filter = Filter(
        feature_col="GENEID",
        features=feat["edgeR_median_lfc_feature_list_3000"],
        inplace=False,
    )
    transformer = Transformer("clr", impute_fn=Imputer("plus_one"), inplace=False)
    x: ad.AnnData = first_filter.fit_transform(adata)
    x = transformer.fit_transform(x)
    classifier.fit(x, y="tumor_type")
    x.var["importance"] = classifier.model.feature_importances_
    cv_results_1 = cross_validate(classifier, x, label_col="tumor_type")
    write_cross_val(cv_results_1, outdir, "cv_before", "_cv_before")

    nonzero = x.var.loc[x.var["importance"] != 0, :]

    nonzero_features = list(nonzero["GENEID"])
    nonzero.to_csv(here(outdir, "nonzero_features.csv"))

    sec_filter = Filter(feature_col="GENEID", features=nonzero_features)
    x2: ad.AnnData = sec_filter.fit_transform(adata)
    x2 = transformer.fit_transform(x2)
    cv_results_2 = cross_validate(classifier, x2, label_col="tumor_type")
    write_cross_val(cv_results_2, outdir, "cv_after", "_cv_after")


# * Run


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-c", "--cores", default=8)
    parser.add_argument(
        "-g", "--recode_go", default=False, help="", action="store_true"
    )
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

    RECODE_GO = args.recode_go
    suffix = ""
    if RECODE_GO:
        print("Recoding adata to GO...")
        adata = recode_to_go(adata)
        suffix = "GO"

    normalized = Transformer("clr", Imputer("plus_one"), inplace=False).fit_transform(
        adata
    )
    n_counts = normalized.X.toarray()
    labels = adata.obs["primary_site"]
    # Need to normalize first to move out of simplex

    variance_file = here(outdir, f"sklearn_variance_{suffix}.csv")
    print(f"Printing out {variance_file}")
    proportionality_file = here(outdir, f"proportionality_matrix_{suffix}.csv")
    mutual_info_file = here(outdir, f"mutual_info_{suffix}.csv")

    with joblib.parallel_backend("loky", n_jobs=args.cores):
        variance = read_existing(variance_file, variance_threshold, pd.read_csv)
        # mutual_info = read_existing(mutual_info_file, mutual_info, pd.read_csv)
        # prop = read_existing(
        #     proportionality_file, get_proportionality, pd.read_csv
        # )  # <2025-02-28 Fri> This fails, too much data?
        # tree_importance(
        #     adata, PredBase(model=XGBEstimator(importance_type="gain")), outdir
        # )
        # importance score with gain are the average gain across all trees

        # TODO: add in recursive selection
        # rfecv = fs.RFECV(estimator=RandomForestPred("clr", "plus_one"), verbose=1)
        # rfecv.fit(adata, y=labels)
