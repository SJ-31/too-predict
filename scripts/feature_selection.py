#!/usr/bin/env ipython

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects as ro
import scanpy as sc
import seaborn as sns
import sklearn.feature_selection as fs
import too_predict._rust_helpers as rh
from pyhere import here
from rpy2.robjects.packages import importr
from sklearn.ensemble import RandomForestClassifier
from too_predict.imputer import Imputer
from too_predict.model import RandomForestPred
from too_predict.normalizer import Normalizer
from too_predict.utils import df_to_r, read_existing

# #  --- CODE BLOCK ---

date = "2025-02-25"  # Insert date
outdir = here("data", "output", "feature_selection")
TEST = True
if TEST:
    data_file = here("data", "tests", "TCGA_CESC-DLBC-ESCA-GBM.h5ad")
else:
    public_data = here("remote", "public_data")
    data_file = here(public_data, "all_tumors_rnaseq.h5ad")

adata = ad.read_h5ad(data_file)

if TEST:
    adata = adata[:, 0:100]

# <2025-02-21 Fri> when you run the real thing, choose the best normalization method
# or do it in a loop and see what happens
normalized = Normalizer(adata, "clr", Imputer("plus_one").run, inplace=False).run()
n_counts = normalized.X.toarray()
labels = adata.obs["primary_site"]
# Need to normalize first to move out of simplex
#
# #  --- CODE BLOCK ---
# * Identify stable features for ALR

# ** Variance-based

low_variance_file = here(outdir, f"sklearn_low_variance-{date}.csv")


def variance_threshold(f):
    vt = fs.VarianceThreshold(threshold=4)
    _ = vt.fit_transform(n_counts)
    low_variance = adata.var.loc[~vt.get_support(), :]
    low_variance.loc[:, "variance"] = np.nanvar(n_counts, axis=0)[~vt.get_support()]
    low_variance = low_variance.sort_values("variance")
    sns.histplot(low_variance["variance"])
    plt.savefig(here(outdir, f"sklearn_variance-{date}.png"))
    low_variance.to_csv(f, index=False)


# low_variance = read_existing(low_variance_file, variance_threshold, pd.read_csv)


# ** Highest proportionality
def get_proportionality(f):
    pairwise = rh.rho_matrix(n_counts, True)
    pair_df = pd.DataFrame(
        pairwise, columns=adata.var["gene_id"], index=adata.var["gene_id"]
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


# rfecv = fs.RFECV(estimator=RandomForestPred("clr", "plus_one"), verbose=1)
# rfecv.fit(adata, y=labels)
