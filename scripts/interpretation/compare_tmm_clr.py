#!/usr/bin/env python

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import plotnine as gg
import sklearn.metrics as met
import too_predict.filter as fil
import too_predict.transformer as tt
import too_predict.utils as ut
from pyhere import here
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from too_predict.model import XGBEstimator

outdir: Path = here("data", "output", "normalization_comparison", "tmm_vs_clr")
outdir.mkdir(exist_ok=True)

RNG: np.random.Generator = np.random.default_rng(30110)

REF, FEAT = ut.ref_feature_lists_internal()

print("COMPARISON SCRIPT STARTED")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", default=False, help="Test", action="store_true")
    parser.add_argument(
        "-b",
        "--bootstrap",
        default=100,
        help="Number of bootstrap rounds",
        action="store",
    )
    parser.add_argument(
        "-p",
        "--per_type",
        default=100,
        help="Number of samples per tumor type",
        action="store",
    )
    parser.add_argument(
        "-n",
        "--n_features",
        default=3000,
        help="Keep this number of features with the highest variances",
        action="store",
    )
    parser.add_argument("-c", "--compare", default="clr", help="", action="store")
    args = vars(parser.parse_args())  # convert to dict
    return args


# Metrics will include rand
METRICS = {
    "adjusted_rand": (met.adjusted_rand_score, True),
    "normalized_mutual_info_score": (met.normalized_mutual_info_score, True),
    "silhouette": (met.silhouette_score, False),
    "ch_index": (met.calinski_harabasz_score, False),
}

CLUSTERINGS: tuple = (KMeans, AgglomerativeClustering)
MODELS: tuple = (
    ("xgboost", XGBEstimator, {}),
    ("logistic regression", LogisticRegression, {"solver": "saga"}),
    ("random forest", RandomForestClassifier, {}),
)


def get_transformed(test, compare="clr"):
    if test:
        adata = ut.training_data_internal_test()
    else:
        adata = ut.training_data_internal()

    filter = fil.Filter(
        features=FEAT["edgeR_median_lfc_feature_list_3000"], feature_col="GENEID"
    )
    adata = filter.transform(adata)
    tmm = tt.Transformer("tmm", "plus_one")
    tmm_transformed = tmm.fit_transform(adata)
    if compare == "none":
        return adata, tmm_transformed
    other = tt.Transformer(compare, "plus_one")
    transformed = other.fit_transform(adata)
    return transformed, tmm_transformed


def main(iter: int, adata: ad.AnnData, per_type: int, n_genes: int):
    results: dict = {
        "metric": [],
        "normalization": [],
        "method": [],
        "value": [],
        "iteration": [],
    }
    unique_types = set(adata.obs["tumor_type"])
    type_spec = {"tumor_type": [(ttype, per_type) for ttype in unique_types]}
    idx = ut.adata_sample_by(adata, label_spec=type_spec, rng=RNG)
    subset = adata[idx, :]
    assert subset.shape[0] == len(unique_types) * per_type

    # TODO: add model evaluation here
    vt = VarianceThreshold()
    vt.fit(ut.xarray_if_sparse(subset))
    gene_vars = pd.Series(vt.variances_, index=subset.var_names).sort_values(
        ascending=False
    )
    kept_genes = set(gene_vars.head(n_genes).index)
    subset = subset[:, subset.var.index.isin(kept_genes)]

    y_true = subset.obs["tumor_type"]
    n_ttypes = len(set(y_true))
    normalizations = ("clr", "fpkm", "tmm", "tpm", "none")
    eval_results = {
        "model": [],
        "normalization": [],
        "iteration": [],
        "accuracy": [],
        "dataset": [],
    }
    for n in normalizations:
        if n != "none":
            transformer = tt.Transformer(n, "plus_one", inplace=False)
            transformed = transformer.fit_transform(subset)
        else:
            transformed = subset.copy()

        x = ut.xarray_if_sparse(transformed)
        for alg in CLUSTERINGS:
            y_pred = alg(n_clusters=n_ttypes).fit_predict(x)
            for name, (fn, is_external) in METRICS.items():
                if is_external:
                    metric_val = fn(y_pred, y_true)
                else:
                    metric_val = fn(x, y_pred)
                results["method"].append(alg.__name__)
                results["value"].append(metric_val)
                results["normalization"].append(n)
                results["metric"].append(name)
                results["iteration"].append(iter)

        train_idx, test_idx = next(ShuffleSplit().split(x))
        x_train, x_test = x[train_idx, :], x[test_idx, :]
        y_train, y_test = (
            y_true[train_idx],
            y_true[test_idx],
        )
        for name, cls, kws in MODELS:
            model = cls(**kws)
            model.fit(x_train, y_train)
            # Since labels are balanced, accuracy is fine here
            train_acc = met.accuracy_score(
                y_true=y_train, y_pred=model.predict(x_train)
            )
            test_acc = met.accuracy_score(y_true=y_test, y_pred=model.predict(x_test))
            for t, score in zip(["train", "test"], [train_acc, test_acc]):
                eval_results["model"].append(name)
                eval_results["dataset"].append(t)
                eval_results["accuracy"].append(score)
                eval_results["iteration"].append(iter)
                eval_results["normalization"].append(n)
    return pd.DataFrame(results), pd.DataFrame(eval_results)


if __name__ == "__main__":
    args = parse_args()
    if args["test"]:
        adata: ad.AnnData = ut.training_data_internal_test()
    else:
        adata = ut.training_data_internal()
        counts = adata.obs["tumor_type"].value_counts()
        to_remove = counts[counts < args["per_type"]].index.to_list()
        adata = adata[~adata.obs["tumor_type"].isin(to_remove), :]
        print(f"n = {len(adata.obs['tumor_type'].unique())} different tumor types")
    cluster_dfs, classify_dfs = [
        main(iter=i, adata=adata, per_type=args["per_type"], n_genes=args["n_features"])
        for i in range(args["bootstrap"])
    ]
    for group, dfs in zip(["cluster", "classification"], [cluster_dfs, classify_dfs]):
        df: pd.DataFrame = pd.concat(cluster_dfs)
        agg = (
            df.drop("iteration", axis="columns")
            .groupby(["metric", "normalization", "method"])
            .agg(["mean", "median", "std"])
            .reset_index()
        )
        agg.columns = [c[1] or c[0] for c in agg.columns.to_flat_index()]
        agg.to_csv(outdir / f"{group}_results_agg.csv", index=False)
        df.to_csv(outdir / f"{group}_all_results.csv", index=False)

        if group == "cluster":
            fill = "method"
            y = "value"
            facet = "metric"
        else:
            fill = "model"
            y = "accuracy"
            facet = "dataset"
        plot = (
            gg.ggplot(df, gg.aes(x="normalization", fill=fill, y=y))
            + gg.geom_boxplot()
            + gg.facet_wrap(facet, scales="free_y")
        )
        plot.save(outdir / f"{group}_metrics.png", height=9, width=13)

# See /home/shannc/Bio_SDD/too-predict/data/output/normalization_comparison/tmm_vs_clr
# TODO: aggregate and report as a table
