#!/usr/bin/env ipython

from pathlib import Path

import pandas as pd
import sklearn.metrics as met
import too_predict.filter as fil
import too_predict.transformer as tt
import too_predict.utils as ut
from pyhere import here
from scipy.stats import false_discovery_control, wilcoxon
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler

outdir: Path = here("data", "output", "normalization_comparison", "tmm_vs_clr")
outdir.mkdir(exist_ok=True)

REF, FEAT = ut.ref_feature_lists_internal()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", default=False, help="Test", action="store_true")
    parser.add_argument("-c", "--compare", default="clr", help="", action="store")
    args = vars(parser.parse_args())  # convert to dict
    return args


# Metrics will include rand
metric_fns = {
    "adjusted_rand": met.adjusted_rand_score,
    "rand": met.rand_score,
    "normalized_mutual_info_score": met.normalized_mutual_info_score,
}


def get_transformed(test, compare="clr"):
    if test:
        adata = ut.training_data_internal_test()
    else:
        adata = ut.training_data_internal()

    filter = fil.Filter(
        features=FEAT["edgeR_median_lfc_feature_list_3000"], feature_col="GENEID"
    )
    adata = filter.transform(adata)
    other = tt.Transformer(compare, "plus_one")
    tmm = tt.Transformer("tmm", "plus_one")
    clr_transformed = other.fit_transform(adata)
    tmm_transformed = tmm.fit_transform(adata)
    return clr_transformed, tmm_transformed


if __name__ == "__main__":
    args = parse_args()
    # Compare another transformation against the baseline tmm
    other_transform = args["compare"]
    clr_transformed, tmm_transformed = get_transformed(args["test"], other_transform)

    result = {"clustering": []}
    result.update({k: [] for k in metric_fns.keys()})
    for method, kwargs in zip(
        [KMeans, AgglomerativeClustering], [{}, {"n_clusters": 8}]
    ):
        clr_clst = method(**kwargs).fit_predict(ut.xarray_if_sparse(clr_transformed))
        tmm_clst = method(**kwargs).fit_predict(ut.xarray_if_sparse(tmm_transformed))
        result["clustering"].append(method.__name__)
        for metric, fn in metric_fns.items():
            result[metric].append(fn(clr_clst, tmm_clst))
    df = pd.DataFrame(result)
    df.to_csv(
        outdir.joinpath(f"{other_transform}_clustering_comparison.csv"), index=False
    )

    n_features = clr_transformed.shape[1]
    other_arr = StandardScaler().fit_transform(ut.xarray_if_sparse(clr_transformed))
    tmm_arr = StandardScaler().fit_transform(ut.xarray_if_sparse(tmm_transformed))

    var: pd.DataFrame = clr_transformed.var
    # Test if features follow the same distribution with wilcoxon
    # Paired test because they are the same samples, under different transformations
    test_results = [
        wilcoxon(x, y, alternative="two-sided").pvalue
        for x, y in zip(other_arr.T, tmm_arr.T)
    ]
    adjusted = false_discovery_control(test_results, method="by")
    # Null hypothesis is that there is no difference, so
    no_diff = [p > 0.05 for p in adjusted]
    var.loc[:, "wilcoxon_p_adj"] = adjusted
    var.loc[:, "no_difference_between_transforms"] = no_diff
    var.to_csv(outdir.joinpath(f"{other_transform}_feature_results.csv"), index=False)
