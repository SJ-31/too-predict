#!/usr/bin/env ipython

import logging
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.metrics as sm
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from pyhere import here
from too_predict.imputer import IMPLEMENTED_IMPUTATION, Imputer
from too_predict.normalizer import IMPLEMENTED_NORMALIZATION, Normalizer
from too_predict.utils import (
    read_existing,
    take_from_ad,
    training_data_internal,
    training_data_internal_test,
)

logger = logging.getLogger()

DATADIR = here("data", "tests")
OUTDIR = here("data", "output", "normalization_comparison")
STORAGE_DIR = here("remote", "repos", "too-predict", "normalization_comparison")
logging.basicConfig(filename=here(OUTDIR, "log"))
failed_tracker = {"normalization": [], "imputation": [], "reason": []}


def helper(adata, i: str, n: str, **kwargs) -> ad.AnnData | None:
    if i == "labelled_median":
        impute_fn = lambda x: Imputer(i).run(x, labels=adata.obs["tumor_type"])
    else:
        impute_fn = Imputer(i).run
    output = here(STORAGE_DIR, f"{n}-{i}.h5ad")
    if not output.exists():
        normalized: ad.AnnData = Normalizer(
            adata,
            method=n,
            impute_fn=impute_fn,
            make_sparse=False,
            inplace=False,
        ).run(**kwargs)
        try:
            sc.pp.pca(normalized)
            if "counts" in normalized.layers:
                del normalized.layers["counts"]
            normalized.write_h5ad(output)
        except ValueError as e:
            logger.error(f"ValueError with imputation {i} and normalization {n}")
            failed_tracker["imputation"].append(i)
            failed_tracker["normalization"].append(n)
            failed_tracker["reason"].append(e)
            logger.error(e)
    else:
        adata = ad.read_h5ad(output)
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(adata)
        if "umap" not in adata.uns:
            sc.tl.umap(adata)
            adata.write_h5ad(output)
        return adata


# #  --- CODE BLOCK ---

DATE = "Monday_Mar-03-2025"
VARS = ["tumor_type", "primary_site"]
ALL_METRICS = []


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-c", "--cores", default=8, type=int)
    parser.add_argument("-f", "--feature_file", required=False)
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-n", "--no_dask", default=False, action="store_true")
    return parser.parse_args()


var_explained: list = []
NORMALIZATION_METHODS = IMPLEMENTED_NORMALIZATION
METRIC_OUTPUT: pd.DataFrame = pd.read_csv(here(OUTDIR, f"{DATE}-label_matrics.csv"))

# Adding selected features for CLR
REF_LISTS: dict = {}
for ref_list in here(
    "data", "output", "feature_selection", "reference_lists"
).iterdir():
    with open(ref_list, "r") as f:
        features = f.read().strip().splitlines()
    name = f"clr_{ref_list.stem}"
    NORMALIZATION_METHODS.append(name)
    REF_LISTS[name] = features


def get_metrics(adata: ad.AnnData, label, normalization, imputation, f):
    df = pd.DataFrame(
        {"label": label, "normalization": normalization, "imputation": imputation},
        index=[0],
    )
    df["silhouette_score"] = sm.silhouette_score(adata.X, adata.obs[label])
    counts = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    df["davies_bouldin_score"] = sm.davies_bouldin_score(counts, adata.obs[label])
    df["calinski_harabasz_score"] = sm.calinski_harabasz_score(counts, adata.obs[label])
    df.to_csv(f)
    return df


def main(adata):
    for i in IMPLEMENTED_IMPUTATION:
        for n in NORMALIZATION_METHODS:
            if "alr" in n or i is None:
                continue
            elif n in REF_LISTS:  # clr with feature subset
                normalized = helper(
                    adata, i=i, n="clr", kwargs={"features": REF_LISTS[n]}
                )
            else:
                normalized = helper(adata, i=i, n=n, kwargs={})
            if normalized is not None:
                for v in VARS:
                    if not ((var_dir := here(OUTDIR, v)).exists()):
                        var_dir.mkdir(exist_ok=True)
                    var_explained.append(
                        [[n, i] + list(normalized.uns["pca"]["variance_ratio"])]
                    )
                    filename = here(var_dir, f"{DATE}-{i}_{n}_pca.png")
                    u_filename = here(var_dir, f"{DATE}-{i}_{n}_umap.png")

                    current_metrics_file = here(OUTDIR, f".{n}_{i}_{v}_metrics.csv")
                    metric_df = read_existing(
                        current_metrics_file,
                        lambda f: get_metrics(
                            normalized, label=v, normalization=n, imputation=i, f=f
                        ),
                        pd.read_csv,
                    )

                    ALL_METRICS.append(metric_df)
                    fig = sc.pl.pca(normalized, color=v, return_fig=True)
                    fig.set_size_inches(10, 10)
                    fig.savefig(filename, dpi=500, bbox_inches="tight")

                    u_fig = sc.pl.umap(normalized, color=v, return_fig=True)
                    u_fig.set_size_inches(10, 10)
                    u_fig.savefig(u_filename, dpi=500, bbox_inches="tight")

        all_df = pd.concat(ALL_METRICS, ignore_index=True)
        var_df = pd.DataFrame(var_explained)
        failed_df = pd.DataFrame(failed_tracker)

        failed_df.to_csv(here(OUTDIR, f"{DATE}-failed.csv"), index=False)
        var_df.to_csv(here(OUTDIR, f"{DATE}-variance_explained.csv"), index=False)
        all_df.to_csv(METRIC_OUTPUT, index=False)


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        print("Using test subset")
        adata = training_data_internal_test()
        OUTDIR = OUTDIR.joinpath("test")
        OUTDIR.mkdir(exist_ok=True, parents=True)
    else:
        adata = training_data_internal()
    cluster = SLURMCluster(cores=args.cores, memory=f"{args.memory} GB")
    client = Client(cluster)
    backend = "dask" if not args.no_dask else "loky"
    par_args = (
        {"wait_for_workers_timeout": 0} if not args.no_dask else {"n_jobs": args.cores}
    )
    with joblib.parallel_backend(backend, **par_args):
        main(adata)
