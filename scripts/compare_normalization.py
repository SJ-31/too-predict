#!/usr/bin/env ipython

import logging

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.metrics as sm
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from pyhere import here
from sklearn.cluster import KMeans
from too_predict.imputer import IMPLEMENTED_IMPUTATION, Imputer
from too_predict.normalizer import IMPLEMENTED_NORMALIZATION, Normalizer
from too_predict.utils import (
    training_data_internal,
    training_data_internal_test,
)

logger = logging.getLogger()

datadir = here("data", "tests")
outdir = here("data", "output", "normalization_comparison")
storage = here("remote", "repos", "too-predict", "normalization_comparison")
logging.basicConfig(filename=here(outdir, "log"))


def helper(adata, i: str, n: str) -> ad.AnnData | None:
    if i == "labelled_median":
        impute_fn = lambda x: Imputer(i).run(x, labels=adata.obs["tumor_type"])
    else:
        impute_fn = Imputer(i).run
    output = here(storage, f"{date}-{n}-{i}.h5ad")
    if not output.exists():
        normalized: ad.AnnData = Normalizer(
            adata,
            method=n,
            impute_fn=impute_fn,
            make_sparse=False,
            inplace=False,
        ).run()
        try:
            sc.pp.pca(normalized)
            if "counts" in normalized.layers:
                del normalized.layers["counts"]
            normalized.write_h5ad(output)
            return normalized
        except ValueError as e:
            logger.error(f"ValueError with imputation {i} and normalization {n}")
            logger.error(e)
    else:
        return ad.read_h5ad(output)


# #  --- CODE BLOCK ---


date = "Thursday_Feb-27-2025"
vars = ["tumor_type", "primary_site"]
all_metrics = []


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-c", "--cores", default=8)
    parser.add_argument("-t", "--test", default=False, action="store_true")
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
    cluster = SLURMCluster(cores=int(args.cores), memory=f"{args.memory} GB")
    client = Client(cluster)
    backend = "dask" if not args.no_dask else "loky"
    with joblib.parallel_backend(backend):
        for i in IMPLEMENTED_IMPUTATION:
            for n in IMPLEMENTED_NORMALIZATION:
                if "alr" in n or i is None:
                    continue
                normalized = helper(adata, i=i, n=n)
                if normalized is not None:
                    try:
                        for v in vars:
                            if not ((var_dir := here(outdir, v)).exists()):
                                var_dir.mkdir(exist_ok=True)
                            filename = here(var_dir, f"{date}-{i}_{n}.png")
                            n_clusters = len(normalized.obs[v].unique())
                            kmm = KMeans(n_clusters=n_clusters)
                            assignments = kmm.fit_predict(normalized.X)
                            normalized.obs["kmm"] = assignments

                            metric_df = pd.DataFrame(
                                {"label": v, "normalization": n, "imputation": i},
                                index=[0],
                            )
                            metric_df["silhouette_score"] = sm.silhouette_score(
                                normalized.X, normalized.obs[v]
                            )
                            counts = (
                                normalized.X
                                if isinstance(normalized.X, np.ndarray)
                                else normalized.X.toarray()
                            )
                            metric_df["davies_bouldin_score"] = sm.davies_bouldin_score(
                                counts, normalized.obs[v]
                            )
                            metric_df["calinski_harabasz_score"] = (
                                sm.calinski_harabasz_score(counts, normalized.obs[v])
                            )
                            all_metrics.append(metric_df)
                            fig = sc.pl.pca(
                                normalized, color=[v, "kmm"], return_fig=True
                            )
                            fig.set_size_inches(15, 15)
                            fig.savefig(filename, dpi=500, bbox_inches="tight")
                    except ValueError as e:
                        logger.error(
                            f"ValueError with imputation {i} and normalization {n}"
                        )
                        logger.error(e)

        all_df = pd.concat(all_metrics, ignore_index=True)
        all_df.to_csv(here(outdir, f"{date}-gini_impurity.csv"), index=False)
