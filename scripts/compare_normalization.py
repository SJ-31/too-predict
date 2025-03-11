#!/usr/bin/env ipython

import logging

#!/usr/bin/env python
import traceback
from datetime import datetime
from pathlib import Path

import anndata as ad
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scanpy as sc
import sklearn.metrics as sm
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from pyhere import here
from too_predict.filter import Filter
from too_predict.imputer import IMPLEMENTED_IMPUTATION, Imputer
from too_predict.simulation import IMPLEMENTED_SIMULATION
from too_predict.transformer import IMPLEMENTED_TRANSFORMATION, Transformer
from too_predict.utils import (
    adata_x_to_r,
    df_to_r,
    read_existing,
    ref_feature_lists_internal,
    source,
    training_data_internal,
    training_data_internal_test,
)

LOGGER = logging.getLogger()
DATE = datetime.today().strftime("%Y-%m-%d")

DATADIR = here("data", "tests")
OUTDIR: Path = here("data", "output", "normalization_comparison")
FS_DIR = here("data", "output", "feature_selection")
STORAGE_DIR = here("remote", "repos", "too-predict", "normalization_comparison")
logging.basicConfig(filename=here(OUTDIR, "log"))
FAILED_TRACKER = {"normalization": [], "imputation": [], "reason": []}


def helper(
    feature_set, adata, i: str, n: str, features=None, **kwargs
) -> ad.AnnData | None:
    if i == "labelled_median":
        impute_fn = Imputer(i, labels=adata.obs["tumor_type"])
    else:
        impute_fn = Imputer(i)
    dir = here(STORAGE_DIR, feature_set)
    dir.mkdir(exist_ok=True)
    output = here(dir, f"{n}-{i}.h5ad")
    if not output.exists():
        if features is not None:
            adata = Filter(features, feature_col="GENEID").fit_transform(adata)
        normalized: ad.AnnData = Transformer(
            n, impute_fn=impute_fn, inplace=False, make_sparse=False, **kwargs
        ).fit_transform(adata)
        if n in IMPLEMENTED_SIMULATION:
            normalized.X = normalized.uns["mc_instances"][0, :, :]
        sc.pp.pca(normalized)
        if "counts" in normalized.layers:
            del normalized.layers["counts"]
        sc.pp.neighbors(normalized)
        sc.tl.umap(normalized)
        normalized.write_h5ad(output)
        return normalized
    else:
        adata = ad.read_h5ad(output)
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(adata)
        if "umap" not in adata.uns:
            sc.tl.umap(adata)
            adata.write_h5ad(output)
        return adata


# #  --- CODE BLOCK ---

VARS = ["tumor_type", "primary_site"]
ALL_METRICS = []


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-c", "--cores", default=8, type=int)
    parser.add_argument("-f", "--feature_file", required=False)
    parser.add_argument(
        "-p", "--profile", required=False, action="store_true", default=False
    )
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-n", "--no_dask", default=False, action="store_true")
    return parser.parse_args()


VAR_EXPLAINED: list = []
NORMALIZATION_METHODS = IMPLEMENTED_TRANSFORMATION
IMPUTATION_METHODS = ["plus_one"]
METRIC_OUTPUT = here(OUTDIR, "label_metrics.csv")
REF_LISTS, FEATURE_LISTS = ref_feature_lists_internal()
NORMALIZATION_METHODS = NORMALIZATION_METHODS | set(FEATURE_LISTS.keys())


def get_metrics(adata: ad.AnnData, label, normalization, imputation, feature_set, f):
    df = pd.DataFrame(
        {
            "label": label,
            "feature_set": feature_set,
            "normalization": normalization,
            "imputation": imputation,
        },
        index=[0],
    )
    df["silhouette_score"] = sm.silhouette_score(adata.X, adata.obs[label])
    counts = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    df["davies_bouldin_score"] = sm.davies_bouldin_score(counts, adata.obs[label])
    df["calinski_harabasz_score"] = sm.calinski_harabasz_score(counts, adata.obs[label])
    df.to_csv(f, index=False)
    return df


def main(adata, feature_set_name):
    outdir = here(OUTDIR, feature_set_name)
    outdir.mkdir(exist_ok=True)
    for i in IMPUTATION_METHODS:
        for n in NORMALIZATION_METHODS:
            if (n is not None and "alr" in n) or i is None or n == "dirichlet_scale":
                continue
            elif n == "dirichlet" and feature_set_name != "all_features":
                normalized = helper(
                    feature_set=feature_set_name,
                    adata=adata,
                    i=i,
                    n=n,
                    features=None,
                    n_instances=1,
                )
            elif n in FEATURE_LISTS:  # clr with feature subset
                normalized = helper(
                    feature_set=feature_set_name,
                    adata=adata,
                    i=i,
                    n="clr",
                    features=FEATURE_LISTS[n],
                )
            else:
                normalized = helper(
                    feature_set=feature_set_name, adata=adata, i=i, n=n, features=None
                )
            shape_info = f"Shape after helper call: {normalized.shape}"
            LOGGER.info(shape_info)
            if normalized is not None:
                if "umap" not in normalized.uns:
                    warn = (
                        f"WARNING: combination {n} {i} {feature_set_name} has no UMAP!"
                    )
                    LOGGER.warning(warn)
                for v in VARS:
                    if not ((var_dir := here(outdir, v)).exists()):
                        var_dir.mkdir(exist_ok=True)
                    VAR_EXPLAINED.append(
                        [n, i] + list(normalized.uns["pca"]["variance_ratio"])
                    )
                    filename = here(var_dir, f"{i}_{n}_pca.png")
                    u_filename = here(var_dir, f"{i}_{n}_umap.png")

                    current_metrics_file = here(outdir, f".{n}_{i}_{v}_metrics.csv")
                    metric_df = read_existing(
                        current_metrics_file,
                        lambda f: get_metrics(
                            normalized,
                            label=v,
                            feature_set=feature_set_name,
                            normalization=n,
                            imputation=i,
                            f=f,
                        ),
                        pd.read_csv,
                    )

                    ALL_METRICS.append(metric_df)
                    colors = [v, "Project_ID", "Sample_Type"]
                    fig = sc.pl.pca(
                        normalized, color=colors, return_fig=True, legend_loc=None
                    )
                    fig.set_size_inches(15, 10)
                    fig.savefig(filename, dpi=500, bbox_inches="tight")

                    u_fig = sc.pl.umap(
                        normalized, color=colors, return_fig=True, legend_loc=None
                    )
                    u_fig.set_size_inches(15, 10)
                    u_fig.savefig(u_filename, dpi=500, bbox_inches="tight")

                    if feature_set_name != "all_features":
                        source("plotting.R", True)
                        df_to_r(normalized.obs, r_symbol="obs")
                        adata_x_to_r(normalized, "counts")
                        heatmap_file = here(var_dir, f"{i}_{n}_heatmap.png")
                        fncall = f"""
                        pheatmap_helper(obs=obs, counts=counts,
                            sample_annotations = list(tumor_type = NULL, Sample_Type = NULL),
                            order_on = 'tumor_type',
                            pheatmap_kwargs = list(file = '{heatmap_file}',
                            show_rownames = FALSE,
                            show_colnames = FALSE))
                        """
                        ro.r(fncall)
                        pass
                    plt.close()

        all_df = pd.concat(ALL_METRICS, ignore_index=True)
        var_df = pd.DataFrame(
            np.array(VAR_EXPLAINED),
            columns=["normalization", "imputation"] + [f"PC{i + 1}" for i in range(50)],
        )
        failed_df = pd.DataFrame(FAILED_TRACKER)

        failed_df.to_csv(here(outdir, "failed.csv"), index=False)
        var_df.to_csv(here(outdir, "variance_explained.csv"), index=False)
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
    print(f"Original data shape {adata.shape}")
    backend = "dask" if not args.no_dask else "loky"
    par_args = (
        {"wait_for_workers_timeout": 0} if not args.no_dask else {"n_jobs": args.cores}
    )
    date_file = OUTDIR.joinpath(f"RECORD_{DATE}")
    try:
        with joblib.parallel_backend(backend, **par_args):
            for fname, lst in FEATURE_LISTS.items():

                def run():
                    if lst is None:
                        main(adata, fname)
                    else:
                        mask = adata.var["GENEID"].isin(lst)
                        main(adata[:, mask], fname)

                run()
        date_file.write_text("Completed")
    except Exception:
        date_file.write_text(f"FAILED WITH EXCEPTION\n\n{traceback.format_exc()}")
