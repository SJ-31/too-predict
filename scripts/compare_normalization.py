#!/usr/bin/env ipython

import logging
import pickle

import anndata as ad
import joblib
import pandas as pd
import scanpy as sc
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from pyhere import here
from sklearn.cluster import KMeans
from too_predict.imputer import IMPLEMENTED_IMPUTATION, Imputer
from too_predict.normalizer import IMPLEMENTED_NORMALIZATION, Normalizer
from too_predict.utils import (
    add_gene_metadata,
    cluster_gini,
    dgelist2anndata,
    training_data_internal,
    write_pickle,
)

logger = logging.getLogger()

datadir = here("data", "tests")
outdir = here("data", "output", "normalization_comparison")
logging.basicConfig(filename=here(outdir, "log"))

# #  --- CODE BLOCK ---
TEST = False
if TEST:
    hcc = here(datadir, "tcga_hcc.rds")
    chol = here(datadir, "tcga_chol.rds")
    coad = here(datadir, "tcga_coad-read.rds")
    test_sets = {"LIHC": hcc, "CHOL": chol, "COAD": coad}

    def loader(path, type):
        adata = dgelist2anndata(str(path))
        adata = adata[:100]
        adata.obs["tumor_type"] = type
        return adata

    adata: ad.AnnData = ad.concat([loader(t, p) for p, t in test_sets.items()])
    adata.var.index = adata.var.index.to_series().str.replace("\\..*", "", regex=True)
    add_gene_metadata(adata)
else:
    adata = training_data_internal()

cluster = SLURMCluster(cores=8, memory="25 GB")
client = Client(cluster)

date = "Wednesday_Feb-26-2025"
vars = ["tumor_type", "primary_site"]
all_ginis = []
with joblib.parallel_backend("dask"):
    for i in IMPLEMENTED_IMPUTATION:
        for n in IMPLEMENTED_NORMALIZATION:
            if "alr" in n or i is None:
                continue
            if i == "labelled_median":
                impute_fn = lambda x: Imputer(i).run(x, labels=adata.obs["tumor_type"])
            else:
                impute_fn = Imputer(i).run
            normalized: ad.AnnData = Normalizer(
                adata,
                method=n,
                impute_fn=impute_fn,
                make_sparse=False,
                inplace=False,
            ).run()

            try:
                sc.pp.pca(normalized)
                pca_file = here(outdir, f"{date}-{n}-{i}-pca.pickle")
                pca_dict = {
                    "PCs": normalized.varm["PCs"],
                    "pca": normalized.uns["pca"],
                    "X_pca": normalized.obsm["X_pca"],
                }
                write_pickle(pca_dict, pca_file)
                for v in vars:
                    if not ((var_dir := here(outdir, v)).exists()):
                        var_dir.mkdir(exist_ok=True)
                    filename = here(var_dir, f"{date}-{i}_{n}.png")
                    n_clusters = len(normalized.obs[v].unique())
                    kmm = KMeans(n_clusters=n_clusters)
                    assignments = kmm.fit_predict(normalized.X)
                    normalized.obs["kmm"] = assignments

                    ginis, whole = cluster_gini(normalized, "kmm", v)
                    gini_df = pd.DataFrame(ginis, index=[0])
                    gini_df["whole"] = whole
                    gini_df["label"] = v
                    gini_df["normalization"] = n
                    gini_df["imputation"] = i
                    all_ginis.append(gini_df)
                    fig = sc.pl.pca(normalized, color=[v, "kmm"], return_fig=True)
                    fig.set_size_inches(15, 15)
                    fig.savefig(filename, dpi=500, bbox_inches="tight")
            except ValueError as e:
                logger.error(f"ValueError with imputation {i} and normalization {n}")
                logger.error(e)

    all_df = pd.concat(all_ginis, ignore_index=True)
    all_df.to_csv(here(outdir, f"{date}-gini_impurity.csv"), index=False)
