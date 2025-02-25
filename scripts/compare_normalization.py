#!/usr/bin/env ipython

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from pyhere import here
from sklearn.cluster import KMeans
from too_predict.imputer import IMPLEMENTED_IMPUTATION, Imputer
from too_predict.normalizer import IMPLEMENTED_NORMALIZATION, Normalizer
from too_predict.utils import (
    add_gene_metadata,
    cluster_gini,
    dgelist2anndata,
)

datadir = here("data", "tests")
outdir = here("data", "output", "normalization_comparison")

# #  --- CODE BLOCK ---
TEST = True
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
else:
    adata: ad.AnnData = ad.read_h5ad()  # TODO: put in the combined file


add_gene_metadata(adata)  # Adds seqlengths among other things

date = ""
vars = ["tumor_type", "primary_site"]
all_ginis = []
for i in IMPLEMENTED_IMPUTATION:
    for n in IMPLEMENTED_NORMALIZATION:
        if "alr" in n or i is None:
            continue
        normalized: ad.AnnData = Normalizer(
            adata,
            method=n,
            impute_fn=Imputer(i).run,
            make_sparse=False,
            inplace=False,
        ).run()

        sc.pp.pca(normalized)
        for v in vars:
            if not (var_dir := here(outdir, v).exists()):
                var_dir.makdir(exist_ok=True)
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
            fig.savefig(filename)

all_df = pd.concat(all_ginis, ignore_index=True)
all_df.to_csv(outdir, f"{date}-gini_impurity.csv", index=False)
