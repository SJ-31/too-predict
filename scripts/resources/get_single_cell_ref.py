#!/usr/bin/env ipython

from pathlib import Path
from typing import Callable

import anndata as ad
import pandas as pd
import rpy2.robjects as ro
import too_predict.r_utils as ru
import too_predict.utils as ut
from pyhere import here

test = False
if str(Path.home()) != "/home/shannc":
    storage_dir = here("remote", "public_data")
else:
    storage_dir = here("data", "tests", "scr_ref")

dirs = {
    "htc_atlas": here(storage_dir, "htca_2025-4-21"),
    "gtex": here(storage_dir, "GTEx_single_cell_2025-4-29"),
    "cellxgene": here(storage_dir, "cellxgene-census"),
}
to_ignore = {
    "htc_atlas": [
        "HTCA_ADULT_SMALL_INTESTINE.rds",
        "HTCA_ADULT_SPLEEN.rds",
        "HTCA_ADULT_STOMACH.rds",
    ],
    "gtex": [],
    "cellxgene": [],
}

shared_cols = ["tissue", "subject", "cell_type", "source"]


def merge_from_files(dir: Path, ignore_list, read_fn: Callable) -> ad.AnnData:
    files = [f for f in dir.iterdir() if f not in ignore_list]
    adatas = [read_fn(f) for f in files]
    if len(adatas) > 1:
        final = ad.concat(adatas, axis="obs", join="inner", merge="first")
    else:
        final = adatas[0]
    return final


def _dataset_id_map():
    df = pd.read_csv(here("data", "mappings", "cellxgene_datasets.csv"))
    val = df["collection_name"]
    val.index = df["dataset_id"]
    return val


dataset_id_map = _dataset_id_map()


def cellxgene_fn(file):
    adata = ad.read_h5ad(file)
    adata = adata[:, ~adata.var["feature_id"].isna()]
    # ut.preserving_sample(adata, "cell_type", 0.5)
    adata.var.index = adata.var["feature_id"]
    mapping = {
        "feature_id": "GENEID",
        "feature_type": "GENEBIOTYPE",
        "feature_length": "SEQLENGTH",
        "feature_name": " GENENAME",
    }
    adata.var = adata.var.rename(mapping, axis=1).loc[:, list(mapping.keys())]
    adata.obs.loc[:, "source"] = list(
        map(lambda x: f"cellxgene-{x}", dataset_id_map[adata.obs["dataset_id"]])
    )
    adata.obs.loc[:, "subject"] = adata.obs["dataset_id"].combine(
        adata.obs["donor_id"], lambda x, y: f"{x}-{y}"
    )
    adata.obs = adata.obs.loc[:, shared_cols]
    return adata


def gtex_fn(file):
    adata = ad.read_h5ad(file)
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"]
    wanted_cols = [
        "tissue",
        "Participant ID",
        "Cell types level 2",
        "batch",
        "prep",
        "Tissue Site Detail",
        "Broad cell type",
        "Granular cell type",
        "Tissue composition",
        "PercentMito",
        "PercentRibo",
        "scrublet",
    ]
    del adata.obsp
    del adata.obsm
    del adata.varm
    adata = adata[~(adata.obs["scrublet"] == "True"), :]
    adata.obs = adata.obs.loc[:, wanted_cols].rename(
        {"Participant ID": "subject", "Granular cell type": "cell_type"}, axis=1
    )
    if adata.shape[0] > 0:
        adata.obs.loc[:, "source"] = "GTEx"
    adata.obs = adata.obs.loc[:, shared_cols]
    adata = adata[:, ~adata.var["gene_ids"].isna()]
    adata.var.index = adata.var["gene_ids"]
    adata = adata[:, ~adata.var.index.duplicated()]
    mapping = {
        "gene_ids": "GENEID",
        "gene_name": "GENENAME",
        "gene_biotype": "GENEBIOTYPE",
        "gene_length": "SEQLENGTH",
    }
    adata.var = adata.var.rename(mapping, axis=1).loc[:, list(mapping.values())]
    adata = adata[:, ~adata.var.index.duplicated()]
    return adata


@ru.r_cleanup
def htca_fn(file):
    ro.r(f"obj <- readRDS('{str(file)}')")
    ro.r("counts <- t(as.matrix(SeuratObject::LayerData(obj)))")
    counts = ru.np_from_r(ro.r("counts"))
    var = ru.df_from_r(ro.r("obj[['RNA']][[]]"))
    obs = ru.df_from_r(ro.r("obj[[]]"))
    adata = ad.AnnData(X=counts, var=var, obs=obs)
    adata, _ = ut.rename_genes(adata, old="symbol", new="ensembl")
    adata.var.index.name = "gene_id"
    adata = adata[:, ~adata.var.index.duplicated()]
    ru.add_gene_metadata(
        adata,
        "",
        columns=("GENENAME", "GENEID", "GENEBIOTYPE", "SEQLENGTH"),
        keytype="GENEID",
    )
    adata.obs.loc[:, "source"] = ("htca-" + adata.obs["Project"]).tolist()
    adata.obs = adata.obs.rename(
        {"Tissue": "tissue", "Sample_ID": "subject", "Cell_Type": "cell_type"}, axis=1
    ).loc[:, shared_cols]
    return adata


def get_combined(f):
    read_fns = {"htc_atlas": htca_fn, "gtex": gtex_fn, "cellxgene": cellxgene_fn}
    adatas = {
        k: merge_from_files(v, to_ignore[k], read_fns[k]) for k, v in dirs.items()
    }
    final = ad.concat(adatas.values(), axis="obs", join="inner", merge="first")
    final.write_h5ad(f)
    return final


combined_file = here(storage_dir, "sc_ref_all.h5ad")
combined = ut.read_existing(combined_file, get_combined, ad.read_h5ad)
combined.obs.to_csv(here("data", "reference", "sc_ref_all_obs.csv"), index=False)
