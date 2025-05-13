#!/usr/bin/env ipython

import re
from pathlib import Path
from typing import Callable

import anndata as ad
import pandas as pd
import scanpy as sc
import scib
import too_predict.r_utils as ru
import too_predict.utils as ut
from pyhere import here

test = False
if str(Path.home()) != "/home/shannc":
    storage_dir = here("remote", "public_data")
    batch_size = 10
else:
    storage_dir = here("data", "tests", "scr_ref")
    batch_size = 2

dirs = {
    # "htc_atlas": here(storage_dir, "htca_2025-4-21"), # [2025-05-09 Fri] OOM... even with 100 gigs assigned
    "gtex": here(storage_dir, "GTEx_single_cell_2025-4-29"),
    "cellxgene": here(storage_dir, "cellxgene-census"),
}
to_ignore = {
    "htc_atlas": [
        "HTCA_ADULT_SMALL_INTESTINE.rds",
        "HTCA_ADULT_SPLEEN.rds",
        "HTCA_ADULT_STOMACH.rds",
    ],
    "gtex": ["GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad"],
    "cellxgene": [],
}

shared_cols = ["tissue", "subject", "cell_type", "source"]


def merge_from_files(dir: Path, ignore_list, read_fn: Callable) -> ad.AnnData:
    files = [f for f in dir.glob("*.h5ad") if f.name not in ignore_list]
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
    adata.var = adata.var.rename(mapping, axis=1).loc[:, list(mapping.values())]
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


def htca_fn(file):
    adata = ad.read_h5ad(file)
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


read_fns = {"htc_atlas": htca_fn, "gtex": gtex_fn, "cellxgene": cellxgene_fn}

for k, v in dirs.items():
    output = v.joinpath("all.h5ad")
    if not output.exists():
        merged = merge_from_files(v, to_ignore[k], read_fn=read_fns[k])
        merged.write_h5ad(output)
        exit()


def get_combined(f):
    adatas = [ad.read_h5ad(v.joinpath("all.h5ad")) for v in dirs.values()]
    final = ad.concat(adatas, axis="obs", join="inner", merge="first")
    sc.pp.calculate_qc_metrics(final, inplace=True)
    print(final)
    final.write_h5ad(f)
    return final


def harmonize_labels_tissues(tissues) -> pd.Series:
    result = []
    for t in tissues:
        match t:
            case "esophagusmucosa" | "esophagusmuscularis":
                result.append("esophagus")
            case (
                "skeletalmuscle"
                | "skeletal muscle organ, vertebrate"
                | "skeletal muscle tissue"
                | "muscle organ"
            ):
                result.append("skeletal muscle")
            case "bone marrow":
                result.append("bone")
            case (
                "anterior segment of eyeball"
                | "posterior segment of eyeball"
                | "eyelid"
            ):
                result.append("eye")
            case lymph if "lymph node" in lymph:
                result.append("lymph")
            case ovary if "ovary" in ovary:
                result.append("ovary")
            case heart if "heart" in heart:
                result.append("heart")
            case "cortex of kidney":
                result.append("kidney")
            case _:
                result.append(t)
    return pd.Series(result)


def harmonize_labels_cells(cell_types) -> pd.Series:
    result = []
    dct = {
        "endothelial": "endothelial cell",
        "epithelial": "epithelial cell",
        "fibroblast": "fibroblast",
        "myocyte": "myocyte",
        "schwann cell": "schwann cell",
        "pericyte": "pericyte",
        "sebaceous gland cell": "sebaceous gland cell",
        "unknown": "unknown",
        "dendritic cell": "dendritic cell",
        "macrophage": "macrophage",
        "stromal cell": "stromal cell",
        "smooth muscle cell": "smooth muscle cell",
        "mucous cell": "mucous cell",
    }

    def match_replace(value):
        for k, v in dct.items():
            if k in value:
                return v

    for cell in cell_types:
        if "activated" in cell:
            cell = cell.replace("activated", "").strip()
        match cell:
            case immune if inner := re.findall(r"immune \((.*)\)", immune):
                match inner:
                    case [macrophage] if "macrophage" in macrophage:
                        result.append("macrophage")
                    case ["dc"]:
                        result.append("dendritic cell")
                    case ["nk cell"]:
                        result.append("natural killer cell")
                    case _:
                        result.append(inner[0])
            case pos if inner := re.search(r"cd[4816]+-positive,*.*", pos):
                renamed = pos.replace("-positive", "+").replace(",", "")
                result.append(renamed)
            case neg if inner := re.search(r"cd[4816]+-negative,*.*", neg):
                renamed = neg.replace("-negative", "+").replace(",", "")
                result.append(renamed)
            case matched if found := match_replace(matched):
                result.append(found)
            case "small pre-b-ii cell":
                result.append("pre-b-ii cell")
            case "type i nk t cell":
                result.append("nk t cell")
            case (
                "slow muscle cell"
                | "fast muscle cell"
                | "cell of skeletal muscle"
                | "muscle cell"
            ):
                result.append("myocyte")
            case _:
                result.append(cell)
    return pd.Series(result, index=cell_types)


def replace_cell_labels(adata) -> None:
    harmonized = harmonize_labels_cells(list(adata.obs["cell_type"].unique()))
    old = list(adata.obs["cell_type"])
    adata.obs.loc[:, "cell_type"] = list(harmonized[old])
    adata.obs.loc[:, "tissue_broad"] = list(
        harmonize_labels_tissues(adata.obs["tissue"])
    )
    adata.obs.loc[:, "cell_type_tissue"] = (
        adata.obs["tissue_broad"].astype(str) + "-" + adata.obs["cell_type"].astype(str)
    )


def average_within_source(adata: ad.AnnData) -> ad.AnnData:
    """Average gene expression profiles for each cell type within each source
    to make for easier integration
    """
    adatas = []
    for s in adata.obs["source"].unique():
        current = adata[adata.obs["source"] == s, :]
        averaged = sc.get.aggregate(current, by="cell_type_tissue", func="mean")
        averaged.X = averaged.layers["mean"]
        adatas.append(averaged)
    return ad.concat(adatas, axis="obs", join="inner", merge="first")


def get_scanorama(combined, f):
    replace_cell_labels(combined)
    ut.mad_outliers(
        combined, mode="cells", columns=["total_counts", "n_genes_by_counts"]
    )
    combined = combined[~combined.obs["is_mad_outlier"], :]
    sc.pp.filter_cells(combined, min_genes=1000)
    sc.pp.filter_genes(combined, min_cells=1000)
    ru.pooled_normalization(combined)
    combined = average_within_source(combined)

    print(combined.shape)
    method = "scanorama"
    corrected = ut.scanorama_correct(  # [2025-05-13 Tue] Got OOM
        combined, batch_key="source", batch_size=batch_size, hvg=4000
    )

    ut.pca_to_leiden(combined)
    ut.pca_to_leiden(corrected)

    scores = scib.metrics.metrics_fast(
        combined, corrected, batch_key="source", label_key="cell_type"
    )

    # metrics_fast only computes
    # - hvg_overlap
    #   1 is best
    # - cell type ASW (ASW_label), silhouette() function
    #   1 is best
    # - isolated_label_silhouette/isolated_labels()
    #   This is the same as cell type ASW, but considering only "isolated" labels, which
    #   are the cell types found in the fewest batches i.e. highly batch-specific cell types
    #   1 is best
    # - silhouette_batch() (ASW_label/batch)
    #   1 is best
    # - pcr_comparison() (PCR_batch)
    #   1 is best (greater difference between batches)
    # - graph_connectivity() (graph_conn)
    #   1 is best (all cells with same identity connected)
    scores = pd.DataFrame(
        {"metric": scores.index, "value": scores.iloc[:, 0]}
    ).reset_index(drop=True)
    scores.to_csv(here("data", "output", f"sc_ref_{method}_metrics.csv"), index=False)
    corrected.write_h5ad(f)
    return corrected


combined_file = here(storage_dir, "sc_ref_all.h5ad")
combined = ut.read_existing(combined_file, get_combined, ad.read_h5ad)
# combined.obs["cell_type"].value_counts().to_csv(
#     here("sc_cell_types.csv"), index_label="cell_type", header=["count"]
# )
combined.obs.to_csv(here("data", "reference", "sc_ref_all_obs.csv"), index=False)
scanorama_file = here(storage_dir, "sc_ref_all_corrected.h5ad")
scan = ut.read_existing(
    scanorama_file, lambda x: get_scanorama(combined, x), ad.read_h5ad
)
