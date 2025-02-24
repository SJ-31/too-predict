#!/usr/bin/env python

import logging
import re
from pathlib import Path
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from pyhere import here
from too_predict.utils import collect_gdc_counts, into_pseudobulks, read_existing

logger = logging.getLogger(__name__)
logging.basicConfig(filename=here("get_counts.log", level=logging.DEBUG))

public_data = here("remote", "public_data")
outdir = here(public_data, "h5ad")
metadata_dir = here(public_data, "metadata")
id_mapping_file = here("data", "Homo_sapiens.GRCh38.113.gene_id_mapping.tsv")

# * GDC downloads
tcga_cases = here(metadata_dir, "TCGA-case_table.tsv")
target_cases = here(metadata_dir, "TARGET-case_table.tsv")
cptac_cases = here(metadata_dir, "CPTAC-case_table.tsv")
cgci_cases = here(metadata_dir, "CGCI-case_table.tsv")

all_tcga = [
    "TCGA_BLCA-BRCA-ACC",
    "TCGA_CESC-DLBC-ESCA-GBM",
    "TCGA_CHOL",
    "TCGA_COAD-READ",
    "TCGA_HCC",
    "TCGA_HNSC-KICH-KIRC-KIRP",
    "TCGA_LAML-LGG-LUAD",
    "TCGA_LUSC-MESO-OV-PAAD",
    "TCGA_PCPG-PRAD-SARC-SKCM-STAD",
    "TCGA_TGCT-THCA-THYM-UCEC-UCS-UVM",
]
all_target = ["TARGET_ALL", "TARGET_AML", "TARGET_NBL", "TARGET_OS-WT-CCSK-RT"]
all_cptac = ["CPTAC"]
all_cgci = ["CGCI"]


def get_coad(f):
    adata = collect_gdc_counts(
        here(public_data, "TCGA_COAD-READ"), count_col="tpm_unstranded"
    )
    adata.write_h5ad(f)


# read_existing(here(public_data, "TCGA_COAD-READ_tpm.h5ad"), get_coad, lambda x: x)

allowed_types = {"Primary Tumor", "Recurrent Tumor", "Metastatic"}
for project, cases in zip(
    [all_tcga, all_target, all_cptac, all_cgci],
    [tcga_cases, target_cases, cptac_cases, cgci_cases],
):
    for t in project:

        def fn(p):
            adata = collect_gdc_counts(here(public_data, t), case_table=cases)
            adata = adata[adata.obs["Sample_Type"].isin(allowed_types), :]
            adata.write_h5ad(p)
            return adata

        read_existing(here(outdir, f"{t}.h5ad"), fn, lambda x: x)

# * In-house organoid samples
print("Collecting in-house samples")
pipeline_data = here("remote", "output")
in_house = {
    "LIHC": here(pipeline_data, "HCC", "RNASEQ", "4-cohort-All_counts.tsv"),
    "COAD-READ": here(pipeline_data, "CRC", "RNASEQ", "4-cohort-All_counts.tsv"),
    "CHOL": here(pipeline_data, "CCA", "RNASEQ", "4-cohort-All_counts.tsv"),
}
id_mapping = pd.read_csv(id_mapping_file, sep="\t")
id2name = {i: n for i, n in zip(id_mapping["gene_id"], id_mapping["gene_name"])}
sitemap = {
    "LIHC": "liver and intrahepatic bile ducts",
    "CHOL": "liver and intrahepatic bile ducts",
    "COAD-READ": "colon",
}


def get_in_house(p):
    dfs = {k: pd.read_csv(v, sep="\t") for k, v in in_house.items()}
    adatas = []
    for type, df in dfs.items():
        gene_ids = df.iloc[:, 0]
        counts = df.iloc[:, 1:]
        samples = counts.columns
        names = [id2name.get(i, np.nan) for i in gene_ids]
        obs = pd.DataFrame(
            {
                "Case_ID": samples,
                "tumor_type": type,
                "Sample_Type": "Organoid",
                "Project_ID": f"CHULA-{type}",
                "primary_site": sitemap.get(type),
            }
        )
        var = pd.DataFrame({"gene_id": gene_ids, "gene_name": names})
        cur = ad.AnnData(X=np.transpose(counts.values), obs=obs, var=var)
        adatas.append(cur)
    final: ad.AnnData = ad.concat(adatas, join="outer", merge="same")
    final.obs = final.obs.reset_index(drop=True)
    final.write_h5ad(p)


in_house_file = here(outdir, "in_house_organoids.h5ad")
read_existing(in_house_file, get_in_house, lambda x: x)

# * Get all other GEO
GEO_PATH: Path = here(public_data, "GEO")


def geo_reader_hf(
    dirname: str, tumor_type: str, primary_site: str, reader: Callable[..., ad.AnnData]
) -> ad.AnnData:
    accession = re.sub("-.*", "", dirname)
    adata: ad.AnnData = reader(here(GEO_PATH, dirname))
    # 'reader' must specify Case_ID
    obs = pd.DataFrame(
        {
            "tumor_type": tumor_type,
            "Sample_Type": "Organoid",
            "Project_ID": accession,
            "primary_site": primary_site,
        }
    )
    adata.obs = pd.concat([adata.obs, obs], axis="columns")
    return adata


def count_reader(
    path: Path,
    count_col: int = None,
    id_col: int = None,
    names_from_paths: bool = False,
) -> ad.AnnData:
    "Read generic counts file"

    def read_one(filename: Path):
        if "csv" in str(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.read_csv(filename, sep="\t")
        if count_col and id_col:
            df = df.iloc[:, [id_col, count_col]]
        gene_ids = df.iloc[:, 0]
        df = df.iloc[:, 1:]
        sample_names = df.columns
        counts = np.transpose(df)
        if names_from_paths:
            name = filename.stem
        else:
            name = sample_names
        adata = ad.AnnData(
            X=counts,
            var=pd.DataFrame(index=gene_ids),
            obs=pd.DataFrame({"Case_ID": name}, index=counts.index),
        )
        return adata

    if path.is_dir():
        adatas = [read_one(p) for p in path.iterdir()]
        return ad.concat(adatas, axis="obs", join="outer", merge="same")
    return read_one(path)


def count_reader_scrnaseq(
    path: Path,
    id_col,
    prefixes=(),
    counts_files=(),
    anno_files=(),
    one_per_file=(),
    is_10x_h5: bool = False,
):
    def scrna_read_one(index) -> ad.AnnData:
        if is_10x_h5 and prefixes:
            adata: ad.AnnData = sc.read_10x_h5(path.joinpath(prefixes[index]))
        elif prefixes:
            adata: ad.AnnData = sc.read_10x_mtx(path=path, prefix=prefixes[index])
        elif counts_files and anno_files:
            cf = counts_files[index]
            af = anno_files[index]
            counts = pd.read_csv(cf, sep="," if "csv" in cf else "\t")
            anno = pd.read_csv(af, sep="," if "csv" in af else "\t")
            adata = ad.AnnData(
                X=counts, obs=anno, var=pd.DataFrame(index=counts.iloc[:, 0])
            )
        else:
            raise ValueError(
                "Either a 10x prefix must be provided, or the names of counts and annotations files"
            )
        if not one_per_file:
            by_sample = adata.obs.groupby(id_col).agg("first")
            pb = into_pseudobulks(adata, how="sum", id_col=id_col)
            return ad.AnnData(
                X=pb.X, obs=by_sample, var=pd.DataFrame(index=pb.var.index)
            )
        else:
            counts = adata.X.sum(axis=0)
            return ad.AnnData(
                X=counts,
                obs=pd.DataFrame({"Case_ID": one_per_file[index]}),
                var=pd.DataFrame(index=adata.var.index),
            )

    if prefixes:
        length = len(prefixes)
    elif (counts_files and anno_files) and (len(counts_files) == len(anno_files)):
        length = len(counts_files)
    else:
        raise ValueError("prefixes or paths to counts and annotations not provided")
    adatas = [scrna_read_one(i) for i in range(length)]
    return ad.concat(adatas, axis="obs", join="outer", merge="same")


# ** Fns for specific GEO
def get_GSE202263(x):
    id_col = "patient.treatment_phase"
    adata = count_reader_scrnaseq(
        x,
        id_col=id_col,
        counts_files=("GSE202263_UMIcounts_all_samples_HGSOC_organoids_v2.tsv.gz"),
        anno_files=("GSE202263_cell_type_sample_annotation_v2.tsv.gz"),
    )
    wanted = adata.obs[id_col].str.contains("organoid\\.primary")
    adata = adata[wanted, :]
    return adata


def get_GSE203608(x):
    id_col = "Patient"
    adata = count_reader_scrnaseq(
        x,
        id_col=id_col,
        counts_files=("GSE203608_gene_by_cell_count_matrix.txt.gz"),
        anno_files=("GSE203608_cell_annotation.csv.gz"),
    )
    wanted = adata.obs["SampleType"] == "Tumoroid"
    adata = adata[wanted, :]
    return adata


def get_GSE218385(x):
    paths = "GSM6744031_P052_RNA_counts.csv.gz GSM6744033_P116_RNA_counts.csv.gz GSM6744035_P166_RNA_counts.csv.gz GSM6744037_P168_RNA_counts.csv.gz GSM6744039_P156_RNA_counts.csv.gz GSM6744041_P041_RNA_counts.csv.gz GSM6744043_P138_RNA_counts.csv.gz".split(
        " "
    )
    dfs = [pd.read_csv(x.joinpath(p), sep=" ").sum(axis=1) for p in paths]
    case_ids = [p.stem for p in paths]
    all = pd.concat(dfs, axis="columns")
    return ad.AnnData(
        X=all,
        obs=pd.DataFrame({"Case_ID": case_ids}),
        var=pd.DataFrame(index=all.index),
    )


# ** Get all GEO
# Map of dirname -> [tumor type, primary_site, fn to read]
geo_map = {
    "GSE185335-pdac_organoid": ["PAAD", "pancreas", count_reader],
    "GSE198697-crc_organoid": ["COAD-READ", "colon", lambda x: count_reader(x, 0, 4)],
    "GSE201740-rb_organoid": ["RB", "eye and adnexa", count_reader],
    "GSE202263-ovary_organoid": ["OV", "ovary", get_GSE202263],
    "GSE203608-crc_organoid": ["COAD-READ", "colon", get_GSE203608],
    "GSE212014-pdac_organoid": ["PAAD", "pancreas", count_reader],
    "GSE214295-pc_organoid": [
        "PAAD",
        "pancreas",
        lambda x: count_reader_scrnaseq(
            x,
            prefixes=["GSM6603327_PDO001_", "GSM6603327_PDO002_", "GSM6603327_PDO003_"],
            one_per_file=["PDO001", "PDO002", "PDO003"],
        ),
    ],
    "GSE218114-rhabdoid_organoid": ["RHBD", "kidney", count_reader],
    "GSE218385-rhabdoid_organoid": ["RHBD", "kidney", get_GSE218385],
    "GSE223554-salivary_organoid": ["SLV", "floor of mouth", count_reader],
    "GSE230383-eac_organoid": ["ESCA", "esophagus", count_reader],
    "GSE233468-lung_organoid": ["LUSC", "bronchus and lung", count_reader],
    "GSE233532-brain_organoid": [
        "HGG",
        "brain",
        lambda x: count_reader_scrnaseq(
            x,
            prefixes=[
                "GSM7429909_19040X1_filtered_feature_bc_matrix.h5"
                "GSM7429910_19040X2_filtered_feature_bc_matrix.h5",
                "GSM7429911_19476X1_filtered_feature_bc_matrix.h5",
                "GSM7429912_19476X2_filtered_feature_bc_matrix.h5",
                "GSM7429913_19476X3_filtered_feature_bc_matrix.h5",
                "GSM7429914_19476X4_filtered_feature_bc_matrix.h5",
                "GSM7429915_19476X5_filtered_feature_bc_matrix.h5",
                "GSM7429916_19476X6_filtered_feature_bc_matrix.h5",
                "GSM7429917_19476X7_filtered_feature_bc_matrix.h5",
                "GSM7429918_19476X8_filtered_feature_bc_matrix.h5",
            ],
            is_10x_h5=True,
            one_per_file=[
                "GSM7429909",
                "GSM7429910",
                "GSM7429911",
                "GSM7429912",
                "GSM7429913",
                "GSM7429914",
                "GSM7429915",
                "GSM7429916",
                "GSM7429917",
                "GSM7429918",
            ],
        ),
    ],
    "GSE235548-pdac_organoid": [
        "PAAD",
        "pancreas",
        lambda x: count_reader(x, names_from_paths=True),
    ],  # HUGO names
    "GSE243649-pdac_organoid": [
        "PAAD",
        "pancreas",
        lambda x: count_reader_scrnaseq(
            x,
            prefixes=["GSM7792313_pFPCO_", "GSM7792314_qFPCO_"],
            one_per_file=["GSM7792313", "GSM7792314"],
        ),
    ],
    "GSE247380-brain_organoid": [
        "LGG",
        "brain",
        lambda x: count_reader_scrnaseq(
            x,
            prefixes=["GSM7888157_PDO_Day28_", "GSM7888158_PDO_Day61_"],
            one_per_file=["GSM7888157_PDO_Day28", "GSM7888158_PDO_Day61"],
        ),
    ],
    "GSE249670-pdac_organoid": [
        "PAAD",
        "pancreas",
        lambda x: count_reader_scrnaseq(
            x, prefixes=["GSM7957064_"], one_per_file=["GSM7957064"]
        ),
    ],
    "GSE253558-crc_organoid": [
        "COAD-READ",
        "colon",
        lambda x: count_reader_scrnaseq(
            x, prefixes=[""], one_per_file=["GSM8023254_PC52"]
        ),
    ],
    "GSE261012-crc_organoid": [
        "COAD-READ",
        "colon",
        lambda x: count_reader_scrnaseq(
            x, prefixes=[""], one_per_file=["GSE261012_P18"]
        ),
    ],
    "GSE262110-breast_organoid": ["BRCA", "breast", count_reader],  # Entrez IDs
    "GSE270210-crc_organoid": ["COAD-READ", "colon", count_reader],  # Entrez Ids
    "GSE276387-lung_organoid": ["LUAD", "bronchus and lung", count_reader],
    "GSE277147-ead_organoid": [
        "ESCA",
        "esophagus",
        lambda x: count_reader(x, names_from_paths=True),
    ],  # HUGO ids
    "GSE278302-osteosarcoma_organoid": [
        "SARC",
        "bones, joints and articular cartilage of limbs",
        count_reader,
    ],  # HUGO ids
    "GSE280749-uc_organoid": ["BLCA", "bladder", count_reader],
    "GSESE247359-crc_organoid": ["COAD-READ", "colon", count_reader],
}
# <2025-02-24 Mon> Might want to recode to combine LGG and HGG (since you only
# have one case of the latter)
# TODO: in combined file, replace spaces and commas with _

geo_adatas = []
index_outdir = here("data", "indices_tmp_Monday_Feb-24-2025")
case_outdir = here("data", "cases_tmp_Monday_Feb-24-2025")
for dir, tup in geo_map.items():
    try:
        current: ad.AnnData = geo_reader_hf(dir, *tup)
        current.var.index.to_series().to_csv(here(index_outdir, f"{dir}.csv"))
        current.obs["Case_ID"].to_csv(here(case_outdir, f"{dir}.csv"))
    except Exception as e:
        logger.debug(f"------\n Reading {dir} failed")
        logger.debug(str(e))
        logger.debug("------------------------")

    # current.write
    # TODO: need to unify gene naming schemes...
    # geo_adatas.append(current)
