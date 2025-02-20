#!/usr/bin/env python

import anndata as ad
import numpy as np
import pandas as pd
from pyhere import here
from too_predict.utils import collect_gdc_counts, read_existing

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


def get_coad(f):
    adata = collect_gdc_counts(
        here(public_data, "TCGA_COAD-READ"), count_col="tpm_unstranded"
    )
    adata.write_h5ad(f)


# read_existing(here(public_data, "TCGA_COAD-READ_tpm.h5ad"), get_coad, lambda x: x)

allowed_types = {"Primary Tumor", "Recurrent Tumor", "Metastatic"}
for project, cases in zip([all_tcga], [tcga_cases]):
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
