#!/usr/bin/env python

from pyhere import here
from too_predict.utils import collect_gdc_counts, read_existing

public_data = here("remote", "public_data")
outdir = here(public_data, "h5ad")
metadata_dir = here(public_data, "metadata")

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


read_existing(here(public_data, "TCGA_COAD-READ_tpm.h5ad"), get_coad, lambda x: x)

allowed_types = {"Primary Tumor", "Recurrent Tumor", "Metastatic"}
for project, cases in zip([all_tcga], [tcga_cases]):
    for t in project:

        def fn(p):
            adata = collect_gdc_counts(here(public_data, t), case_table=cases)
            adata = adata[adata.obs["Sample_Type"].isin(allowed_types), :]
            adata.write_h5ad(p)
            return adata

        read_existing(here(outdir, f"{t}.h5ad"), fn, lambda x: x)
