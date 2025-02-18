#!/usr/bin/env ipython

import anndata as ad
import scanpy as sc
from pyhere import here
from too_predict.model import PredBase
from too_predict.normalizer import IMPLEMENTED_NORMALIZATION
from too_predict.utils import (
    add_gene_metadata,
    dgelist2anndata,
)

datadir = here("data", "tests")
hcc = here(datadir, "tcga_hcc.rds")
chol = here(datadir, "tcga_chol.rds")
coad = here(datadir, "tcga_coad-read.rds")

test_sets = {"LIHC": hcc, "CHOL": chol, "COAD": coad}
hg38 = here("data", "Homo_sapiens.GRCh38.113.sqlite")


def loader(path, type):
    adata = dgelist2anndata(str(path))
    adata = adata[:100]
    adata.obs["tumor_type"] = type
    return adata


adata: ad.AnnData = ad.concat([loader(t, p) for p, t in test_sets.items()])
adata.var.index = adata.var.index.to_series().str.replace("\\..*", "", regex=True)

add_gene_metadata(adata)

# TODO: make a comparison of PCA with different normalization methods on the full data
# and also imputation
for n in IMPLEMENTED_NORMALIZATION:
    M = PredBase(adata, n, "foo")
    M.normalize()
    sc.pp.pca(M.ad)
    fig = sc.pl.pca(M.ad, color="tumor_type", return_fig=True)
    fig.savefig(here())
