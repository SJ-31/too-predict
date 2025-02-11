#!/usr/bin/env ipython

import importlib.resources as res
from pathlib import Path

import anndata as ad
import pandas as pd
import rpy2.robjects as ro
import too_predict
from pyhere import here
from rpy2.rinterface import SexpVector
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr
from scipy import sparse
from too_predict.utils import (
    add_gene_metadata,
    df_from_r,
    dgelist2anndata,
    get_data,
    r_cleanup,
)

base = importr("base")
ensembldb = importr("ensembldb")
datadir = here("data", "tests")
hcc = here(datadir, "tcga_hcc.rds")

adata = dgelist2anndata(str(hcc))

hg38 = here("data", "Homo_sapiens.GRCh38.113.sqlite")

adata: ad.AnnData = dgelist2anndata(str(hcc))
adata.var.index = adata.var.index.to_series().str.replace("\\..*", "", regex=True)
add_gene_metadata(adata)
