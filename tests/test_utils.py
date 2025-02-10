#!/usr/bin/env ipython

import importlib

import anndata as ad
import rpy2.robjects as ro
from pyhere import here
from rpy2.robjects.packages import importr
from scipy import sparse
from too_predict.utils import df_from_r, dgelist2anndata, r_cleanup

base = importr("base")
datadir = here("data", "tests")
hcc = here(datadir, "tcga_hcc.rds")


adata = dgelist2anndata(str(hcc))
