#!/usr/bin/env ipython
from abc import abstractmethod
from pathlib import Path
from typing import override

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scanpy as sc
import skbio.stats.composition as comp
import too_predict
from pyhere import here
from rpy2.rinterface import SexpVector
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr
from scipy import sparse, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from too_predict.evaluation import (
    classification_report2df,
    precision_recall_multiclass,
    roc_multiclass,
)
from too_predict.imputer import Imputer
from too_predict.model import AlrBase, PredBase
from too_predict.normalizer import Normalizer
from too_predict.utils import (
    add_gene_metadata,
    df_from_r,
    df_to_r,
    dgelist2anndata,
    get_data,
    library,
    phi_proportionality,
    r_cleanup,
    rename_genes,
    source,
    str_mode,
)

base = importr("base")
ensembldb = importr("ensembldb")
ALDex2 = importr("ALDEx2")
datadir = here("data", "tests")
hcc = here(datadir, "tcga_hcc.rds")
chol = here(datadir, "tcga_chol.rds")
coad = here(datadir, "tcga_coad-read.rds")

test_sets = {"LIHC": hcc, "CHOL": chol, "COAD": coad}
hg38 = here("data", "Homo_sapiens.GRCh38.113.sqlite")


# #  --- CODE BLOCK ---
def loader(path, type):
    adata = dgelist2anndata(str(path))
    adata = adata[:100]
    adata.obs["tumor_type"] = type
    return adata


adata: ad.AnnData = ad.concat([loader(t, p) for p, t in test_sets.items()])
adata.var.index = adata.var.index.to_series().str.replace("\\..*", "", regex=True)

# #  --- CODE BLOCK ---
