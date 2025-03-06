#!/usr/bin/env ipython
from abc import abstractmethod
from pathlib import Path
from typing import override

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scanpy as sc
import seaborn as sns
import skbio.stats.composition as comp
import sklearn.metrics as sm
import sklearn.neighbors as sn
import too_predict
from imblearn.under_sampling import TomekLinks
from pyhere import here
from rpy2.rinterface import SexpVector
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr
from scipy import sparse, spatial, stats
from sklearn.cluster import KMeans
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
from too_predict.filter import count_tomek_links
from too_predict.imputer import Imputer
from too_predict.model import AlrBase, PredBase
from too_predict.transformer import Transformer
from too_predict.utils import (
    adata_x_to_r,
    add_gene_metadata,
    df_from_r,
    df_to_r,
    dgelist2anndata,
    get_data,
    library,
    np_to_r,
    phi_proportionality,
    r_cleanup,
    rename_genes,
    source,
    str_mode,
    training_data_internal_test,
)

# #  --- CODE BLOCK ---
#
base = importr("base")
ensembldb = importr("ensembldb")
ALDex2 = importr("ALDEx2")
datadir = here("data", "tests")
hcc = here(datadir, "tcga_hcc.rds")
chol = here(datadir, "tcga_chol.rds")
coad = here(datadir, "tcga_coad-read.rds")

test_sets = {"LIHC": hcc, "CHOL": chol, "COAD": coad}
hg38 = here("data", "Homo_sapiens.GRCh38.113.sqlite")
adata = training_data_internal_test()
adata = adata[:, :50]
# Transformer("robust_clr", Imputer("replace_one").run, inplace=True).fit_transform(adata)


# #  --- CODE BLOCK ---
class Dist:
    """Class for conveniently indexing a condensed distance matrix"""

    def _validate_label(self, labels: np.ndarray) -> bool:
        if len(labels) != self.len:
            raise ValueError("The length of the given labels is incorrect!")
        if np.unique(labels).shape != labels.shape:
            raise ValueError("Given labels aren't unique!")
        return True

    def __init__(self, x: np.ndarray, names=None, classes=None) -> None:
        if not spatial.distance.is_valid_y(x):
            raise ValueError("x is not a valid condensed distance matrix")
        self.len = spatial.distance.num_obs_y(x)
        self.x = x
        if names is not None and self._validate_label((n := np.array(names))):
            self.names = n
        else:
            self.names = None
        if classes is not None and self._validate_label((c := np.array(classes))):
            self.classes = c
        else:
            self.classes = None

    def _find_label(self, labels: tuple[str, str]) -> tuple[int, int]:
        return tuple(map(lambda x: np.where(self.labels == x)[0][0], labels))

    def __getitem__(self, indices):
        if not isinstance(indices, tuple) or len(indices) > 2:
            raise ValueError("Must index as a pair!")
        if list(map(type, indices) == [str, str]):
            i, j = self._find_label(indices)
        else:
            i, j = indices
        if i > j:
            i, j = j, i
        index = self.len * i + j - ((i + 2) * (i + 1)) // 2
        return self.x[index]


labels = adata.obs.index
target = adata.obs["tumor_type"]

dirich = Transformer("dirichlet", None, make_sparse=False, inplace=False, n=10, prior=1)
data = dirich.fit_transform(adata)
# cf, mf, matrix = count_tomek_links(adata, "tumor_type")
#
