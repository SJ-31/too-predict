#!/usr/bin/env ipython
import itertools
import math
from abc import abstractmethod
from os import replace
from pathlib import Path
from typing import override

import alibi.api.interfaces as aai
import anndata as ad
import h5py
import interpret.blackbox as ib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import rpy2.robjects as ro
import scanpy as sc
import scipy.cluster as cluster
import scipy.cluster.hierarchy as sch
import scipy.optimize as opt
import scipy.spatial.distance as spd
import seaborn as sns
import seaborn.objects as so
import shap
import skbio.stats.composition as comp
import sklearn.feature_selection as fs
import sklearn.metrics as sm
import sklearn.neighbors as sn
import sklearn.preprocessing as sp
import too_predict._rust_helpers as rh
import too_predict.evaluation as te
import too_predict.explanation as te
import too_predict.filter as fil
import too_predict.go_utils as gu
import too_predict.model as tm
import too_predict.utils as ut
from joblib import Parallel, delayed
from pyhere import here
from rpy2.robjects.packages import importr
from scipy import sparse
from scipy.stats import mode
from too_predict._train_utils import MODELS, read_model_spec

# #  --- CODE BLOCK ---
#
base = importr("base")
ensembldb = importr("ensembldb")
obs = pd.read_csv(here("data", "training_data_obs.csv"))

hg38 = here("data", "Homo_sapiens.GRCh38.113.sqlite")
adata = ut.training_data_internal_test()


def make_contingency(pair, pair_lookup, mat, current_label, label_vec):
    index = pair_lookup[pair]
    vals = mat[:, index]
    contingency = [
        [
            len(vals[(label_vec == current_label) & vals > 0]),
            len(vals[(label_vec == current_label) & vals == 0]),
        ],
        [
            len(vals[(label_vec != current_label) & vals > 0]),
            len(vals[(label_vec != current_label) & vals == 0]),
        ],
    ]
    return contingency


# #  --- CODE BLOCK ---

spc = MODELS["clr_random_forest_edger"]

F, M, T, B, E = read_model_spec(spc)
filtered = F.fit_transform(adata)
transformed = T.fit_transform(filtered)

transformed.obs["foo"] = "foo"
train, test = ut.train_test_split_ad(transformed)

ccs = fil.CompareSplits(train, test)
fig = ccs.plot_pca(
    plot_together=True,
    subset=("BRCA", "COAD_READ", "DLBC"),
    style=["Sample_Type", "usage"],
)
fig.show()

# ccs.get_prototypes()
# fig = ccs.plot_prototypes(plot_together=True)
# fig.savefig(Path.home().joinpath("test.png"))
#


# #  --- CODE BLOCK ---

obs = pl.read_csv("/home/shannc/Bio_SDD/too-predict/data/training_data_obs.csv")
wanted = ["tumor_type", "primary_site", "Sample_Type"]
obs.select(
    pl.col("Project_ID"),
    pl.col("tumor_type"),
    pl.col("primary_site"),
    pl.col("Sample_Type"),
).group_by("Project_ID").agg(n=pl.count(), *[pl.col(w).first() for w in wanted]).sort(
    "Sample_Type"
).filter(~pl.col("Project_ID").str.contains("CHULA")).write_csv(
    Path.home().joinpath("Downloads").joinpath("external_sources_2025-4-10.csv")
)
