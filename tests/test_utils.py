#!/usr/bin/env ipython
import itertools
import math
from abc import abstractmethod
from os import replace
from pathlib import Path
from typing import override

import anndata as ad
import h5py
import interpret.blackbox as ib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import rpy2.robjects as ro
import scanpy as sc
import seaborn as sns
import skbio.stats.composition as comp
import sklearn.metrics as sm
import sklearn.neighbors as sn
import sklearn.preprocessing as sp
import too_predict._rust_helpers as rh
import too_predict.evaluation as te
import too_predict.model as tm
import too_predict.utils as ut
from distributed.deploy import spec
from imblearn.under_sampling import TomekLinks
from pyhere import here
from rpy2.rinterface import SexpVector
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr
from scipy import sparse, spatial, stats
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from too_predict.filter import Filter, count_tomek_links
from too_predict.imputer import Imputer
from too_predict.model import (
    AlrBase,
    PredBase,
    RandomForestPred,
    SimEstimator,
    XGBEstimator,
)
from too_predict.transformer import Transformer

# #  --- CODE BLOCK ---
#
base = importr("base")
ensembldb = importr("ensembldb")
obs = pd.read_csv(here("data", "training_data_obs.csv"))

hg38 = here("data", "Homo_sapiens.GRCh38.113.sqlite")
adata = ut.training_data_internal_test()
# adata = adata[:]


labels = adata.obs.index
target = adata.obs["tumor_type"]
# #  --- CODE BLOCK ---
refs, features = ut.ref_feature_lists_internal(False)
chosen = features["edgeR_median_lfc_feature_list_3000"]

adata = adata[:, :500]

counts = adata.X.toarray()

# encoded, pairs = rh.encode_pairs(counts)

# current = "BRCA"
# pair_lookup = {tuple(p): i for i, p in enumerate(pairs)}


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


filter = Filter(feature_col="GENEID", features=chosen)
t = Transformer("clr", impute_fn=Imputer("plus_one"), inplace=False)
result = t.fit_transform(adata)

model = PredBase(model=XGBEstimator(importance_type="gain"))


model.fit(result)
xgb = model.get_model()
