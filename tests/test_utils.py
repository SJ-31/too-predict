#!/usr/bin/env ipython
import itertools
import math
from abc import abstractmethod
from os import replace
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
import too_predict._rust_helpers as rh
from imblearn.under_sampling import TomekLinks
from pyhere import here
from rpy2.rinterface import SexpVector
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr
from scipy import sparse, spatial, stats
from scripts.cross_validate import FEATURE_LISTS
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from too_predict.evaluation import (
    classification_report2df,
    get_all_metrics,
    precision_recall_multiclass,
    roc_multiclass,
)
from too_predict.filter import Filter, count_tomek_links
from too_predict.imputer import Imputer
from too_predict.model import (
    AlrBase,
    PredBase,
    RandomForestPred,
    SimEstimator,
)
from too_predict.transformer import Transformer
from too_predict.utils import (
    adata_x_to_r,
    add_gene_metadata,
    df_from_r,
    df_to_r,
    dgelist2anndata,
    find_confounded,
    get_data,
    library,
    np_to_r,
    phi_proportionality,
    r_cleanup,
    ref_feature_lists_internal,
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
adata = adata[:, :100]


labels = adata.obs.index
target = adata.obs["tumor_type"]
# #  --- CODE BLOCK ---
refs, features = ref_feature_lists_internal()
chosen = features["mutual_info_feature_list_3000"]

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


counts = StandardScaler().fit_transform(counts)
model = LinearSVC()
model.fit(counts, target)
pred = model.predict(counts)
scores = model.decision_function(counts)
score_df = pd.DataFrame(scores, columns=model.classes_)

s = score_df.idxmax(axis=1)
assert (s == pred).all()

# TODO: try out a SVM
# will need to update metrics and the alrbase to use `decision_function` instead
# of `predict_proba`
# Explore class_weight parameters,
