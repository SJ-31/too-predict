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
import scipy.cluster as cluster
import scipy.cluster.hierarchy as sch
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
import too_predict.model as tm
import too_predict.utils as ut
from imblearn.under_sampling import TomekLinks
from pyhere import here
from rpy2.rinterface import SexpVector
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr
from scipy import sparse, spatial, stats
from scipy.cluster.hierarchy import cut_tree
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration, KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from too_predict._train_utils import MODELS, read_model_spec
from too_predict.filter import CompareSplits, Filter, count_tomek_links
from too_predict.imputer import Imputer
from too_predict.plotting import plot_diagonal_matrix

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
filtered = T.fit_transform(adata)
transformed = F.fit_transform(filtered)

train, test = ut.train_test_split_ad(adata[:, :50])

cc = CompareSplits(train, test)
df = cc.edgeR_lfc()

# #  --- CODE BLOCK ---
