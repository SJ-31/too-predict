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
import too_predict.recoder as rt
import too_predict.utils as ut
from joblib import Parallel, delayed, parallel
from pyhere import here
from rpy2.robjects.packages import importr
from scipy import sparse
from scipy.stats import mode
from sklearn.linear_model import LogisticRegressionCV
from too_predict._train_utils import MODELS, read_model_spec
from too_predict.corrector import Corrector
from too_predict.plotting import plot_adata
from too_predict.transformer import Transformer

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

F, M, T, B, E, C = read_model_spec(spc)
adata.obs.loc[:, "not_primary"] = adata.obs["Sample_Type"] != "primary"
filtered = F.fit_transform(adata)
transformed = T.fit_transform(filtered)

transformed.obs["foo"] = "foo"
train, test = ut.train_test_split_ad(transformed)

counts = adata.X.toarray()
# #  --- CODE BLOCK ---
# meta = ut.cell_markers_internal(meta=True)
# reference_file = ut.cell_markers_internal(file_only=True)
# reference = ut.cell_markers_internal()


# gs = ut.gs_internal()

# over = ut.pairwise_overlaps(gs, do_parallel=True)

# ut.write_pickle(over, ut.get_data("reference/gene_sets_overlap.pckl", must_exist=False))

# over = ut.load_pickle(ut.get_data("reference/gene_sets_overlap.pckl", must_exist=False))
