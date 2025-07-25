#!/usr/bin/env ipython
import itertools
import math
from abc import abstractmethod
from datetime import date
from functools import reduce
from os import replace
from pathlib import Path
from typing import Literal, override

import anndata as ad
import h5py
import marsilea as ma
import marsilea.plotter as mp
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
import sklearn.model_selection as ms
import sklearn.neighbors as sn
import sklearn.preprocessing as sp
import too_predict._rust_helpers as rh
import too_predict.deep.evaluation as d_ev
import too_predict.deep.torch_utils as d_ut
import too_predict.evaluation as te
import too_predict.explanation as ex
import too_predict.filter as fil
import too_predict.go_utils as gu
import too_predict.model as tm
import too_predict.multitask as multi
import too_predict.plotting as tp
import too_predict.r_utils as ru
import too_predict.utils as ut
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numba import jit
from pyhere import here
from rpy2.robjects.packages import importr
from scipy import sparse
from scipy.stats import mode
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from too_predict._train_utils import (
    ADDITIONAL_SPLITS,
    MODELS,
    organoid_test_task,
    read_model_spec,
)
from too_predict.corrector import Corrector
from too_predict.deep.logistic import MultiLevel
from too_predict.imputer import Imputer
from too_predict.transformer import Transformer
from torch.utils.data import DataLoader

# %%

#
base = importr("base")
ensembldb = importr("ensembldb")
obs = pd.read_csv(here("data", "training_data_obs.csv"))

hg38 = here("data", "Homo_sapiens.GRCh38.113.sqlite")
adata = ut.training_data_internal_test(minimal=True)


spc = MODELS["clr_ranks_mean_xgb_edger_per_type_ovp"]

adata.obs["is_organoid"] = adata.obs["Sample_Type"] != "primary"
adata.obs.loc[:, "not_primary"] = adata.obs["Sample_Type"] != "primary"

adata = adata[
    adata.obs["Sample_Type"].isin(["primary", "metastatic", "primary_blood"]), :
]

F, model, T, B, R, C = read_model_spec(spc, pipeline=False)
filtered = F.fit_transform(adata)
filtered.X = filtered.X.toarray()

train, test = ut.train_test_split_ad(adata[:, :50])


def get_labs(adata) -> np.ndarray:
    return adata.obs.loc[:, ["tumor_type", "Sample_Type"]].values


transformer = Transformer("clr", Imputer("plus_one"), post_process=)
