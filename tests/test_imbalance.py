#!/usr/bin/env ipython

import math

import anndata as ad
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scanpy as sc
import too_predict.evaluation as te
import too_predict.imbalance as tb
import too_predict.model as tm
import too_predict.r_utils as ru
import too_predict.utils as ut
from too_predict.transformer import Transformer

# %%

#
adata = ut.training_data_internal_test(minimal=True)

transformer = Transformer("clr", "plus_one")

train, test = ut.train_test_split_ad(adata)

y = "tumor_type"

model = tm.PredBase(tm.XGBEstimator())
result = te.train_test_wrapper(model, (train, test), y)


def test_sim_nb():
    tcga = adata[adata.obs["Project_ID"].str.contains("TCGA"), :]
    counts = tcga.obs[y].value_counts()
    valid = counts[counts > 10]
    tcga = tcga[tcga.obs[y].isin(list(valid.index)), :]
    balancer = tb.Balancer(method="nb_edgeR", sample_mus=False, blocking=False)
    # Seems to do better without sampling, and instead taking the mean
    new = balancer.fit_transform(tcga, y=y)
    result = te.train_test_wrapper(model, (new, tcga), y)
    print(result)


def test_sim_nb_block():
    tcga = adata[adata.obs["Project_ID"].str.contains("TCGA"), :]
    counts = tcga.obs[y].value_counts()
    valid = counts[counts > 10]
    tcga = tcga[tcga.obs[y].isin(list(valid.index)), :]
    balancer = tb.Balancer(method="nb_edgeR", blocking=True, sample_mus=False)
    new = balancer.fit_transform(tcga, y=y)
    result = te.train_test_wrapper(model, (new, tcga), y)
    print(result)


test_sim_nb_block()

# TODO: you need to make this as good as the original

import scipy.stats as stats


def test_dirichlet():
    tcga = adata[adata.obs["Project_ID"].str.contains("TCGA"), :]
    train, test = ut.train_test_split_ad(tcga)
    tcga = tb.dirichlet_sim(tcga, as_transform=True)
    balancer = tb.Balancer(method="dirichlet")
    new = balancer.fit_transform(tcga, y=y)
    result = te.train_test_wrapper(model, (new, tcga), y)
    print(result)


def test_empirical():
    tcga = adata[adata.obs["Project_ID"].str.contains("TCGA"), :]
    tcga = transformer.fit_transform(tcga)
    train, test = ut.train_test_split_ad(tcga)
    balancer = tb.Balancer(method="empirical")
    new = balancer.fit_transform(tcga, y=y)
    result = te.train_test_wrapper(model, (new, tcga), y)
    print(result)


def test_splatter():
    tcga = adata[adata.obs["Project_ID"].str.contains("TCGA"), :]
    counts = tcga.obs[y].value_counts()
    valid = counts[counts > 10]
    tcga = tcga[tcga.obs[y].isin(list(valid.index)), :]
    train, test = ut.train_test_split_ad(tcga)

    print(tcga.obs[y].value_counts())
    balancer = tb.Balancer(method="splatter")
    new = balancer.fit_transform(tcga, y=y)
    result = te.train_test_wrapper(model, (new, tcga), y)
    base = te.train_test_wrapper(model, (tcga, tcga), y)
    print(base, result)


#
# Pool the simulated samples together, grouping by the same label. Then sample
# according to the specification
targets = {"BRCA": 50, "UCEC": 40, "LGG": 60}
