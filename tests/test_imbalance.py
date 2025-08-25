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
import torch
from pyhere import here
from too_predict.transformer import Transformer

# %%
torch.set_default_dtype(torch.float32)

#
adata = ut.training_data_internal_test(minimal=True)

transformer = Transformer("clr", "plus_one")

train, test = ut.train_test_split_ad(adata)

y = "tumor_type"

model = tm.PredBase(tm.XGBEstimator(max_depth=1))
result, _ = te.train_test_wrapper(model, (train, test), y)


def test_sim_nb():
    tcga = adata[adata.obs["Project_ID"].str.contains("TCGA"), :]
    counts = tcga.obs[y].value_counts()
    valid = counts[counts > 10]
    tcga = tcga[tcga.obs[y].isin(list(valid.index)), :]
    balancer = tb.Balancer(method="nb_edgeR", sample_mus=False, blocking=False)
    # Seems to do better without sampling, and instead taking the mean
    new = balancer.fit_transform(tcga, y=y)
    result, _ = te.train_test_wrapper(model, (new, tcga), y)
    print(result)


def test_sim_nb_block():
    tcga = adata[adata.obs["Project_ID"].str.contains("TCGA"), :]
    counts = tcga.obs[y].value_counts()
    valid = counts[counts > 10]
    tcga = tcga[tcga.obs[y].isin(list(valid.index)), :]
    balancer = tb.Balancer(
        method="nb_edgeR", blocking=True, sample_mus=False, sampling_strategy="none"
    )
    new = balancer.fit_transform(tcga, y=y)
    result, _ = te.train_test_wrapper(model, (new, tcga), y)
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
    result, _ = te.train_test_wrapper(model, (new, tcga), y)
    print(result)


def test_empirical():
    tcga = adata[adata.obs["Project_ID"].str.contains("TCGA"), :]
    tcga = transformer.fit_transform(tcga)
    train, test = ut.train_test_split_ad(tcga)
    balancer = tb.Balancer(method="empirical")
    new = balancer.fit_transform(tcga, y=y)
    result, _ = te.train_test_wrapper(model, (new, tcga), y)
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
    result, _ = te.train_test_wrapper(model, (new, tcga), y)
    base = te.train_test_wrapper(model, (tcga, tcga), y)
    print(base, result)


def test_cvae():
    # tcga = adata[adata.obs["Project_ID"].str.contains("TCGA"), :]
    # counts = tcga.obs[y].value_counts()
    # valid = counts[counts > 10]
    # tcga = tcga[tcga.obs[y].isin(list(valid.index)), :]

    balancer = tb.Balancer(
        method="cvae",
        sampling_strategy="none",
        trainer_kwargs={
            "enable_checkpointing": False,
            "max_epochs": 100,
            "enable_progress_bar": False,
        },
        loader_kwargs={"batch_size": 20},
        logger_kwargs={
            "exp_name": "imb_cvae",
            "platform": "tensorboard",
            "save_dir": here("tests", "lightning_logs"),
        },
        n_latent=5,
    )
    trans = transformer.fit_transform(adata)
    new = balancer.fit_transform(trans, y=y)
    result, _ = te.train_test_wrapper(model, (new, trans), y)
    base = te.train_test_wrapper(model, (trans, trans), y)
    print(base, result)
    return new


new = test_cvae()
#
# Pool the simulated samples together, grouping by the same label. Then sample
# according to the specification
