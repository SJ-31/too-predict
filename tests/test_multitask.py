#!/usr/bin/env ipython

import functools
import itertools
from ast import MatchClass
from pathlib import Path
from typing import override

import anndata as ad
import numpy as np
import scipy.spatial.distance as spd
import sklearn.datasets as datasets
import sklearn.linear_model as sl
import sklearn.metrics as met
import sklearn.model_selection as ms
import sklearn.preprocessing as sp
import skorch
import too_predict.deep.torch_utils as d_ut
import too_predict.multitask as multi
import too_predict.utils as ut
import torch
import torch.nn as nn
import torch.optim as optim
from pyhere import here
from too_predict.deep.logistic import DummyLR, MtcLr, MtcLrSkorch
from torch.utils.data import DataLoader

# %%

# adata = ut.training_data_internal_test()
# adata = adata[~adata.obs["Sample_Type"].isna(), :50]

# %%
cache = here("data", ".sklearn")


torch.set_default_dtype(torch.float64)
data = list(datasets.load_wine(return_X_y=True))
data[0] = sp.StandardScaler().fit_transform(data[0])
X_train, X_test, y_train, y_test = ms.train_test_split(data[0], data[1])

# dataset = d_ut.AnnDataset(adata, to_encode=("tumor_type",))
dset = d_ut.make_dataset(X_train, y_train)


# scores = d_ut.train_model(model, loader, n_epochs=500)
# acc = model.predict(y_)

n_tasks = 1


def test_skorch():
    model = MtcLrSkorch.new(
        optimizer=optim.Adam,
        max_epochs=200,
        lr=0.0002,
        batch_size=64,
        module__n_features=X_train.shape[1],
        module__initial_fit=None,
        module__task_spec=[len(np.unique(data[1]))],
    )
    model = model.fit(X_train, y_train)
    new_acc = met.accuracy_score(y_test, model.predict(X_test))
    print(new_acc)


def test_pt():
    loader = DataLoader(dset, batch_size=len(dset))
    model = MtcLr(
        n_features=X_train.shape[1],
        task_spec=[len(np.unique(data[1]))],
    )
    opt = optim.SGD(model.named_parameters(), lr=0.00002)
    metrics = d_ut.train_model(
        model,
        loader,
        opt,
        criterion=MtcLr.criterion,
        needs_model=True,
        n_epochs=1000,
        needs_closure=True,
    )
    print(metrics)
    pred = model(torch.tensor(X_test))[0].argmax(dim=1).numpy()
    print(f"skorch {met.accuracy_score(y_test, pred)}")
    return model


def test_dummy():
    loader = DataLoader(dset, batch_size=len(dset))
    model = DummyLR(
        n_features=X_train.shape[1],
        n_classes=len(np.unique(data[1])),
    )
    opt = optim.Adam(model.named_parameters())
    metrics = d_ut.train_model(
        model,
        loader,
        opt,
        criterion=DummyLR.criterion,
        needs_model=True,
        n_epochs=2000,
        needs_closure=True,
    )
    pred = model.predict(X_test)
    print(f"Dummy lr acc: {met.accuracy_score(y_test, pred)}")
    return model


# test_skorch() # BUG: the parameters won't update with skorch
dummy = test_dummy()

model = test_pt()

old = sl.SGDClassifier(loss="log_loss")
old.fit(X_train, y_train)
old_acc = met.accuracy_score(y_test, old.predict(X_test))
print(f"sklearn: {old_acc}")

# [2025-06-11 Wed]
# Why is the performance so much worse than an sklearn model???
