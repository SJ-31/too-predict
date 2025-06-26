#!/usr/bin/env ipython

import anndata as ad
import numpy as np
import scipy.spatial.distance as spd
import sklearn.datasets as datasets
import sklearn.linear_model as sl
import sklearn.metrics as met
import sklearn.model_selection as ms
import sklearn.preprocessing as sp
import too_predict.deep.evaluation as d_ev
import too_predict.deep.nns as d_nn
import too_predict.deep.torch_utils as d_ut
import too_predict.filter as fil
import too_predict.multitask as multi
import too_predict.transformer as tt
import too_predict.utils as ut
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
from pyhere import here
from too_predict.deep.evaluation import multitask_acc, train_test_split_torch
from too_predict.deep.logistic import DummyLR, MtcLr, MultiLevel
from too_predict.imputer import Imputer
from torch.utils.data import DataLoader

cache = here("data", ".sklearn")
# %%


torch.set_default_dtype(torch.float64)


adata = ut.training_data_internal_test(minimal=True)  # 1000 features
transformer = tt.Transformer("clr", impute_fn=Imputer("plus_one"), inplace=False)
adata = transformer.fit_transform(adata)


adset = d_ut.AnnDataset(adata, to_encode=("Sample_Type", "tumor_type"))

# train_l, test_l, valid_l = train_test_split_torch(adset, valid=0.1, batch_size=32)
train_l, test_l = train_test_split_torch(adset, batch_size=32)

n_features, n_classes = d_ut.data_spec(train_l)

base = d_ev.Baseline(n_features, n_classes, max_depth=1)
base.fit(*train_l.dataset[:])
res = base.predict(test_l.dataset[:][0])
base_acc = d_ev.multitask_acc(test_l.dataset[:][1], res, ["Sample_Type", "tumor_type"])
print(f"Base acc: {base_acc}")

# %%
#


def test_disyak():
    model = d_nn.Disyak(
        n_features,
        n_classes_per_task=n_classes,
        reduce_features=False,
    )
    optimizer = optim.Adam(model.named_parameters(), lr=0.001)
    sch = schedule.ReduceLROnPlateau(optimizer=optimizer, patience=40)
    trainer = d_ut.Trainer(
        model,
        optimizer=optimizer,
        n_epochs=10,
        record_test_score=False,
        at_batch_level=True,
        scheduler=sch,
    )
    result = trainer(train_l)
    acc = d_ev.multitask_acc(
        test_l.dataset[:][1], model.predict(test_l), ["Sample_Type", "tumor_type"]
    )
    print(f"Disyak acc: {acc}")
    return model, result


disyak, res = test_disyak()
