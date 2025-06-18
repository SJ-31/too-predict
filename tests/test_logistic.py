#!/usr/bin/env ipython

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
import torch.optim.lr_scheduler as schedule
from pyhere import here
from too_predict.deep.evaluation import multitask_acc, train_test_split_torch
from too_predict.deep.logistic import DummyLR, MtcLr, MultiLevel
from torch.utils.data import DataLoader

# %%

cache = here("data", ".sklearn")


torch.set_default_dtype(torch.float64)
data = list(datasets.load_wine(return_X_y=True))
data[0] = sp.StandardScaler().fit_transform(data[0])
X_train, X_test, y_train, y_test = ms.train_test_split(data[0], data[1])

dset = d_ut.make_dataset(X_train, y_train)

adata = ut.training_data_internal_test(minimal=True)
adset = d_ut.AnnDataset(adata, to_encode=("Sample_Type", "tumor_type"))
train_l, test_l = train_test_split_torch(adset, batch_size=32)


def test_pt():
    loader = DataLoader(dset, batch_size=len(dset))
    model = MtcLr(
        n_features=X_train.shape[1],
        n_classes_per_task=[len(np.unique(data[1]))],
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
    model = DummyLR(n_classes_per_task=len(np.unique(data[1])))
    opt = optim.Adam(
        model.named_parameters()
    )  # [2025-06-11 Wed] This is way better than SGD
    # TODO: make optimization more dynamic
    # metrics = d_ut.train_model(
    #     model,
    #     loader,
    #     opt,
    #     criterion=DummyLR.criterion,
    #     needs_model=True,
    #     n_epochs=2000,
    #     needs_closure=True,
    # )
    # pred = model.predict(X_test)
    # print(f"Dummy lr acc: {met.accuracy_score(y_test, pred)}")

    smodel = skorch.NeuralNetClassifier(
        module=DummyLR,
        module__n_classes_per_task=len(np.unique(data[1])),
        max_epochs=200,
    )
    smodel.fit(X_train, y_train)
    spred = smodel.predict(X_test)
    print(f"Dummy lr acc: {met.accuracy_score(y_test, spred)}")
    return smodel


def baseline():
    old = sl.SGDClassifier(loss="log_loss")
    old.fit(X_train, y_train)
    old_acc = met.accuracy_score(y_test, old.predict(X_test))
    print(f"sklearn: {old_acc}")


def test_multilevel():
    n_features, n_classes = d_ut.data_spec(train_l)

    mll = MultiLevel(in_features=n_features, n_classes_per_task=n_classes)
    optimizer = mll.get_optimizers()
    sch = schedule.ReduceLROnPlateau(optimizer)

    trainer = d_ut.Trainer(
        mll, n_epochs=1000, record_test_score=False, at_batch_level=False, scheduler=sch
    )
    metrics = trainer(train_l)

    pred = mll.predict(test_l)
    print(metrics)
    print(multitask_acc(test_l, pred))

    return mll


# %%

mll = test_multilevel()

par = next(mll.named_parameters())[1]

# smodel = test_dummy()
# dummy = test_dummy()

# model = test_pt()
