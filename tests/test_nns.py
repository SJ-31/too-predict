#!/usr/bin/env ipython

import anndata as ad
import lightning as L
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
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pyhere import here
from too_predict.deep.evaluation import multitask_acc, train_test_split_torch
from too_predict.deep.logistic import DummyLR, MtcLr, MultiLevel
from too_predict.deep.trainer import AverageModelL, Trainer
from too_predict.imputer import Imputer
from torch.utils.data import DataLoader

cache = here("data", ".sklearn")
# %%


torch.set_default_dtype(torch.float32)


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
base_acc = d_ev.multitask_acc(
    test_l.dataset[:][1],
    res,
    task_names=["Sample_Type", "tumor_type"],
    n_classes=n_classes,
)
print(f"Base acc: {base_acc}")

# %%
#
EPOCHS = 10


def test_disyak():
    model = d_nn.Disyak(n_features, n_classes_per_task=n_classes, record_metrics=False)
    optimizer = optim.Adam(model.named_parameters(), lr=0.001)
    sch = schedule.ReduceLROnPlateau(optimizer=optimizer, patience=40)
    trainer = Trainer(
        model,
        optimizer=optimizer,
        n_epochs=EPOCHS,
        record_test_score=False,
        at_batch_level=True,
        scheduler=sch,
    )
    result = trainer(train_l, n_classes=n_classes)
    acc = d_ev.multitask_acc(
        test_l.dataset[:][1],
        model.predict_step(test_l),
        task_names=["Sample_Type", "tumor_type"],
        n_classes=n_classes,
    )
    print(f"Disyak acc: {acc}")
    return model, result


# disyak, time = d_ut.timed(lambda: test_disyak())


def test_lightning():
    model = d_nn.Disyak(n_features, n_classes_per_task=n_classes, record_metrics=True)
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        callbacks=[
            EarlyStopping(
                monitor="train_loss", patience=10, mode="min"
            ),  # If "min", lower is better
            DeviceStatsMonitor(),
            AverageModelL(model=model, mode="EMA", best_target="train_acc", n_best=5),
        ],
    )
    model.set_cache("train_acc")
    trainer.fit(model=model, train_dataloaders=train_l, val_dataloaders=None)
    trainer.test(model=model, dataloaders=test_l)
    return model, trainer


(model, trainer), time = d_ut.timed(lambda: test_lightning())
print(model._cache)
# trainer.fit(model=_[0])
