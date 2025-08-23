#!/usr/bin/env ipython

import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import too_predict.deep.evaluation as d_ev
import too_predict.deep.nns as d_nn
import too_predict.deep.torch_utils as d_ut
import too_predict.transformer as tt
import too_predict.utils as ut
import torch
from pyhere import here
from too_predict.deep.distillation import TeacherResponse, use_kd_criterion
from too_predict.deep.evaluation import (
    init_test,
    multitask_acc,
    random_softmax_loss,
    train_test_split_torch,
)
from too_predict.deep.metrics import multitask_all_metrics
from too_predict.imputer import Imputer
from torch.utils.data import (
    DataLoader,
)

import lightning as L

cache = here("data", ".sklearn")
# %%
labels = ("Sample_Type", "tumor_type")

torch.set_default_dtype(torch.float32)


adata1 = ut.training_data_internal_test(minimal=True)  # 1000 features
adata = ut.training_data_internal_test(minimal=True, backed=True)
transformer = tt.Transformer("clr", impute_fn=Imputer("plus_one"), inplace=False)
# adata = transformer.fit_transform(adata)
#
toy = d_ut.AnnDataset(adata, to_encode=labels)

train, test = train_test_split_torch(toy, as_dataloader=False)

ad1 = d_ut.AnnDataset(adata1, to_encode=("Sample_Type", "tumor_type"))

adset = d_ut.AnnDataset(adata, to_encode=("Sample_Type", "tumor_type"))

# train_l, test_l, valid_l = train_test_split_torch(adset, valid=0.1, batch_size=32)
train_l, test_l = train_test_split_torch(adset, batch_size=32)

train, test = ut.train_test_split_ad(adata)
train_adset = d_ut.AnnDataset(train, to_encode=labels)
valid_adset = d_ut.AnnDataset(test, to_encode=("Sample_Type", "tumor_type"))

n_features, n_classes = d_ut.data_spec(train_l)


base = d_nn.Baseline(n_features, n_classes, max_depth=1)
base.fit(*train_l.dataset[:])
base.fit(train_l.dataset)
res = base.predict_step(test_l.dataset[:][0])
base_acc = multitask_acc(
    test_l.dataset[:][1],
    res,
    task_names=["Sample_Type", "tumor_type"],
    n_classes=n_classes,
)
print(f"Base acc: {base_acc}")

cv_kwargs = {
    "n_classes": n_classes,
    "device": "cpu",
    "trainer_kwargs": {
        "max_epochs": 3,
        "enable_progress_bar": False,
        "enable_checkpointing": False,
        "log_every_n_steps": 1,
    },
    "validation": valid_adset,
    "in_features": n_features,
    "n_splits": 2,
    # "init_bias": False,
}

# %%
#
EPOCHS = 20


# %%

# %%


def test_lightning():
    train_adset_1 = d_ut.AnnDataset(train, to_encode=("tumor_type",))
    valid_adset_1 = d_ut.AnnDataset(test, to_encode=("tumor_type",))
    n_features_1, n_classes_1 = d_ut.data_spec(train_adset_1)
    model = d_nn.HardSharer(
        n_features_1,
        n_classes_per_task=n_classes_1,
        conf=d_ut.ModuleConfig(record_metrics=True),
    )
    trainer = L.Trainer(
        max_epochs=10,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        default_root_dir=here("tests", "lightning"),
        logger=None,
    )
    model.set_cache("train_acc")
    trainer.fit(
        model=model, train_dataloaders=DataLoader(train_adset_1), val_dataloaders=None
    )
    trainer.test(model=model, dataloaders=DataLoader(valid_adset_1))
    return (
        model,
        trainer,
    )


model, trainer = test_lightning()


# %%
def test_lightning_multi():
    model = d_nn.HardSharer(
        n_features,
        n_classes_per_task=n_classes,
        conf=d_ut.ModuleConfig(record_metrics=True),
    )
    trainer = L.Trainer(
        max_epochs=10,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        default_root_dir=here("tests", "lightning"),
        logger=None,
    )
    model.set_cache("train_acc")
    trainer.fit(model=model, train_dataloaders=train_l, val_dataloaders=None)
    trainer.test(model=model, dataloaders=test_l)
    y_test = test_l.dataset[:][1]
    x_test = test_l.dataset[:][0]
    score = model.predict_proba(x_test)
    metrics = multitask_all_metrics(y_true=y_test, scores=score, n_classes=n_classes)
    return model, trainer, metrics


model, trainer, metrics = test_lightning_multi()


def cm_correctness(cm: pd.DataFrame):
    """Report the count of correct predictions for individual
    labels in confusion matrix `cm`, as well as accuracy
    Columns in `cm` are taken to be predictions, rows are truth
    """
    if cm.shape[0] != cm.shape[1]:
        raise ValueError("Given confusion matrix is not square!")
    total_counts = cm.sum(axis=1)
    tp = np.diag(cm)
    result = pd.DataFrame(
        {
            "label": list(cm.index),
            "true_positives": tp,
            "accuracy": tp / total_counts,
            "total_count": total_counts,
            "label_prop": total_counts / total_counts.sum(),
        }
    )
    return pd.DataFrame(result).reset_index(drop=True)


# %%


def test_overfit():
    set = d_ut.AnnDataset(adata1[:2, :], to_encode=("Sample_Type", "tumor_type"))
    cv: pd.DataFrame = d_ev.cross_validate(
        model_cls=d_nn.HardSharer,
        adset=set,
        batch_size=-1,
        model_config=d_ut.ModuleConfig(cache="val_acc"),
        **cv_kwargs,
    )
    return cv


def test_whole_dataset():
    cv: pd.DataFrame = d_ev.cross_validate(
        model_cls=d_nn.Disyak, batch_size=-1, adset=train_adset, **cv_kwargs
    )
    print(cv)


def test_acc_whole_dataset():
    cv: pd.DataFrame = d_ev.cross_validate(
        model_cls=d_nn.Disyak,
        batch_size=-1,
        grad_accumulation=True,
        grad_accumulation_batch_size=32,
        adset=train_adset,
        **cv_kwargs,
    )
    print(cv)


def test_acc_larger():
    cv: pd.DataFrame = d_ev.cross_validate(
        model_cls=d_nn.Disyak,
        batch_size=700,
        grad_accumulation=False,
        grad_accumulation_batch_size=32,
        adset=train_adset,
        **cv_kwargs,
    )
    print(cv)


def test_acc():
    cv: pd.DataFrame = d_ev.cross_validate(
        model_cls=d_nn.Disyak,
        batch_size=150,
        grad_accumulation=True,
        grad_accumulation_batch_size=30,
        adset=train_adset,
        **cv_kwargs,
    )
    print(cv)


def test_bootstrap_dataset():
    cv: pd.DataFrame = d_ev.cross_validate(
        model_cls=d_nn.Disyak, adset=train_adset, batch_size=700, **cv_kwargs
    )
    print(cv)


test_acc()
# test_whole_dataset()
# %%


# %%
def test_splittable():
    response = TeacherResponse(
        train_adset, teacher=d_nn.Baseline(n_features, n_classes, max_depth=1)
    )
    cv = ms.KFold(n_splits=3, shuffle=True)
    print([i for i in cv.split(response)])


# %%


def test_distillation():
    response = TeacherResponse(
        train_adset, teacher=d_nn.Baseline(n_features, n_classes, max_depth=1)
    )
    trainer = L.Trainer(
        max_epochs=10, enable_checkpointing=False, enable_progress_bar=False
    )
    model = d_nn.Disyak(
        in_features=n_features,
        n_classes_per_task=n_classes,
        conf=d_ut.ModuleConfig(record_metrics=False, outlayer_type="regression"),
    )  # You can't calculate accuracy while using distillation
    use_kd_criterion(model)
    trainer.fit(model, train_dataloaders=DataLoader(response, batch_size=32))
    acc = multitask_acc(
        predictions=model.predict_step(valid_adset[:]),
        y_true=valid_adset[:][1],
        n_classes=n_classes,
    )
    print(acc)
    cv = d_ev.cross_validate(
        model_cls=d_nn.Disyak,
        adset=response,
        model_config=d_ut.ModuleConfig(
            record_metrics=False, outlayer_type="regression"
        ),
        **dict(cv_kwargs.copy(), **{"validation": None, "init_bias": True}),
    )
    print(cv)


# test_distillation()

# %%


def test_holdout():
    result = d_ev.holdout(
        module_cls=d_nn.Disyak,
        in_features=n_features,
        data=adata1,
        split_fns={
            n.lower(): lambda x: (
                (
                    x[x.obs["Project_ID"].str.contains(n)],
                    x[~x.obs["Project_ID"].str.contains(n)],
                )
            )
            for n in ["TARGET", "TCGA"]
        },
        n_classes=n_classes,
        to_encode=("Sample_Type", "tumor_type"),
        trainer_kwargs={
            "max_epochs": 5,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
        },
    )
    return result


result = test_holdout()

t = result["target"]

# %%


def test_cross_val():
    cv: pd.DataFrame = d_ev.cross_validate(
        model_cls=d_nn.Disyak,
        adset=d_ut.AnnDataset(adata, to_encode=labels),
        **dict(
            cv_kwargs.copy(),
            **{
                "trainer_kwargs": {
                    "max_epochs": 3,
                    "enable_checkpointing": False,
                    "enable_progress_bar": False,
                    # "num_sanity_val_steps": 0,
                }
            },
        ),
        minimal=False,
    )
    return cv


cv, cm = test_cross_val()

# %%


def test_random():
    return random_softmax_loss(
        model=d_nn.Disyak(in_features=n_features, n_classes_per_task=n_classes),
        trainer=L.Trainer(
            max_epochs=100,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
        ),
        train=train_l,
        test=test_l,
    )


def test_init():
    model = d_nn.Disyak(in_features=n_features, n_classes_per_task=n_classes)
    return init_test(model, train_l)


# pred, bias = test_init()
