#!/usr/bin/env ipython

import too_predict.deep.branchnet as bn
import too_predict.deep.evaluation as d_ev
import too_predict.deep.torch_utils as d_ut
import too_predict.transformer as tt
import too_predict.utils as ut
import torch
from too_predict.deep.evaluation import multitask_acc
from too_predict.imputer import Imputer
from torch.utils.data import DataLoader

import lightning as L

# %%


torch.set_default_dtype(torch.float32)


adata = ut.training_data_internal_test(minimal=True)  # 1000 features
transformer = tt.Transformer("clr", impute_fn=Imputer("plus_one"), inplace=False)
adata = transformer.fit_transform(adata)

train, test = ut.train_test_split_ad(adata)
encode = ["Sample_Type", "tumor_type"]
adset = d_ut.AnnDataset(adata, to_encode=encode)
train_adset = d_ut.AnnDataset(train, to_encode=encode)
test_adset = d_ut.AnnDataset(test, to_encode=encode)


def test_train_together():
    n_features, n_classes = d_ut.data_spec(adata, y=encode)

    trainer = L.Trainer(max_epochs=10)

    mbn = bn.MultiBranch(
        n_features, n_classes_per_task=n_classes, fit_separately_kwargs={"epochs": 2}
    )
    mbn.fit_trees(train_adset)

    train_acc_before = multitask_acc(
        y_true=train_adset[:][1],
        predictions=mbn.predict_step(train_adset[:][0]),
        task_names=["Sample_Type", "tumor_type"],
        n_classes=n_classes,
    )
    test_acc_before = multitask_acc(
        y_true=test_adset[:][1],
        predictions=mbn.predict_step(test_adset[:][0]),
        task_names=["Sample_Type", "tumor_type"],
        n_classes=n_classes,
    )
    trainer.fit(model=mbn, train_dataloaders=DataLoader(train_adset, batch_size=32))
    train_acc_after = multitask_acc(
        y_true=train_adset[:][1],
        predictions=mbn.predict_step(train_adset[:][0]),
        task_names=["Sample_Type", "tumor_type"],
        n_classes=n_classes,
    )
    test_acc_after = multitask_acc(
        y_true=test_adset[:][1],
        predictions=mbn.predict_step(test_adset[:][0]),
        task_names=["Sample_Type", "tumor_type"],
        n_classes=n_classes,
    )
    print(f"{train_acc_before}")
    print(f"{train_acc_after}")
    print(f"{test_acc_before}")
    print(f"{test_acc_after}")


def test_train_before():
    n_features, n_classes = d_ut.data_spec(adata, y=encode)
    trainer = L.Trainer(max_epochs=10)
    mbn = bn.MultiBranch(
        n_features, n_classes_per_task=n_classes, fit_separately_kwargs={"epochs": 2}
    )
    mbn.fit_trees(train_adset)
    mbn.fit_branchnets(train_adset, freeze=True, epochs=20, show_progress=False)
    train_acc_before = multitask_acc(
        y_true=train_adset[:][1],
        predictions=mbn.predict_step(train_adset[:][0]),
        task_names=["Sample_Type", "tumor_type"],
        n_classes=n_classes,
    )
    test_acc_before = multitask_acc(
        y_true=test_adset[:][1],
        predictions=mbn.predict_step(test_adset[:][0]),
        task_names=["Sample_Type", "tumor_type"],
        n_classes=n_classes,
    )
    trainer.fit(model=mbn, train_dataloaders=DataLoader(train_adset, batch_size=32))
    train_acc_after = multitask_acc(
        y_true=train_adset[:][1],
        predictions=mbn.predict_step(train_adset[:][0]),
        task_names=["Sample_Type", "tumor_type"],
        n_classes=n_classes,
    )
    test_acc_after = multitask_acc(
        y_true=test_adset[:][1],
        predictions=mbn.predict_step(test_adset[:][0]),
        task_names=["Sample_Type", "tumor_type"],
        n_classes=n_classes,
    )
    print(f"{train_acc_before=}")
    print(f"{train_acc_after=}")
    print(f"{test_acc_before=}")
    print(f"{test_acc_after=}")


def test_callback():
    n_features, n_classes = d_ut.data_spec(adata, y=encode)
    trainer = L.Trainer(max_epochs=10)
    mbn = bn.MultiBranch(
        n_features,
        fit_separately=True,
        n_classes_per_task=n_classes,
        fit_separately_kwargs={"epochs": 2},
    )
    trainer.fit(model=mbn, train_dataloaders=DataLoader(train_adset, batch_size=32))
    train_acc_after = multitask_acc(
        y_true=train_adset[:][1],
        predictions=mbn.predict_step(train_adset[:][0]),
        task_names=["Sample_Type", "tumor_type"],
        n_classes=n_classes,
    )
    test_acc_after = multitask_acc(
        y_true=test_adset[:][1],
        predictions=mbn.predict_step(test_adset[:][0]),
        task_names=["Sample_Type", "tumor_type"],
        n_classes=n_classes,
    )
    print(f"{train_acc_after=}")
    print(f"{test_acc_after=}")


# test_train_together()
# test_train_before()
# test_callback()


def test_cross_val():
    n_features, n_classes = d_ut.data_spec(adata, y=encode)
    mkwargs = {"in_features": n_features, "n_classes_per_task": n_classes}
    tkwargs = {"max_epochs": 2}
    d_ev.cross_validate(
        model_fn=lambda **kwargs: bn.MultiBranch(**kwargs),
        model_kwargs=mkwargs,
        trainer_kwargs=tkwargs,
        n_classes=n_classes,
        adset=adset,
        batch_size=32,
        # validation=test_adset,
    )


test_callback()
test_cross_val()
