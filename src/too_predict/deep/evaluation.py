#!/usr/bin/env ipython

import math
from collections.abc import Callable, Sequence
from pathlib import Path

import anndata as ad
import lightning as L
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import too_predict.deep.torch_utils as d_ut
import too_predict.evaluation as te
import too_predict.utils as ut
import torch
import torch.nn as nn
from lightning.pytorch.loggers import CometLogger, Logger
from too_predict.deep.distillation import TeacherResponse, use_kd_criterion
from too_predict.deep.metrics import multitask_acc, multitask_all_metrics
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split


def train_test_split_torch(
    dataset: Dataset,
    train_size: float | int | None = None,
    test_size: float | int = 0.25,
    valid: float | int | None | bool = None,
    **kwargs,
) -> tuple[DataLoader, ...]:
    """Get train, test and optionally validation splits from dataset

    Parameters
    ----------
    train_size : count or proportion of train dataset
    test_size : count or proportion of test dataset
    valid_size : if a boolean True, valid_size is interpreted as
        valid_size = 1 - train_size - test_size (if both are given)
        valid_size = test_size (if train_size is not given)

    Returns
    -------
    Tuple of DataLoaders
    """
    n_samples = len(dataset)
    if isinstance(train_size, int):
        train_size = math.floor(train_size / n_samples)
    if isinstance(test_size, int):
        test_size = math.floor(test_size / n_samples)
    if (
        not isinstance(valid, bool)
        and valid is not None
        and not isinstance(valid, float)
    ):
        valid = math.floor(valid / n_samples)

    if not test_size:
        raise ValueError("A test split must be provided!")

    # Only test size provided, valid True
    if isinstance(valid, bool) and valid and train_size is None:
        valid = test_size
        lengths = (1 - test_size - valid, test_size, valid)
    # Only test provided, valid float
    elif isinstance(valid, float) and train_size is None:
        lengths = (1 - test_size - valid, test_size, valid)
    # Test and train provided, valid True
    elif isinstance(valid, bool) and valid and train_size:
        valid = 1 - train_size - test_size
        lengths = (train_size, test_size, valid)
    # Test size provided, valid False|None
    elif test_size and train_size is None and not valid:
        lengths = (1 - test_size, test_size)
    # Test and train provided, valid False|None
    elif train_size and test_size and not valid:
        lengths = (train_size, test_size)
    # All provided
    else:
        lengths = (train_size, test_size, valid)

    print(f"Split proportions: {lengths}")
    if np.sum(lengths) > 1:
        raise ValueError("Invalid split sizes provided!")

    return tuple(DataLoader(d, **kwargs) for d in random_split(dataset, lengths))


# TODO: give this params for working with distillation
def holdout(
    trainer_kwargs: dict,
    model_cls: d_ut.MultiModule,
    adata: ad.AnnData,
    to_encode: tuple[str],
    n_classes: Sequence[int],
    in_features: int,
    split_fns: dict[str, Callable[[ad.AnnData], tuple[ad.AnnData, ad.AnnData]]]
    | None = None,
    model_config: d_ut.ModuleConfig | None = None,
    model_kwargs: dict | None = None,
    logger_fn: Callable[[str], Logger] | None = None,
    save_split_path: Path | None = None,
    split_masks: dict[str, tuple] | None = None,
    verbose: bool = False,
    minimal: bool = False,
    device: str = "cpu",
    outdir: Path | None = None,
    scaler: d_ut.TorchScaler | None = None,
    validation: float | None = None,
    **kwargs,
) -> dict:
    """Wrapper function for doing the classic holdout method (train-test-split) with
    torch module

    Parameters
    ---------
    Same parameters as original holdout
    split_fn : A dictionary of function that splits adata into a tuple of train, test
    split_indices : A dictionary of mapping test set names to (train, test) boolean indices
    validation : float | None
        If not none, the percentage of the test split to use for validation data in
        early stopping and reporting training metrics
    logger_fn : Optional function taking name of holdout set and returning Lightning
        logger

    Returns
    ------
    A dictionary containing model evaluation results for each task i.e.
        split->task->results_dict
    If `minimal`, is a dictionary of split->dict[task->accuracy]
    """
    if split_fns is None and split_masks is None:
        raise ValueError("Either split_fns or split_indices must be given!")
    split_is_fn: bool = split_fns is not None
    model_kwargs = ut.if_none(model_kwargs, {})

    def helper(set_label, splitter, cur_adata):
        cur_adata = cur_adata.copy()
        if save_split_path is not None:
            root = save_split_path.joinpath(set_label)
            root.mkdir(exist_ok=True)
            trainer_kwargs["default_root_dir"] = root
        if logger_fn is not None:
            trainer_kwargs["logger"] = logger_fn(set_label)
        trainer = L.Trainer(**trainer_kwargs)
        if split_is_fn:
            x_train, x_test = splitter(cur_adata)
        else:
            x_train = cur_adata[splitter[0], :]
            x_test = cur_adata[splitter[1], :]
        v_adata: ad.AnnData | None
        if validation is not None:
            x_test, v_adata = ut.train_test_split_ad(x_test, test_size=validation)
        else:
            v_adata = None
        if verbose:
            print(
                f"Train, test sizes for set {set_label}: {x_train.shape[0]}, {x_test.shape[0]}"
            )
        v_loader = (
            DataLoader(d_ut.AnnDataset(v_adata, to_encode=to_encode, device=device))
            if isinstance(v_adata, ad.AnnData)
            else None
        )
        if save_split_path is not None:
            x_train.obs.to_csv(
                save_split_path.joinpath(f"{set_label}_train_obs.csv"), index=False
            )
            x_test.obs.to_csv(
                save_split_path.joinpath(f"{set_label}_test_obs.csv"), index=False
            )
            if validation is not None:
                v_adata.obs.to_csv(
                    save_split_path.joinpath(f"{set_label}_validation_obs.csv"),
                    index=False,
                )
        train_dset: d_ut.AnnDataset = d_ut.AnnDataset(
            x_train, to_encode=to_encode, device=device
        )
        updated = d_ut.update_batch_strategy(kwargs, train_dset, default_batch_size=32)
        test_dset = d_ut.AnnDataset(x_test, to_encode=to_encode, device=device)
        train_l = DataLoader(train_dset, **updated)
        x_test_tensor, y_true = test_dset[:]
        if scaler is not None:
            scaler.fit(train_l.dataset[:][0])
            model_config.scaler = scaler
        model_config.init_device = device
        with trainer.init_module():
            model = model_cls.new(
                in_features=in_features,
                n_classes_per_task=n_classes,
                conf=model_config,
                **model_kwargs,
            )

        trainer.fit(model=model, train_dataloaders=train_l, val_dataloaders=v_loader)
        model.to(device)

        if not minimal:
            proba = model.predict_proba(x_test_tensor)
            y_true = test_dset.decode(y_true)
            res: dict = multitask_all_metrics(
                y_true=y_true,
                scores=proba,
                task_names=to_encode,
                n_classes=n_classes,
            )
            return res
        acc_kwargs = {"n_classes": n_classes, "task_names": to_encode}
        test_acc = multitask_acc(
            y_true=y_true, predictions=model.predict_step(test_dset[:]), **acc_kwargs
        )
        train_acc = multitask_acc(
            y_true=train_dset[:][1],
            predictions=model.predict_step(train_dset[:]),
            **acc_kwargs,
        )
        return test_acc, train_acc

    splitters: dict = split_fns if split_fns is not None else split_masks
    result: dict = {}
    for set_label, splitter in splitters.items():
        result[set_label] = helper(set_label, splitter, adata)
        if outdir is not None and not minimal:
            cur_outdir = outdir.joinpath(set_label)
            cur_outdir.mkdir(exist_ok=True)
            for task in to_encode:
                te.write_cross_val(
                    result[set_label][task], outdir=cur_outdir, prefix=f"{task}_"
                )
    if outdir is not None and minimal:
        tasks = list(next(iter(result.values()))[0].keys())
        to_df = {"set": result.keys()}
        for t in tasks:
            to_df[f"{t}_test_acc"] = [v[0][t] for v in result.values()]
            to_df[f"{t}_train_acc"] = [v[1][t] for v in result.values()]
        df = pd.DataFrame(to_df)
        df = d_ut.tensor_cols_to_float(df)
        df.to_csv(outdir.joinpath("accuracy.csv"), index=False)
    return result


def cross_validate(
    model_cls: d_ut.MultiModule,
    trainer_kwargs: dict,
    adset: d_ut.AnnDataset | TeacherResponse,
    n_classes: Sequence[int],
    in_features: int,
    model_config: d_ut.ModuleConfig | None = None,
    model_kwargs: dict | None = None,
    random_state: int | None = ut.RANDOM_STATE,
    callbacks: list[Callable[[], L.Callback]] | None = None,
    n_splits: int = 5,
    validation: Dataset | None = None,
    verbose: bool = False,
    logger_fn: Callable[[int], Logger] | None = None,
    save_path: Path | None = None,
    scaler: d_ut.TorchScaler | None = None,
    with_train_acc: bool = True,
    device: str = "cpu",
    init_bias: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Run cross-validation

    Parameters
    ----------
    trainer_kwargs : Keyword arguments passed to Lightning trainer
    logger_fn : Optional function taking in the fold number and returning a Lightning
        logger
    kwargs : key-word arguments passed to DataLoader. Set batch_size == -1 to use the entire
        training set in each cross-validation loop

    Returns
    -------
    Dataframe summarizing cv accuracies
    """
    if verbose:
        print("Beginning cross validation...")
    cv = ms.KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    splits = cv.split(adset)
    metrics: dict = {"fold": []}
    model_config = model_config if model_config else d_ut.ModuleConfig()
    tasks = adset.label_cols
    trainer_kwargs["accelerator"] = device
    for task in tasks:
        metrics[f"{task}_valid_acc"] = []
        if with_train_acc:
            metrics[f"{task}_train_acc"] = []
    if scaler is not None:
        model_config.scaler = scaler
    if isinstance(adset, TeacherResponse):
        distillation: bool = True
        labels: Tensor | None = adset.get_targets()
    else:
        distillation = False
        labels = None
    model_kwargs = ut.if_none(model_kwargs, {})
    for fold, (train_idx, test_idx) in enumerate(splits):
        if verbose:
            print(f"fold {fold} started")
        train_set = Subset(adset, train_idx)
        updated = d_ut.update_batch_strategy(kwargs, train_set, default_batch_size=32)
        train: DataLoader = DataLoader(train_set, **updated)
        test: Dataset = Subset(adset, test_idx)
        val_loader = DataLoader(validation) if validation is not None else None

        if save_path is not None:
            root = save_path.joinpath(f"fold_{fold}")
            root.mkdir(exist_ok=True)
            trainer_kwargs["default_root_dir"] = root

        if logger_fn is not None:
            trainer_kwargs["logger"] = logger_fn(fold)

        if scaler is not None:
            scaler.fit(train.dataset[:][0])

        if callbacks is not None:
            trainer_kwargs["callbacks"] = [c() for c in callbacks]

        model_config.init_device = device
        trainer = L.Trainer(**trainer_kwargs)
        with trainer.init_module():
            model = model_cls.new(
                in_features=in_features,
                n_classes_per_task=n_classes,
                conf=model_config,
                **model_kwargs,
            )
            if distillation:
                use_kd_criterion(model)
            if init_bias:
                d_ut.init_lazy(model, loader=train)
                model.init_out_bias(targets=train)
        trainer.fit(model=model, train_dataloaders=train, val_dataloaders=val_loader)
        model = model.to(torch.device(device))

        if not isinstance(adset, TeacherResponse):
            test_y = test[:][1]
            train_y = train.dataset[:][1]
        else:
            test_y = labels[test_idx, :]
            train_y = labels[train_idx, :]
        acc = multitask_acc(
            y_true=test_y,
            predictions=model.predict_step(test[:]),
            task_names=tasks,
            n_classes=n_classes,
        )
        if with_train_acc:
            train_acc = multitask_acc(
                y_true=train_y,
                predictions=model.predict_step(train.dataset[:]),
                task_names=tasks,
                n_classes=n_classes,
            )
        for t in tasks:
            metrics[f"{t}_valid_acc"].append(acc[t])
            if with_train_acc:
                metrics[f"{t}_train_acc"].append(train_acc[t])
        metrics["fold"].append(fold)
    if verbose:
        print("Cross validation completed")
    df = pd.DataFrame(metrics)
    df = d_ut.tensor_cols_to_float(df)
    if isinstance(trainer.logger, CometLogger):
        trainer.logger.experiment.log_table("cv_summary.csv", df, True)
    return df


# * Test functions


def init_test(model: d_ut.MultiModule, loader: DataLoader) -> Tensor | tuple:
    d_ut.init_lazy(model, loader)
    biases = d_ut.get_initial_out_bias(model.conf.outlayer_type, loader)
    model.conf.out_bias = biases
    model.init_out_bias()
    return model.predict_proba(loader), biases


def random_softmax_loss(
    model: d_ut.MultiModule,
    trainer: L.Trainer,
    train: DataLoader,
    test: DataLoader,
):
    x, y = train.dataset[:]
    rand_indices = torch.randperm(y.shape[0])
    y = y[rand_indices]
    shuffled_train = TensorDataset(x, y)
    if train.batch_size:
        new_loader = DataLoader(shuffled_train, batch_size=train.batch_size)
    elif train.batch_sampler:
        new_loader = DataLoader(shuffled_train, batch_sampler=train.batch_sampler)
    else:
        new_loader = DataLoader(shuffled_train)
    trainer.fit(model, train_dataloaders=new_loader)
    train_result = trainer.test(model, dataloaders=train)
    test_result = trainer.test(model, dataloaders=test)
    return train_result, test_result
