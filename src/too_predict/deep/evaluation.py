#!/usr/bin/env ipython

import math
from collections.abc import Callable, Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import sklearn.metrics as met
import sklearn.model_selection as ms
import too_predict.deep.torch_utils as d_ut
import too_predict.evaluation as te
import too_predict.utils as ut
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split


def multitask_acc(
    y_true: Tensor | np.ndarray | DataLoader | Dataset,
    predictions: Tensor | np.ndarray,
    task_names: Sequence[str] | None = None,
) -> dict:
    """Compute accuracy independently on each prediction task

    Parameters
    ----------
    y_true : true values, of shape n_samples x n_tasks
    predictions : multitask predictions, same shape as y_true
    task_names : names of prediction tasks

    Returns
    -------
    Dictionary of task_name->task_accuracy. If names not provided, indices in
        y_true are used instead
    """
    if isinstance(y_true, Dataset):
        y_true = y_true[:][1]
    elif isinstance(y_true, DataLoader):
        y_true = y_true.dataset[:][1]
    y_iter = d_ut.iter_cols(y_true)
    pred_iter = d_ut.iter_cols(predictions)
    if task_names is None:
        task_names = [str(i) for i in range(predictions.shape[1])]
    result = {}
    for task, y, pred in zip(task_names, y_iter, pred_iter):
        result[task] = met.accuracy_score(y, pred)
    return result


def multitask_all_metrics(
    y_true: Tensor | np.ndarray,
    scores: Sequence[Tensor | np.ndarray],
    task_names: Sequence[str] | None = None,
    task_classes: Sequence[Sequence] | None = None,
) -> dict:
    if y_true.shape[1] != len(scores):
        raise ValueError(
            "The given truth matrix does not match the sequence of scores!"
        )
    to_iter = d_ut.iter_cols(y_true)
    if task_classes is None and isinstance(y_true, Tensor):
        task_classes = [list(y.unique()) for y in to_iter]
    elif task_classes is None and isinstance(y_true, np.ndarray):
        task_classes = [list(np.unique(y)) for y in to_iter]
    if task_names is None:
        task_names = [str(i) for i in range(len(scores))]
    result = {}
    for task, truth, score, class_names in zip(
        task_names, to_iter, scores, task_classes
    ):
        result[task] = te.get_all_metrics(truth, score, classes=class_names)
    return result


def train_test_split_torch(
    dataset: Dataset,
    train_size: float | int | None = None,
    test_size: float | int = 0.25,
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    n_samples = len(dataset)

    def _final(dsets):
        return DataLoader(dsets[0], **kwargs), DataLoader(dsets[1], **kwargs)

    if isinstance(test_size, float):
        if train_size is None:
            train_size = 1 - test_size
        if isinstance(train_size, float):
            return _final(random_split(dataset, (train_size, test_size)))
        test_size = math.floor(len(dataset) * test_size)
    elif train_size is None:
        train_size = n_samples - test_size
    indices = set(range(n_samples))
    test_indices = ut.RNG.choice(indices, size=test_size)
    indices -= set(test_indices)
    train_indices = ut.RNG.choice(indices, size=train_size)
    return _final([Subset(train_indices), Subset(test_indices)])


def holdout(
    trainer: d_ut.Trainer,
    adata: ad.AnnData,
    to_encode: tuple[str],
    split_fns: dict[str, Callable[[ad.AnnData], tuple[ad.AnnData, ad.AnnData]]]
    | None = None,
    save_split_path: Path | None = None,
    split_masks: dict[str, tuple] | None = None,
    verbose: bool = False,
    minimal: bool = False,
    **kwargs,
) -> dict:
    """Wrapper function for doing the classic holdout method (train-test-split) with
    torch module

    Parameters
    ---------
    Same parameters as original holdout
    split_fn: A dictionary of function that splits adata into a tuple of train, test
    split_indices: A dictionary of mapping test set names to (train, test) boolean indices

    Return
    ------
    A dictionary containing model evaluation results for each task
    If `minimal`, is a dictionary of split->dict[task->accuracy]
    """
    if split_fns is None and split_masks is None:
        raise ValueError("Either split_fns or split_indices must be given!")

    split_is_fn: bool = split_fns is not None

    def helper(set_label, splitter, cur_adata):
        cur_adata = cur_adata.copy()
        if split_is_fn:
            x_train, x_test = splitter(cur_adata)
        else:
            x_train = cur_adata[splitter[0], :]
            x_test = cur_adata[splitter[1], :]
        if verbose:
            print(
                f"Train, test sizes for set {set_label}: {x_train.shape[0]}, {x_test.shape[0]}"
            )
        if save_split_path is not None:
            x_train.obs.to_csv(
                save_split_path.joinpath(f"{set_label}_train_obs.csv"), index=False
            )
            x_test.obs.to_csv(
                save_split_path.joinpath(f"{set_label}_test_obs.csv"), index=False
            )
        train_l = DataLoader(d_ut.AnnDataset(x_train, to_encode=to_encode), **kwargs)
        test_dset = d_ut.AnnDataset(x_test, to_encode=to_encode)
        x_test_tensor, y_true = test_dset[:]

        trainer(train_l)

        if not minimal:
            task_classes = [list(np.unique(adata.obs[t])) for t in to_encode]
            proba = trainer.model.predict_proba(x_test_tensor)
            y_true = test_dset.decode(y_true)
            res: dict = multitask_all_metrics(
                y_true, proba, task_names=to_encode, task_classes=task_classes
            )
            return res
        pred = trainer.model.predict(x_test_tensor)

        return multitask_acc(y_true=y_true, predictions=pred, task_names=to_encode)

    splitters: dict = split_fns if split_fns is not None else split_masks
    result: dict = {}
    for set_label, splitter in splitters.items():
        result[set_label] = helper(set_label, splitter, adata)
    return result
