#!/usr/bin/env ipython

import math
from collections.abc import Sequence

import numpy as np
import sklearn.metrics as met
import too_predict.deep.torch_utils as d_ut
import too_predict.evaluation as te
import too_predict.utils as ut
import torch
import torch.utils.data as tdata
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def multitask_acc(
    y_true: Tensor | np.ndarray | DataLoader | Dataset,
    predictions: Tensor | np.ndarray,
    task_names: Sequence[str] | None = None,
) -> dict:
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
            return _final(tdata.random_split(dataset, (train_size, test_size)))
        test_size = math.floor(len(dataset) * test_size)
    elif train_size is None:
        train_size = n_samples - test_size
    indices = set(range(n_samples))
    test_indices = ut.RNG.choice(indices, size=test_size)
    indices -= set(test_indices)
    train_indices = ut.RNG.choice(indices, size=train_size)
    return _final([tdata.Subset(train_indices), tdata.Subset(test_indices)])
