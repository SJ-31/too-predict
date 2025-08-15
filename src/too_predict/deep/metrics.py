#!/usr/bin/env ipython

from collections.abc import Sequence

import lightning as L
import numpy as np
import too_predict.deep.torch_utils as d_ut
import torch
import torch.nn as nn
import torchmetrics.functional.classification as tmet
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def multitask_acc(
    predictions: Tensor | np.ndarray,
    y_true: Tensor | DataLoader | Dataset | np.ndarray,
    n_classes: Sequence[int],
    task_names: Sequence[str] | None = None,
) -> dict:
    """Compute accuracy independently on each prediction task

    Parameters
    ----------
    predictions : multitask predictions, same shape as y_true
    y_true : true values, of shape n_samples x n_tasks
    n_classes : iterable where the ith index is the number of classes in the ith task
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
    elif isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions)
    y_iter = d_ut.iter_cols(y_true)
    pred_iter = d_ut.iter_cols(predictions)
    if task_names is None:
        task_names = [str(i) for i in range(predictions.shape[1])]
    result = {}
    for i, (task, y, pred) in enumerate(zip(task_names, y_iter, pred_iter)):
        result[task] = tmet.accuracy(
            preds=pred, target=y, num_classes=n_classes[i], task="multiclass"
        ).item()
    return result


def multitask_all_metrics(
    scores: Sequence[Tensor],
    y_true: Tensor,
    n_classes: Sequence[int],
    task_names: Sequence[str] | None = None,
) -> dict:
    if y_true.shape[1] != len(scores):
        raise ValueError(
            "The given truth matrix does not match the sequence of scores!"
        )
    to_iter = d_ut.iter_cols(y_true)
    if task_names is None:
        task_names = [str(i) for i in range(len(scores))]
    result = {}
    for task, truth, score, n in zip(task_names, to_iter, scores, n_classes):
        result[task] = {}
        result[task]["acc"] = tmet.accuracy(
            preds=score, target=truth, num_classes=n, task="multiclass"
        )  # NOTE: the multiclass_accuracy version produced a different result
        result[task]["kappa"] = tmet.cohen_kappa(
            preds=score, target=truth, num_classes=n, task="multiclass"
        )
        result[task]["mcc"] = tmet.matthews_corrcoef(
            preds=score, target=truth, num_classes=n, task="multiclass"
        )
        result[task]["auroc"] = tmet.auroc(
            preds=score, target=truth, num_classes=n, task="multiclass"
        )
        result[task]["cm"] = tmet.confusion_matrix(
            preds=score, target=truth, num_classes=n, task="multiclass"
        )
    return result


def multitask_cross_entropy_loss(
    y_pred: Tensor,
    y_true: Tensor,
    weights: Tensor | None = None,
    model: L.LightningModule | None = None,
    prefix: str = "",
) -> Tensor:
    losses: Tensor = torch.empty(y_true.shape[1])
    for i, (task_pred, task_y) in enumerate(
        zip(y_pred, torch.unbind(y_true, dim=1))
    ):  # Gives y_hat = softmax(Xw + b)
        # tensor of shape n_samples, n_classes
        loss = nn.functional.cross_entropy(task_pred, task_y)
        if model is not None:
            name = f"loss_{i}" if not prefix else f"{prefix}_loss_{i}"
            model.log(name, loss)
        losses[i] = loss
        # Get loss on tasks separately
    if weights is not None and len(weights) == len(losses):
        losses = losses * weights
    return losses.sum()
