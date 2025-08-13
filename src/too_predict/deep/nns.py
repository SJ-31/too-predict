#!/usr/bin/env ipython

from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, override

import numpy as np
import too_predict.deep.torch_utils as d_ut
import torch
import torch.nn as nn
from too_predict.deep.metrics import multitask_cross_entropy_loss
from torch import Tensor
from torch.utils.data import Dataset
from xgboost import XGBClassifier

"""
References
[1] Disyak, M. (2021). A multi-task machine learning pipeline for the classification and analysis of cancers from gene expression data (T). University of British Columbia. Retrieved from https://open.library.ubc.ca/collections/ubctheses/24/items/1.0395883
"""


class FullyConnected(nn.Module):
    """Helper class for instantiating a fully-connected layer with dropout and batch norm"""

    def __init__(
        self,
        n_in: int,
        activation: nn.Module = nn.Tanh,
        dropout_p: float = 0.2,
        batch_norm: Literal["before", "after", "none"] = "none",
        batch_norm_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        mods: nn.ModuleList = nn.ModuleList([nn.LazyLinear(n_in)])
        bn_kwargs: dict = {} if batch_norm_kwargs is None else batch_norm_kwargs
        if batch_norm == "before":
            mods.append(nn.LazyBatchNorm1d(**bn_kwargs))
        mods.append(activation())
        if batch_norm == "after":
            mods.append(nn.LazyBatchNorm1d(**bn_kwargs))
        if dropout_p > 0:
            mods.append(nn.Dropout(p=dropout_p))
        self.net: nn.Sequential = nn.Sequential(*mods)

    @override
    def forward(self, x):
        return self.net(x)

    @staticmethod
    def at_depth(n: int = 1, **kwargs) -> nn.Sequential:
        return nn.Sequential(*[FullyConnected(**kwargs) for _ in range(n)])


class HardSharer(d_ut.MultiModule):
    """Implementation of [1]

    Notes
    -----
    During training, columns of the tensor y_true should be ordered by increasing
        task specificity e.g. [ organ_system, disease_state, cancer_type ] etc.
        so that the more specific tasks can make use of all hidden layers
    """

    def __init__(
        self,
        in_features: int,
        n_classes_per_task: list[int],
        conf: d_ut.ModuleConfig | None = None,
        n_hidden: int | None = None,
        n_shared_layers: int = 1,
        n_layers_per_task: Sequence[int] | None = None,
        task_weights: Tensor | Sequence | None = None,
        dropout_p: float = 0.2,
        batch_norm: Literal["before", "after", "none"] = "none",
        batch_norm_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            in_features=in_features,
            n_classes_per_task=n_classes_per_task,
            conf=conf,
            **kwargs,
        )
        if n_hidden is None:
            n_hidden = in_features
        self.task_layers: nn.ModuleList = nn.ModuleList()

        self.hidden_shared: nn.Sequential = FullyConnected.at_depth(
            n=n_shared_layers,
            activation=nn.Tanh,
            n_in=n_hidden,
            dropout_p=dropout_p,
            batch_norm=batch_norm,
            batch_norm_kwargs=batch_norm_kwargs,
        )
        if n_layers_per_task is None:
            n_layers_per_task = [1 for _ in n_classes_per_task]

        for n_classes, n_layers in zip(n_classes_per_task, n_layers_per_task):
            task_modules: nn.ModuleList = nn.ModuleList()
            if n_layers > 0:
                task_modules.append(
                    FullyConnected.at_depth(
                        n=n_layers,
                        activation=nn.Tanh,
                        n_in=n_hidden,
                        dropout_p=dropout_p,
                        batch_norm=batch_norm,
                        batch_norm_kwargs=batch_norm_kwargs,
                    )
                )
            out = nn.LazyLinear(n_classes)
            task_modules.append(out)
            self.task_layers.append(nn.Sequential(*task_modules))

    @override
    def forward(self, X):
        result = []
        hidden: torch.Tensor = self.hidden_shared(X)
        for mod in self.task_layers:
            result.append(mod(hidden))
        return tuple(result)

    @override
    def criterion(self, y_pred, y_true, context: str | None = None):
        total_loss: torch.Tensor = 0
        if self.n_tasks > 1:
            total_loss += multitask_cross_entropy_loss(
                y_pred,
                y_true,
                weights=self.conf.task_weights,
                model=self,
                prefix=context,
            )
        else:
            total_loss += nn.functional.cross_entropy(y_pred, y_true)
        total_loss += self.l2() + self.l1()
        return total_loss


class Disyak(d_ut.MultiModule):
    """Implementation of [1]

    Notes
    -----
    During training, columns of the tensor y_true should be ordered by increasing
        task specificity e.g. [ organ_system, disease_state, cancer_type ] etc.
        so that the more specific tasks can make use of all hidden layers
    """

    def __init__(
        self,
        in_features: int,
        n_classes_per_task: list[int],
        n_hidden: int | None = 2000,
        conf: d_ut.ModuleConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            in_features=in_features,
            n_classes_per_task=n_classes_per_task,
            conf=conf,
            **kwargs,
        )
        if n_hidden is None:
            n_hidden = in_features
        self.hlayers: nn.ModuleList = nn.ModuleList()
        self.olayers: nn.ModuleList = nn.ModuleList()
        self.acti: nn.Module = nn.Tanh()
        self.softmax: nn.Softmax = nn.Softmax()
        self.dropout: nn.Dropout = nn.Dropout(p=self.conf.dropout_p)
        for n_classes in n_classes_per_task:
            self.hlayers.append(nn.LazyLinear(n_hidden))
            out = nn.LazyLinear(n_classes)
            self.olayers.append(out)

    def _activate(self, input: Tensor) -> Tensor:
        return self.dropout(self.acti(input))

    @override
    def reset_parameters(self):
        for m, o in zip(self.hlayers, self.olayers):
            d_ut.reset_sequential(o)
            d_ut.reset_sequential(m)

    @override
    def forward(self, X):
        modules, outs = iter(self.hlayers), iter(self.olayers)
        result = []
        hidden: torch.Tensor = self._activate(next(modules)(X))
        o1 = next(outs)
        result.append(o1(hidden))
        for out, m in zip(outs, modules):
            hidden = self._activate(m(hidden))
            result.append(out(hidden))
        return tuple(result)

    @override
    def criterion(self, y_pred, y_true, context: str | None = None):
        total_loss: torch.Tensor = 0
        if self.n_tasks > 1:
            total_loss += multitask_cross_entropy_loss(
                y_pred,
                y_true,
                weights=self.conf.task_weights,
                model=self,
                prefix=context,
            )
        else:
            total_loss += nn.functional.cross_entropy(y_pred, y_true)
        total_loss += self.l2() + self.l1()
        return total_loss


class RepLearner(d_ut.MultiModule):
    def __init__(
        self,
        in_features: int,
        n_classes_per_task: list[int],
        record_metrics: bool = True,
        model_fn: Callable = lambda: XGBClassifier(),
        pretrained: Sequence[Path] | None = None,
        task_names: Sequence[str] | None = None,
        task_weights: Tensor | Sequence | None = None,
        l1_pars: dict | None = None,
        l2_pars: dict | None = None,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
        scheduler_config: dict | None = None,
        cache: str | None | Sequence = None,
        log_norm: bool = False,
        scaler: d_ut.TorchScaler | None = None,
    ) -> None:
        super().__init__(
            in_features,
            n_classes_per_task,
            record_metrics,
            task_names,
            task_weights,
            l1_pars,
            l2_pars,
            optimizer_fn,
            scheduler_fn,
            scheduler_config,
            cache,
            log_norm,
            scaler,
        )
        self._model_fn: Callable = model_fn
        self._pretrained: Sequence[Path | None] = (
            pretrained if pretrained else [None] * len(self.n_classes)
        )
        self.models: list = []

    def fit(self, dataset):
        x, ys = dataset[:]
        for i, y, pretrained_path in enumerate(
            zip(d_ut.iter_cols(ys), self.pretrained)
        ):
            if not pretrained_path:
                model = self.model_fn()
                model.fit(x, y)
            else:
                with open(pretrained_path, "rb") as f:
                    model = pickle.load(f)
            self.models[i] = model


class Baseline:
    """A baseline class for multitask prediction. Consists of XGBoost models
    trained independently on each task
    """

    def __init__(self, in_features: int, n_classes_per_task: list[int], **kwargs):
        """ """
        self.models: list = [XGBClassifier(**kwargs) for _ in n_classes_per_task]

    def fit(self, X, y=None):
        if isinstance(X, Tensor):
            X = X.numpy()
        elif isinstance(X, Dataset):
            x_tensor, y_tensor = X[:]
            X = x_tensor.numpy()
            y = y_tensor.numpy()
        for model, y in zip(self.models, d_ut.iter_cols(y)):
            model.fit(X, y)

    def predict_step(self, batch):
        try:
            x, _ = batch
        except ValueError:
            x = batch
        if isinstance(x, Tensor):
            x = x.numpy()
        return torch.tensor(np.column_stack(tuple(m.predict(x) for m in self.models)))

    def predict_proba(self, batch):
        try:
            x, _ = batch
        except ValueError:
            x = batch
        if isinstance(x, Tensor):
            x = x.numpy()
        return tuple(m.predict_proba(x) for m in self.models)
