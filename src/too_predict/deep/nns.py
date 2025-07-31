#!/usr/bin/env ipython

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, override

import too_predict.deep.torch_utils as d_ut
import torch
import torch.nn as nn
from too_predict.deep.evaluation import multitask_cross_entropy_loss
from too_predict.deep.logistic import logistic_hook
from torch import Tensor

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
        n_hidden: int | None = None,
        n_shared_layers: int = 1,
        n_layers_per_task: Sequence[int] | None = None,
        task_weights: Tensor | Sequence | None = None,
        dropout_p: float = 0.2,
        batch_norm: Literal["before", "after", "none"] = "none",
        batch_norm_kwargs: dict | None = None,
        l1_pars: dict | None = None,
        l2_pars: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            in_features=in_features,
            n_classes_per_task=n_classes_per_task,
            task_weights=task_weights,
            l1_pars=l1_pars,
            l2_pars=l2_pars,
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
            out.register_forward_hook(logistic_hook)
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
        if self._n_tasks > 1:
            total_loss += multitask_cross_entropy_loss(
                y_pred, y_true, weights=self._task_weights, model=self, prefix=context
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
        task_weights: Tensor | Sequence | None = None,
        dropout_p: float = 0.2,
        l1_pars: dict | None = None,
        l2_pars: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            in_features=in_features,
            n_classes_per_task=n_classes_per_task,
            task_weights=task_weights,
            l1_pars=l1_pars,
            l2_pars=l2_pars,
            **kwargs,
        )
        if n_hidden is None:
            n_hidden = in_features
        self.hlayers: nn.ModuleList = nn.ModuleList()
        self.olayers: nn.ModuleList = nn.ModuleList()
        self.acti: nn.Module = nn.Tanh()
        self.softmax: nn.Softmax = nn.Softmax()
        self.dropout: nn.Dropout = nn.Dropout(p=dropout_p)
        for n_classes in n_classes_per_task:
            self.hlayers.append(nn.LazyLinear(n_hidden))
            out = nn.LazyLinear(n_classes)
            out.register_forward_hook(logistic_hook)
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
        if self._n_tasks > 1:
            total_loss += multitask_cross_entropy_loss(
                y_pred, y_true, weights=self._task_weights, model=self, prefix=context
            )
        else:
            total_loss += nn.functional.cross_entropy(y_pred, y_true)
        total_loss += self.l2() + self.l1()
        return total_loss
