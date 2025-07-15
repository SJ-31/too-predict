#!/usr/bin/env ipython

from collections.abc import Sequence
from typing import override

import too_predict.deep.torch_utils as d_ut
import torch
import torch.nn as nn
from too_predict.deep.evaluation import multitask_cross_entropy_loss
from too_predict.deep.logistic import logistic_hook
from too_predict.utils import if_none
from torch import Tensor

"""
References
[1] Disyak, M. (2021). A multi-task machine learning pipeline for the classification and analysis of cancers from gene expression data (T). University of British Columbia. Retrieved from https://open.library.ubc.ca/collections/ubctheses/24/items/1.0395883
"""


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
            in_features, n_classes_per_task, task_weights, l1_pars, l2_pars, **kwargs
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
    def criterion(self, y_pred, y_true):
        total_loss: torch.Tensor = 0
        if self.n_tasks > 1:
            total_loss += multitask_cross_entropy_loss(
                y_pred, y_true, weights=self.task_weights
            )
        else:
            total_loss += nn.functional.cross_entropy(y_pred, y_true)
        total_loss += self.l2() + self.l1()
        return total_loss
