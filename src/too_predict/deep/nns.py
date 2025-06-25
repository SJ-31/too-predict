#!/usr/bin/env ipython

from collections.abc import Sequence
from typing import override

import too_predict.deep.torch_utils as d_ut
import torch
import torch.nn as nn
from too_predict.deep.evaluation import multitask_cross_entropy_loss
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
        reduce_features: bool = True,
        task_weights: Tensor | Sequence | None = None,
    ) -> None:
        super().__init__(in_features, n_classes_per_task, task_weights)
        n_units = 2000
        if not reduce_features:
            n_units = in_features
        self.layers: nn.ModuleList = nn.ModuleList()
        self.relu: nn.ReLU = nn.ReLU()
        self.dropout: nn.Dropout = nn.Dropout()
        self.sum_to: list = []
        for n_classes in n_classes_per_task:
            self.layers.append(nn.LazyLinear(n_units))
            self.sum_to.append(torch.ones((n_units, n_classes)))

    @override
    def reset_parameters(self):
        for m in self.layers:
            d_ut.reset_sequential(m)

    @override
    def forward(self, X):
        modules, sums = iter(self.layers), iter(self.sum_to)
        result = []
        hidden: torch.Tensor = self.dropout(self.relu(next(modules)(X)))
        s1: torch.Tensor = next(sums)
        result.append(torch.matmul(hidden, s1))
        for sum, m in zip(sums, modules):
            hidden = self.dropout(self.relu(m(hidden)))
            result.append(torch.matmul(hidden, sum))
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
        return total_loss
