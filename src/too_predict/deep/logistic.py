#!/usr/bin/env ipython
from __future__ import annotations

import itertools
from typing import Sequence, override

import numpy as np
import sklearn.linear_model as sl
import too_predict.deep.torch_utils as d_ut
import torch
from torch import Tensor, nn

"""
References
[1] X. Gu, F. -L. Chung, H. Ishibuchi and S. Wang, "Multitask Coupled Logistic Regression and its Fast Implementation for Large Multitask Datasets," in IEEE Transactions on Cybernetics, vol. 45, no. 9, pp. 1953-1966, Sept. 2015, doi: 10.1109/TCYB.2014.2362771.
[2] Aurélie C. Lozano and Grzegorz Swirszcz. 2012. Multi-level lasso for sparse multi-task regression. In Proceedings of the 29th International Coference on International Conference on Machine Learning (ICML'12). Omnipress, Madison, WI, USA, 595–602.

"""
# * Utils


def logistic_hook(_m, _input, output) -> Tensor:
    return nn.functional.softmax(output, dim=1)


class DummyLR(d_ut.Module):
    def __init__(self, n_classes_per_task, l2=1) -> None:
        super().__init__()
        self.linear: nn.LazyLinear = nn.LazyLinear(out_features=n_classes_per_task)
        self.l2: float = l2
        self.softmax: nn.Softmax = nn.Softmax(dim=1)

    @override
    def forward(self, X):
        return self.softmax(self.linear(X))

    @override
    def criterion(self, y_pred, y_true):
        cel = nn.functional.cross_entropy(input=y_pred, target=y_true)
        l2 = cel + self.l2 * torch.sum(self.linear.weight**2)
        # TODO: this naive implementation hinders performance...
        return l2


def multitask_cross_entropy_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    total_loss: Tensor = 0
    for task_pred, task_y in zip(
        y_pred, torch.unbind(y_true, dim=1)
    ):  # Gives y_hat = softmax(Xw + b)
        # tensor of shape n_samples, n_classes
        total_loss += nn.functional.cross_entropy(task_pred, task_y)
        # Get loss on tasks separately
    return total_loss


# * Implementation of [1]


class MtcLr(d_ut.Module):
    """Implementation of multitask logistic regression by [1]

    Parameters
    ----------
    l2 : l2 regularization parameter for multiclass cross-entropy loss
    lmbda : parameter controlling influence of weight regularization between tasks
    intial_fit : dictionary mapping task indices to sklearn LogisticRegression
       models, fitted independently on their corresponding task
    n_classes_per_task : Sequence where the i-th entry is the number of classes in the
        i-th learning task
    """

    def __init__(
        self,
        n_features: int,
        n_classes_per_task: list[int],
        lmbda: float = 5.0,
        l2: float = 1,
        initial_fit: dict | None = None,
    ) -> None:
        super().__init__(n_tasks=len(n_classes_per_task))
        self.lmbda: float = lmbda
        self.l2: float = l2
        self.lrs: nn.ModuleDict = nn.ModuleDict()
        self.task_label_map: dict = {}
        for i, n_classes in enumerate(n_classes_per_task):
            # Initialize LR model for each task
            lr = nn.LazyLinear(n_classes)
            lr.register_forward_hook(logistic_hook)
            self.lrs[str(i)] = lr
        if initial_fit is not None:
            self.init_lr_weights(initial_fit)

    @staticmethod
    def get_initial_fit(
        X: np.ndarray, ys: np.ndarray, **kwargs
    ) -> dict[str, sl.LogisticRegression]:
        result = {}
        for i in range(ys.shape[1]):
            y_true = ys[:, i]
            lr = sl.LogisticRegression(**kwargs)
            lr.fit(X, y_true)
            result[str(i)] = lr
        return result

    def init_lr_weights(self, initial_fit: dict) -> None:
        """Initialize weights in model's lr objects as an independent fit on the
        different tasks
        """
        for key, model in initial_fit.items():
            self.lrs[key].weight = torch.nn.Parameter(
                torch.tensor(model.coef_, dtype=torch.float64)
            )
            self.lrs[key].bias = torch.nn.Parameter(
                torch.tensor(model.intercept_, dtype=torch.float64)
            )

    def forward(self, X) -> tuple[Tensor]:
        results = []
        for i in range(self.n_tasks):
            results.append(self.lrs[str(i)](X))
        return tuple(results)

    @override
    def criterion(self, y_pred, y_true):
        total_loss = 0.0
        if len(y_pred) > 1:
            total_loss += multitask_cross_entropy_loss(y_pred, y_true)
        else:
            total_loss += nn.functional.cross_entropy(y_pred[0], y_true) + self._l2(0)

        # Regularization by distances between parameters
        if len(self.lrs) > 1:
            tuples = itertools.product(self.lrs.keys())
            distance_reg: float = 0
            for tup in tuples:
                for combo in itertools.combinations(tup, 2):
                    first = self.lrs[str(combo[0])].weight
                    sec = self.lrs[str(combo[1])].weight
                    summed_dist = torch.cdist(first, sec).flatten().sum()
                    distance_reg += summed_dist
            total_loss -= self.lmbda * distance_reg

        return total_loss


# * Implementation of [2]


class DecomposedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        theta: Tensor | None = None,
        bias: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if theta is None:
            self.theta: Tensor = nn.Parameter()
        else:
            self.theta = theta  # shared weight
        self.gamma: Tensor = nn.Parameter(
            torch.normal(0, 1, size=(out_features, in_features))
        )  # Shape of n_classes x n_features
        # TODO: better way to initialize weights???
        if bias:
            self.bias: Tensor | float = nn.Parameter(torch.zeros(1))
        else:
            self.bias = 0

    def forward(self, X):
        beta: Tensor = self.gamma.mul(self.theta)
        # y = x * torch.transpose(beta, 0, 1) + self.bias
        return nn.functional.linear(X, beta, self.bias)


class MultiLevel(d_ut.Module):
    def __init__(
        self,
        in_features: int,
        n_classes_per_task: Sequence[int],
        lmbda_1: float = 1.0,
        lmbda_2: float = 1.0,
        bias: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(n_tasks=len(n_classes_per_task), *args, **kwargs)
        self.lrs: nn.ModuleDict = nn.ModuleDict()
        self.theta: Tensor = nn.Parameter(torch.normal(0, 1, (in_features,)))
        self.task_label_map: dict = {}
        for i, n_classes in enumerate(n_classes_per_task):
            self.task_label_map[i] = list(range(n_classes))
            self.lrs[str(i)] = DecomposedLinear(
                in_features=in_features,
                out_features=n_classes,
                theta=self.theta,
                bias=bias,
            )
            self.lrs[str(i)].register_forward_hook(logistic_hook)
        self.lmbda_1: float = lmbda_1
        self.lmbda_2: float = lmbda_2

    @override
    def forward(self, X) -> tuple[Tensor]:
        results = []
        for i in range(self.n_tasks):
            results.append(self.lrs[str(i)](X))
        return tuple(results)

    @override
    def criterion(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        total_loss: Tensor = 0
        n_samples: int = y_true.shape[0]
        if self.n_tasks > 1:
            total_loss += 1 / 2 * multitask_cross_entropy_loss(y_pred, y_true)
        else:
            total_loss += nn.functional.cross_entropy(y_pred, y_true)

        # Regularization
        reg_theta = self.lmbda_1 * torch.sum(torch.abs(self.theta))
        reg_gamma = 0
        for lr in self.lrs.values():
            reg_gamma += torch.sum(torch.abs(lr.gamma))
        reg_gamma *= self.lmbda_2
        total_loss = total_loss / n_samples + reg_theta + reg_gamma

        return total_loss


# class MultiLevelSkorch(skorch.NeuralNetClassifier):
#     @classmethod
#     def new(cls, **kwargs):
#         return cls(module=MultiLevel, **kwargs)

#     @override
#     def get_loss(self, y_pred, y_true):
#         return MultiLevel.criterion(self, y_pred=y_pred, y_true=y_true)
