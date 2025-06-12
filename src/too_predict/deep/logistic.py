#!/usr/bin/env ipython
import itertools
from typing import override

import numpy as np
import sklearn.linear_model as sl
import skorch
import torch
from torch import Tensor, nn


def logistic_hook(_m, _input, output) -> Tensor:
    return nn.functional.softmax(output, dim=1)


class MtcLr(nn.Module):
    def __init__(
        self,
        n_features: int,
        task_spec: list[int],
        lmbda: float = 5.0,
        initial_fit: dict | None = None,
        sk_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.n_tasks: int = len(task_spec)
        self.lmbda: float = lmbda
        self.lrs: nn.ModuleDict = nn.ModuleDict()
        self.task_label_map: dict = {}
        self.sk_kwargs: dict = sk_kwargs if sk_kwargs else {}
        for i, n_classes in enumerate(task_spec):
            # Initialize LR model for each task
            lr = nn.Linear(n_features, n_classes)
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

    @override
    def forward(self, X) -> list[Tensor]:
        results = []
        for i in range(self.n_tasks):
            results.append(self.lrs[str(i)](X))
        return results


class MtcLrSkorch(skorch.NeuralNetClassifier):
    @override
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        print("custom loss used")
        total_loss = 0.0
        if len(y_pred.shape) > 1:
            for task_pred, task_y in zip(
                y_pred, torch.unbind(y_true, dim=1)
            ):  # Gives y_hat = softmax(Xw + b)
                # tensor of shape n_samples, n_classes
                total_loss += nn.functional.cross_entropy(task_pred, task_y)
                # task_sum = task_pred.sum(dim=1)
                # for class_idx, class_vec in enumerate(torch.unbind(task_pred, dim=1)):
                #     mask: torch.Tensor = y == class_idx
                #     total_loss += (mask * torch.log(class_vec / task_sum)).sum()
            # Get loss on tasks separately
        else:
            total_loss += nn.functional.cross_entropy(y_pred[0], y_true)

        # Regularization by distances between parameters
        if len(self.module_.lrs) > 1:
            tuples = itertools.product(self.module_.lrs.keys())
            for tup in tuples:
                for combo in itertools.combinations(tup, 2):
                    first = self.module_.lrs[str(combo[0])].weight
                    sec = self.module_.lrs[str(combo[1])].weight
                    summed_dist = torch.cdist(first, sec).flatten().sum()
                    total_loss -= summed_dist

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

    @override
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
        super().__init__(*args, **kwargs)
        self.lrs: nn.ModuleDict = nn.ModuleDict()
        self.theta: Tensor = nn.Parameter(torch.normal(0, 1, (in_features,)))
        self.n_tasks: Tensor = len(n_classes_per_task)
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

    @staticmethod
    @override
    def criterion(model: MultiLevel, y_pred: Tensor, y_true: Tensor) -> Tensor:
        total_loss: Tensor = torch.tensor(0)
        n_samples = y_pred.shape[0]
        if model.n_tasks > 1:
            total_loss += 1 / 2 * multitask_cross_entropy_loss(y_pred, y_true)
        else:
            total_loss += nn.functional.cross_entropy(y_pred, y_true)

        # Regularization
        reg_theta = model.lmbda_1 * torch.sum(torch.abs(model.theta))
        reg_gamma = 0
        for lr in model.lrs.values():
            reg_gamma += torch.sum(torch.abs(lr.gamma), ord=1)
        reg_gamma *= model.lmbda_2
        total_loss = total_loss / n_samples + reg_theta + reg_gamma

        return total_loss
