#!/usr/bin/env ipython
import itertools
from typing import override

import sklearn.linear_model as sl
import torch
from too_predict.deep.torch_utils import AnnDataset, Module
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def logistic_hook(_m, _input, output) -> Tensor:
    return nn.functional.softmax(output, dim=1)


class MtcLr(Module):
    def __init__(
        self,
        task_spec: list[tuple[int, int]],
        C: float,
        lmbda: float = 5.0,
        sk_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.n_tasks: int = len(task_spec)
        self.lmbda: float = lmbda
        self.lrs: nn.ModuleDict = nn.ModuleDict()
        self.sk_kwargs: dict = sk_kwargs if sk_kwargs else {}
        for i, (n_features, n_classes) in enumerate(task_spec):
            # Initialize LR model for each task
            lr = nn.Linear(n_features, n_classes)
            lr.register_forward_hook(logistic_hook)
            self.lrs[str(i)] = lr

    @override
    def fit(
        self, loader: DataLoader, optimizer: Optimizer, max_epochs: int = 1000
    ) -> None:
        init_lr_weights(self, loader, **self.sk_kwargs)
        super().fit(loader, optimizer, max_epochs)

    @override
    def forward(self, X) -> list[Tensor]:
        results = []
        for i in range(self.n_tasks):
            results[i] = self.lrs[str(i)](X)
        return results

    @override
    def training_step(self, X, y) -> torch.Tensor:
        total_loss: Tensor = torch.tensor(0, requires_grad=True)
        for task_pred in self.forward(X):  # Gives y_hat = softmax(Xw + b)
            # tensor of shape n_samples, n_classes
            task_sum = task_pred.sum(dim=0)
            print(task_sum.shape)
            for _class_idx, class_vec in enumerate(torch.unbind(task_pred)):
                total_loss += -torch.log(class_vec / task_sum)
        # Get loss on tasks separately

        # Regularization by distances between parameters
        tuples = itertools.product(self.lrs.keys())
        for tup in tuples:
            for combo in itertools.combinations(tup, 2):
                first = self.lrs[str(combo[0])].weight
                sec = self.lrs[str(combo[1])].weight
                total_loss -= torch.cdist(first, sec).flatten().su

        return total_loss


def init_lr_weights(model: MtcLr, dataset: AnnDataset, **kwargs) -> None:
    """Initialize weights in model's lr objects as an independent fit on the
    different tasks
    """
    X = dataset.X
    y = dataset.labels
    for i, y_true in enumerate(torch.unbind(y, dim=1)):
        lr = sl.LogisticRegression(**kwargs)
        lr.fit(X, y_true)
        model.lrs[str(i)].weight = torch.nn.Parameter(torch.tensor(lr.coef_))
        model.lrs[str(i)].bias = torch.nn.Parameter(torch.tensor(lr.intercept_))
