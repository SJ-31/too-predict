#!/usr/bin/env ipython

from collections.abc import Callable

import numpy as np
import pandas as pd
import sklearn.datasets as datasets
import sklearn.linear_model as sl
import sklearn.metrics as met
import sklearn.model_selection as ms
import sklearn.preprocessing as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

SEED = 21
torch.manual_seed(SEED)
RNG = np.random.RandomState(SEED)
torch.set_default_dtype(torch.float64)
DATA = list(datasets.load_wine(return_X_y=True))
DATA[0] = sp.StandardScaler().fit_transform(DATA[0])


def make_dataset(X: np.ndarray, y_true: np.ndarray) -> Dataset:
    return torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y_true))


class Module(nn.Module):
    def _predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        if isinstance(proba, tuple):
            return np.hstack([p.argmax(axis=1).reshape(-1, 1) for p in proba])
        return proba.argmax(axis=1)

    def predict(self, X: Tensor | np.ndarray | DataLoader | Dataset) -> np.ndarray:
        if isinstance(X, DataLoader):
            prediction = []
            for x, _ in X:
                prediction.append(self._predict(x))
            return np.vstack(prediction)
        if isinstance(X, Dataset):
            X = X[:]
        return self._predict(X)

    def predict_proba(self, X) -> np.ndarray | tuple:
        X = torch.tensor(X) if isinstance(X, np.ndarray) else X
        proba = self(X)
        if isinstance(proba, tuple):
            return tuple(p.detach().numpy() for p in proba)
        return proba.detach().numpy()


def train_model(
    model: Module,
    loader: DataLoader,
    criterion: Callable,
    optimizer: Optimizer | None = None,
    needs_model: bool = False,
    needs_closure: bool = False,
    n_epochs: int = 1000,
) -> pd.DataFrame:
    metrics: dict = {"loss": [], "epoch": [], "minibatch": []}

    model.train()
    record: bool = "record_metrics" in dir(model)
    for i in range(n_epochs):
        for j, (X, y) in enumerate(loader):

            def closure():
                optimizer.zero_grad()
                y_pred = model(X)
                loss: torch.Tensor
                if not needs_model:
                    loss = criterion(y_pred, y)
                else:
                    loss = criterion(model, y_pred, y)
                loss.backward()
                if record:
                    model.record_metrics(metrics)
                metrics["epoch"].append(i)
                metrics["minibatch"].append(j)
                metrics["loss"].append(loss.detach().numpy())
                return loss

            if not needs_closure:
                _ = closure()
                optimizer.step()
            else:
                optimizer.step(closure)

    model.eval()
    return pd.DataFrame(metrics)


class DummyLR(Module):
    def __init__(self, n_classes_per_task, l2=1) -> None:
        super().__init__()
        self.linear: nn.LazyLinear = nn.LazyLinear(out_features=n_classes_per_task)
        self.l2: float = l2
        self.softmax: nn.Softmax = nn.Softmax(dim=1)

    def forward(self, X):
        return self.softmax(self.linear(X))

    @staticmethod
    def criterion(model, X, y):
        cel = nn.functional.cross_entropy(input=X, target=y)
        l2 = cel + model.l2 * torch.sum(model.linear.weight**2)
        # [2025-06-18 Wed]
        # l2 is used by default in skLearn's LogisticRegression with C = 1
        # But this lowers acuracy dramatically
        return l2


def torch_model(l2, loader, x_test, y_test, **kwargs):
    model = DummyLR(
        n_classes_per_task=len(np.unique(DATA[1])),
        l2=l2,
    )
    opt = optim.Adam(model.named_parameters(), **kwargs)
    _ = train_model(
        model=model,
        loader=loader,
        optimizer=opt,
        criterion=DummyLR.criterion,
        needs_model=True,
        n_epochs=2000,
        needs_closure=True,
    )
    pred = model.predict(x_test)
    acc = met.accuracy_score(y_test, pred)
    return acc


def compare_models():
    X_train, X_test, y_train, y_test = ms.train_test_split(
        DATA[0], DATA[1], random_state=RNG
    )
    dset = make_dataset(X_train, y_train)
    loader = DataLoader(dset, batch_size=len(dset))
    acc_custom_l2 = torch_model(1, loader, X_test, y_test)
    acc_no_l2 = torch_model(0, loader, X_test, y_test)
    acc_adam_l2 = torch_model(0, loader, X_test, y_test, weight_decay=1)
    acc_adam_l2_decoupled = torch_model(
        0,
        loader,
        X_test,
        y_test,
        weight_decay=1,
        decoupled_weight_decay=True,
    )

    sgd = sl.SGDClassifier(loss="log_loss")
    sgd.fit(X_train, y_train)
    sgd_acc = met.accuracy_score(y_test, sgd.predict(X_test))

    lbfgs = sl.LogisticRegression()
    lbfgs.fit(X_train, y_train)
    lbfgs_acc = met.accuracy_score(y_test, lbfgs.predict(X_test))

    return (
        sgd_acc,
        lbfgs_acc,
        acc_custom_l2,
        acc_no_l2,
        acc_adam_l2,
        acc_adam_l2_decoupled,
    )


n_iter = 10

scores = np.array([np.array(compare_models()) for _ in range(n_iter)])

comparison = pd.DataFrame(
    scores,
    columns=[
        "sklearn-SGD",
        "sklearn-LBFGS",
        "torch-custom_l2",
        "torch-no_l2",
        "torch-Adam_l2",
        "torch-Adam_l2_decoupled",
    ],
)

print(comparison)
