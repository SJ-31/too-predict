#!/usr/bin/env ipython

from collections.abc import Callable

import numpy as np
import pandas as pd
import sklearn.datasets as datasets
import sklearn.linear_model as sl
import sklearn.metrics as met
import sklearn.model_selection as ms
import sklearn.preprocessing as sp
import too_predict.deep.torch_utils as d_ut
import torch
import torch.nn as nn
import torch.optim as optim
from too_predict.deep.logistic import DummyLR
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


def torch_model(l2, loader, x_test, y_test, **kwargs):
    model = DummyLR(
        n_classes_per_task=len(np.unique(DATA[1])),
        l2=l2,
    )
    opt = optim.Adam(model.named_parameters(), **kwargs)
    trainer = d_ut.Trainer(
        model=model,
        optimizer=opt,
        n_epochs=2000,
        at_batch_level=True,
        record_test_score=False,
    )
    metrics = trainer(loader)
    print(metrics)
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
