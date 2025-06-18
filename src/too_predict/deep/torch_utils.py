#!/usr/bin/env ipython

from collections.abc import Iterable, Sequence
from typing import Callable, Literal, override

import anndata as ad
import numpy as np
import pandas as pd
import sklearn.metrics as met
import sklearn.preprocessing as sp
import too_predict.utils as ut
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset


class AnnDataset(torch.utils.data.Dataset):
    """Custom dataset class for AnnData objects

    Parameters
    ----------
    device : torch device string to move expression data to
    to_encode : discrete labels to encode and output during dataset iteration

    Returns
    -------
    torch.DataSet object that is indexed to produce a sample expression tensor,
        and a vector of labels for the sample in the order of `to_encode`
    """

    def __init__(
        self,
        adata: ad.AnnData,
        device: str = "cpu",
        to_encode: tuple[str] | str = ("Sample_Type", "tumor_type", "primary_site"),
    ) -> None:
        self.X: torch.Tensor = torch.tensor(ut.xarray_if_sparse(adata), device=device)
        self.encoders: dict[str, sp.LabelEncoder] = {}
        self.labels: torch.Tensor = torch.zeros(
            self.X.shape[0], len(to_encode), dtype=int
        )
        self.n_classes: dict = {}
        self.label_cols: tuple = to_encode
        if isinstance(to_encode, str):
            to_encode = (to_encode,)
        for i, col in enumerate(to_encode):
            encoder = sp.LabelEncoder()
            labs = adata.obs[col]
            self.n_classes[col] = len(labs.unique())
            self.labels[:, i] = torch.as_tensor(encoder.fit_transform(labs))
            self.encoders[col] = encoder

    def decode(
        self,
        y: torch.Tensor,
        label_cols: Sequence | None = None,
        indices: Sequence | None = None,
    ) -> np.ndarray:
        if indices is None:
            indices = list(range(len(self.encoders)))
        vals = []
        if label_cols:
            for codes, label in zip(torch.unbind(y, dim=1), label_cols):
                vals.append(
                    self.encoders[label].inverse_transform(codes).reshape((-1, 1))
                )
        else:
            id2lab: dict = dict(zip(range(len(self.encoders)), self.encoders.keys()))
            for codes, index in zip(torch.unbind(y, dim=1), indices):
                decoded = (
                    self.encoders[id2lab[index]]
                    .inverse_transform(codes)
                    .reshape((-1, 1))
                )
                vals.append(decoded)
        return np.hstack(vals)

    @property
    def shape(self) -> tuple:
        return self.X.shape

    def __len__(self) -> int:
        return self.X.shape[0]

    @override
    def __getitem__(self, index):
        return self.X[index, :], self.labels[index, :]


def make_dataset(X: np.ndarray, y_true: np.ndarray) -> Dataset:
    return torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y_true))


def is_atomic(x: torch.Tensor | np.ndarray) -> bool:
    return len(x.shape) <= 1


class Module(nn.Module):
    def __init__(self, n_tasks: int = 1) -> None:
        super().__init__()
        self.n_tasks: int = n_tasks

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

    def get_optimizers(self) -> Optimizer:
        return optim.Adam(self.named_parameters())

    def criterion(self, y_pred, y_true):
        raise NotImplementedError


def n_uniques(x: torch.Tensor | np.ndarray | Sequence) -> int:
    one_d = is_atomic(x)
    if one_d and isinstance(x, torch.Tensor):
        return x.unique().size()[0]
    elif isinstance(x, torch.Tensor):
        return x.flatten().unique().size()[0]
    elif (one_d and isinstance(x, np.ndarray)) or isinstance(x, pd.Series):
        return np.unique(x).shape[0]
    return len(set(x))


def data_spec(
    X: Dataset | DataLoader | torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray | None = None,
) -> tuple:
    """Return a tuple of (n_features, n_classes) for the given dataset
    If multitask, the second element is a tuple of length n_tasks
    """

    def _for_dataset(data):
        x, y = data[:]
        if is_atomic(y) or y.shape[1] == 1:
            n_classes = n_uniques(y)
        else:
            n_classes = tuple([n_uniques(y[:, i]) for i in range(y.shape[1])])
        return x.shape[1], n_classes

    if isinstance(X, Dataset):
        return _for_dataset(X)
    elif isinstance(X, DataLoader):
        return _for_dataset(X.dataset)
    return X.shape[1], len(set(y))


class Trainer:
    """Wrapper class for training pytorch models

    Parameters
    ----------
    model : class inheriting torch_utils.Module, to take advantage of custom methods
    tol : tolerance to .. TODO:
    scheduler : custom scheduler
    score_metric : Built-in function to measure model performance at each iteration.
        Supports most scores in sklearn.metrics (pass without "_score")
    score_fn : If score_metric is not provided, a Callable with the signature:
        (y_true, y_pred) -> float
    output_names : Names of the output tasks in a multitask model. A column with
        name_<score_metric> will be added for each entry here

    Returns
    -------
    Pandas dataframe containing training metrics, namely `loss` and the
        performance measurement

    Notes
    -----
    This function modifies `model` inplace
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer | None = None,
        n_epochs: int = 1000,
        tol: float | None = None,
        scheduler: schedule.LRScheduler | None = None,
        score_metric: Literal[
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "mean_squared_error",
            "recall",
        ] = "accuracy",
        score_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        score_fn_name: str = "custom_metric",
        record_train_score: bool = True,
        record_test_score: bool = True,
        output_names: Sequence | None = None,
        at_batch_level: bool = True,
    ) -> None:
        self.evaluate: Callable
        self.n_epochs: int = n_epochs
        self.optimizer: Optimizer = (
            optimizer if optimizer is not None else model.get_optimizers()
        )
        self.at_batch_level: bool = at_batch_level
        self.scheduler: schedule.LRScheduler | None = None
        self.model: Module = model
        self.record_train_score: bool = record_train_score
        self.record_test_score: bool = record_test_score

        if score_fn is not None:
            self.train_score_key: str = f"train_{score_fn_name}"
            self.test_score_key: str = f"test_{score_fn_name}"
            self.evaluate = score_fn
        else:
            self.train_score_key = f"train_{score_metric}"
            self.test_score_key = f"test_{score_metric}"
            if score_metric == "accuracy":
                self.evaluate = met.accuracy_score
            elif score_metric == "balanced_accuracy":
                self.evaluate = met.balanced_accuracy_score
            elif score_metric == "f1":
                self.evaluate = met.f1_score
            elif score_metric == "precision":
                self.evaluate = met.precision_score
            elif score_metric == "mean_squared_error":
                self.evaluate = met.mean_squared_error
            elif score_metric == "recall":
                self.evaluate = met.recall_score

        self.metrics: dict = {"epoch": []}
        if self.at_batch_level:
            self.metrics["minibatch"] = []
            self.metrics["loss"] = []
        else:
            self.metrics["avg_loss"] = []
        if model.n_tasks > 1 and output_names is None:
            output_names = range(model.n_tasks)

        self.train_keys: list = []
        self.test_keys: list = []
        if model.n_tasks == 1:
            if record_train_score:
                self.metrics[self.train_score_key] = []
            if record_test_score:
                self.metrics[self.test_score_key] = []
        elif output_names is not None:
            for name in output_names:
                if record_train_score:
                    key = f"{name}_{self.train_score_key}"
                    self.train_keys.append(key)
                    self.metrics[key] = []
                if record_test_score:
                    key = f"{name}_{self.test_score_key}"
                    self.test_keys.append(key)
                    self.metrics[key] = []

    def _record(self, X, y, single_key: str, multi_key: list[str]):
        self.model.eval()
        y_pred = self.model.predict(X)
        if self.train_keys:
            for i, k in enumerate(multi_key):
                self.metrics[k].append(self.evaluate(y_pred[:, i], y[:, i]))
        else:
            score = self.evaluate(y_pred, y)
            self.metrics[single_key].append(score)
        self.model.train()

    def __call__(self, loader: DataLoader) -> pd.DataFrame:
        self.model.train()
        for i in range(self.n_epochs):
            losses = []
            for j, (X, y) in enumerate(loader):
                self.optimizer.zero_grad()
                out = self.model(X)
                loss: torch.Tensor = self.model.criterion(y_pred=out, y_true=y)
                loss.backward()

                if self.record_train_score and self.at_batch_level:
                    self._record(
                        X, y, multi_key=self.train_keys, single_key=self.train_score_key
                    )
                if self.record_test_score and self.at_batch_level:
                    self._record()  # TODO: how to do this

                if self.at_batch_level:
                    self.metrics["epoch"].append(i)
                    self.metrics["minibatch"].append(j)
                    self.metrics["loss"].append(loss.detach().numpy())
                else:
                    losses.append(loss)
                self.optimizer.step()

            if not self.at_batch_level:
                with torch.no_grad():
                    self.metrics["epoch"].append(i)
                    self.metrics["avg_loss"].append(np.mean(losses))
                x_train, y_train = loader.dataset[:]
                self._record(
                    x_train,
                    y_train,
                    multi_key=self.train_keys,
                    single_key=self.train_score_key,
                )

        self.model.eval()
        return pd.DataFrame(self.metrics)


def iter_cols(x: Tensor | np.ndarray) -> Iterable:
    if isinstance(x, Tensor):
        to_iter = torch.unbind(x, dim=1)
    else:
        to_iter = [x[:, i] for i in range(x.shape[1])]
    return to_iter


# def optimize():
