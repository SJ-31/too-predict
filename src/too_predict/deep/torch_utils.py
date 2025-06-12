#!/usr/bin/env ipython

from collections.abc import Sequence
from typing import Callable, override

import anndata as ad
import numpy as np
import pandas as pd
import sklearn.preprocessing as sp
import too_predict.utils as ut
import torch
import torch.nn as nn
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
    def predict(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        X = torch.tensor(X) if isinstance(X, np.ndarray) else X
        proba = self(X)
        if isinstance(proba, tuple):
            return np.hstack([p.argmax(dim=1).numpy().reshape(-1, 1) for p in proba])
        return proba.argmax(dim=1).numpy()

    def get_optimizers(self) -> Optimizer:
        return optim.Adam(self.named_parameters())

    @staticmethod
    def criterion(model, y_pred, y_true):
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


def train_model(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: Callable,
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
                pred = model(X)
                loss: torch.Tensor
                if not needs_model:
                    loss = criterion(pred, y)
                else:
                    loss = criterion(model, pred, y)
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
