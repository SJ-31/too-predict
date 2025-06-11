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
        to_encode=("Sample_Type", "tumor_type", "primary_site"),
    ) -> None:
        self.X: torch.Tensor = torch.tensor(ut.xarray_if_sparse(adata), device=device)
        self.encoders: dict[str, sp.LabelEncoder] = {}
        self.labels: torch.Tensor = torch.zeros(
            self.X.shape[0], len(to_encode), dtype=int
        )
        self.n_classes: dict = {}
        self.label_cols: tuple = to_encode
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
