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


class Module(nn.Module):
    optimizer: Optimizer | None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer = None

    def prefit(self, data: Dataset) -> None:
        """Things the model might need to do with access to do the entire dataset"""
        print("No prefit specified")
        return

    def record_metrics(self, record: dict, **kwargs) -> None:
        return

    def get_optimizers(self, **kwargs) -> Optimizer:
        raise NotImplementedError()

    def objective(self, prediction: torch.Tensor, y: torch.Tensor):
        """Objective function, computes loss to minimize

        Parameters
        ----------
        predition : the result of Module.__call__()
        y : true values

        Returns
        -------
        A tensor capable of autograd
        """
        raise NotImplementedError()


def train_model(
    model: Module, loader: DataLoader, n_epochs: int = 1000
) -> pd.DataFrame:
    metrics: dict = {"loss": [], "epoch": [], "minibatch": []}
    optimizer: Optimizer = model.get_optimizers()
    model.train()
    model.prefit(loader.dataset)
    for i in range(n_epochs):
        for j, (X, y) in enumerate(loader):
            pred = model(X)
            loss: torch.Tensor = model.objective(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.record_metrics(metrics)
            metrics["epoch"].append(i)
            metrics["minibatch"].append(j)
            metrics["loss"].append(loss)
    model.eval()
    return pd.DataFrame(metrics)
