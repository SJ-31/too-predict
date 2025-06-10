#!/usr/bin/env ipython

from collections.abc import Sequence
from typing import override

import anndata as ad
import numpy as np
import sklearn.preprocessing as sp
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import too_predict.utils as ut


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
        self.label_cols: tuple = to_encode
        for i, col in enumerate(to_encode):
            encoder = sp.LabelEncoder()
            self.labels[:, i] = torch.as_tensor(encoder.fit_transform(adata.obs[col]))
            self.encoders[col] = encoder

    def decode(
        self,
        y: torch.Tensor,
        label_cols: Sequence | None = None,
        indices: Sequence | None = None,
    ) -> np.ndarray:
        print(y)
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

    def fit(
        self,
        loader: DataLoader,
        optimizer: Optimizer,
        max_epochs: int = 1000,
    ) -> None:
        self.optimizer = optimizer
        self.train()
        for _ in range(max_epochs):
            self._fit_epoch(loader)
        self.eval()

    def _fit_epoch(self, loader: DataLoader):
        for X, y in loader:
            self.optimizer.zero_grad()
            loss: torch.Tensor = self.training_step(X, y)
            loss.backward()
            self.optimizer.step()

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
