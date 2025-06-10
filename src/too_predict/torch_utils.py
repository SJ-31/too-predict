#!/usr/bin/env ipython

from typing import Sequence, override

import anndata as ad
import numpy as np
import sklearn.preprocessing as sp
import torch
import torch.nn as nn
from anndata.experimental import AnnLoader
from torch.utils.data import DataLoader

import too_predict.multitask as multi
import too_predict.utils as ut


def get_annloader(
    adata: ad.AnnData,
    kept_obs=("Project_ID", "Case_ID", "Sample_ID"),
    batch_size: int = 128,
    shuffle: bool = True,
    **kwargs,
) -> AnnLoader:
    """Create custom torch data loader for `adata`

    Parameters
    ----------
    kwargs : additional arguments for PyTorch DataLoader or AnnCollection
    kept_obs : Any observations to keep in adata.obs

    Return
    ------
    An annloader object which you can use as a PyTorch dataloader
    Index the desired attributes in the batch as usual, e.g. adata.obs['tumor_type']
    """
    to_encode = ["Sample_Type", "tumor_type", "primary_site"]
    adata.obs = adata.obs.loc[:, to_encode + list(kept_obs)]
    label_encoders = {s: sp.LabelEncoder() for s in to_encode}
    for k, v in label_encoders.items():
        v.fit(adata.obs[k])
    use_cuda = torch.cuda.is_available()
    converters = {"obs": {k: v.transform for k, v in label_encoders.items()}}
    return ad.experimental.AnnLoader(
        adata,
        batch_size=batch_size,
        shuffle=shuffle,
        convert=converters,
        use_cuda=use_cuda,
        **kwargs,
    )


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

    # def inverse_transform(self, labels: torch.Tensor) -> :
