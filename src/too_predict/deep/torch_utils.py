#!/usr/bin/env ipython

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Callable, Literal, override

import anndata as ad
import lightning as L
import numpy as np
import pandas as pd
import sklearn.preprocessing as sp
import too_predict.utils as ut
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
from lightning.pytorch.utilities.types import OptimizerConfig
from too_predict.utils import if_none
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import Accuracy

# * Utility functions


def timed(fn, verbose: bool = True):
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        t_taken = start.elapsed_time(end) / 1000
        if verbose:
            print(f"Time taken: {t_taken}")
        return result, t_taken
    start = time.time()
    result = fn()
    end = time.time()
    t_taken = f"{end - start:.4f}"
    if verbose:
        print(f"Time taken: {t_taken}")
    return result, t_taken


def tensor_cols_to_float(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {k: float for k in df.select_dtypes(torch.Tensor).columns}
    return df.astype(mapping)


def reset_sequential(mod: nn.Module) -> None:
    """reset_sequential.

    Parameters
    ----------
    mod : nn.Module
        mod

    Returns
    -------
    None

    """

    def reset(m):
        """reset.

        Parameters
        ----------
        m :
            m
        """
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Linear)
            or isinstance(m, nn.Module)
        ):
            m.reset_parameters()

    mod.apply(reset)


def iter_cols(x: Tensor | np.ndarray) -> Iterable:
    """Iterate over columnes of x

    Parameters
    ----------
    x : Tensor | np.ndarray
        x

    Returns
    -------
    Iterable

    """
    if isinstance(x, Tensor):
        to_iter = torch.unbind(x, dim=1)
    else:
        to_iter = [x[:, i] for i in range(x.shape[1])]
    return to_iter


def n_uniques(x: torch.Tensor | np.ndarray | Sequence) -> int:
    """Count the number of unique elements in x

    Parameters
    ----------
    x : torch.Tensor | np.ndarray | Sequence
        x

    Returns
    -------
    int

    """
    one_d = is_atomic(x)
    if one_d and isinstance(x, torch.Tensor):
        return x.unique().size()[0]
    elif isinstance(x, torch.Tensor):
        return x.flatten().unique().size()[0]
    elif (one_d and isinstance(x, np.ndarray)) or isinstance(x, pd.Series):
        return np.unique(x).shape[0]
    return len(set(x))


def data_spec(
    X: Dataset | DataLoader | torch.Tensor | np.ndarray | ad.AnnData,
    y: torch.Tensor | np.ndarray | None | pd.DataFrame = None,
) -> tuple[int, tuple[int, ...]]:
    """Return a tuple of (n_features, n_classes) for the given dataset
    If multitask, the second element is a tuple of length n_tasks
    """

    def _for_dataset(data):
        """_for_dataset.

        Parameters
        ----------
        data :
            data
        """
        x, y = data[:]
        if is_atomic(y) or y.shape[1] == 1:
            n_classes = n_uniques(y)
        else:
            n_classes: tuple[int, int] = tuple(
                [n_uniques(y[:, i]) for i in range(y.shape[1])]
            )
        return x.shape[1], n_classes

    if (
        (isinstance(y, tuple) or isinstance(y, list))
        and isinstance(next(iter(y)), str)
        and isinstance(X, ad.AnnData)
    ):
        y = X.obs.loc[:, y]
    if isinstance(X, ad.AnnData) and not X.isbacked:
        X = ut.xarray_if_sparse(X)
    elif isinstance(X, ad.AnnData):
        X = X.X

    if isinstance(X, Dataset):
        return _for_dataset(X)
    elif isinstance(X, DataLoader):
        return _for_dataset(X.dataset)
    elif isinstance(y, pd.DataFrame):
        return X.shape[1], tuple([len(y[s].unique()) for s in y])
    elif isinstance(y, np.ndarray) and len(y.shape) > 1:
        return X.shape[1], tuple([n_uniques(y[:, i]) for i in range(y.shape[1])])
    return X.shape[1], tuple(len(set(y)))


# * Datasets
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
        """__init__.

        Parameters
        ----------
        adata : ad.AnnData
            adata
        device : str
            device
        to_encode : tuple[str] | str
            to_encode

        Returns
        -------
        None

        """
        self.isbacked: bool = adata.isbacked
        self.file: Path = adata.file
        self.device: str = device
        if not adata.isbacked:
            self.X: torch.Tensor = torch.tensor(
                ut.xarray_if_sparse(adata), device=self.device
            )
        else:
            self.X = adata.X
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
        """decode.

        Parameters
        ----------
        y : torch.Tensor
            y
        label_cols : Sequence | None
            label_cols
        indices : Sequence | None
            indices

        Returns
        -------
        np.ndarray

        """
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
        val = self.X[index, :], self.labels[index, :]
        if not self.isbacked:
            return val
        arr = val[0].toarray().astype(np.float32)
        if arr.shape[0] == 1:
            arr = arr.flatten()
        as_tensor = torch.from_numpy(arr).to(device=self.device)
        return as_tensor, val[1]


def make_dataset(X: np.ndarray, y_true: np.ndarray) -> Dataset:
    """make_dataset.

    Parameters
    ----------
    X : np.ndarray
        X
    y_true : np.ndarray
        y_true

    Returns
    -------
    Dataset

    """
    return torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y_true))


def is_atomic(x: torch.Tensor | np.ndarray) -> bool:
    """is_atomic.

    Parameters
    ----------
    x : torch.Tensor | np.ndarray
        x

    Returns
    -------
    bool

    """
    return len(x.shape) <= 1


# * Custom module


def linear_reset_parameters(weight: Tensor, bias: Tensor | None = None) -> None:
    """Reset parameters for a linear model

    Parameters
    ----------
    weight : Tensor storing weights (must be an attribute of a instantiated module)
    bias : Tensor storing bias

    Notes
    -----
    Taken directly from pytorch repo
    """
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(bias, -bound, bound)


class MultiModule(L.LightningModule):
    """MultiModule."""

    def __init__(
        self,
        in_features: int,
        n_classes_per_task: list[int],
        record_metrics: bool = True,
        task_names: Sequence[str] | None = None,
        task_weights: Tensor | Sequence | None = None,
        l1_pars: dict | None = None,
        l2_pars: dict | None = None,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
        scheduler_config: dict | None = None,
        cache: str | None | Sequence = None,
    ) -> None:
        """__init__.

        Parameters
        ----------
        in_features : int
            in_features
        n_classes_per_task : list[int]
            n_classes_per_task
        task_weights : Tensor | Sequence | None
            task_weights
        l1_pars : dict
            Parameters for l1 regularization, dict of two keys: {"lambda", "exclude"}
            "lambda" is the regularization constant, "exclude" denotes named parameters
            to ignore from the calculation
        l2_pars : dict
            Parameters for l2 regularization, dict of two keys: {"lambda", "exclude"}

        Returns
        -------
        None

        """
        super().__init__()
        self._in_features: int = in_features
        self._n_tasks: int = len(n_classes_per_task)
        self._record: bool = record_metrics
        self._l1_pars: dict = if_none(l1_pars, {"lambda": 0, "exclude": set()})
        self._l2_pars: dict = if_none(l1_pars, {"lambda": 0, "exclude": set()})
        self._n_classes: Sequence[int] = n_classes_per_task
        self._task_weights: Tensor = None
        if task_weights is not None and not isinstance(task_weights, Tensor):
            self._task_weights = torch.tensor(task_weights)
        elif task_weights is not None:
            self._task_weights = task_weights
        self._accs: list[Accuracy] | None = None
        if self._record:
            self._accs = [
                Accuracy(task="multiclass", num_classes=n) for n in n_classes_per_task
            ]
        if task_names is None:
            self._task_names: Sequence[str] = [str(i) for i in range(self._n_tasks)]
        else:
            self._task_names = task_names

        self._optimizer_fn: Callable | None = optimizer_fn
        self._scheduler_fn: Callable | None = scheduler_fn
        self._scheduler_config: dict | None = scheduler_config

        # Cache results after iterations or validation for custom callbacks
        self._cache: dict[str, tuple[bool, list]] = {
            "train_loss": (False, []),
            "train_acc": (False, []),
            "val_acc": (False, []),
            "val_loss": (False, []),
            "test_loss": (False, []),
            "test_acc": (False, []),
        }
        if isinstance(cache, str):
            self.set_cache(cache)
        elif cache is not None:
            for c in cache:
                self.set_cache(c)

    @override
    def forward(self, X):
        raise NotImplementedError()

    def set_cache(
        self, value: Literal["train_loss", "val_acc", "val_loss", "train_acc"]
    ):
        if value not in self._cache:
            raise ValueError(f"Value to cache must be one of {self._cache.keys()}")
        self._cache[value] = (True, [])

    def _try_cache_to(self, target: str, value: Tensor) -> None:
        """Record ``value`` to the cache if it has been set for recording"""
        if self._cache[target][0]:
            self._cache[target][1].append(value.detach())

    def cache_clear(self, target) -> None:
        self._cache[target][1].clear()

    def _calc_accuracy(
        self,
        output: Tensor | tuple[Tensor],
        y_true: Tensor,
        prefix: str,
    ) -> None:
        if isinstance(output, tuple):
            preds: Tensor = torch.hstack(
                [p.argmax(axis=1).reshape(-1, 1) for p in output]
            )
        else:
            preds = output.argmax(axis=1)
        if isinstance(output, tuple):
            accs = []
            for i, (name, y_true, pred) in enumerate(
                zip(self._task_names, iter_cols(y_true), iter_cols(preds))
            ):
                acc = self._accs[i](pred, y_true)
                accs.append(acc)
                self.log(f"{prefix}_acc_{name}", acc)
            self._try_cache_to(f"{prefix}_acc", torch.tensor(accs).mean())
        else:
            acc = self._accs[0](preds, y_true)
            self.log(f"{prefix}_acc_step", acc)
            self._try_cache_to(f"{prefix}_acc", acc)

    def predict_proba(self, X) -> Tensor | tuple:
        if isinstance(X, DataLoader):
            X = X.dataset[:][0]
        X = torch.tensor(X) if isinstance(X, np.ndarray) else X
        proba = self(X)
        if isinstance(proba, tuple):
            return tuple(p.detach() for p in proba)
        return proba.detach()

    @override
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(y_pred=output, y_true=y)
        self.log("train_loss", loss)
        if self._record:
            self._calc_accuracy(output=output, y_true=y, prefix="train")
        self._try_cache_to("train_loss", loss)
        return loss

    @override
    def predict_step(self, batch, batch_idx=None, dataloader_idx=0) -> Tensor:
        try:
            X, _ = batch
        except ValueError:
            X = batch
        if isinstance(X, DataLoader):
            X = X.dataset[:][0]
        elif isinstance(X, Dataset):
            X = X[:]
        proba = self(X)
        if isinstance(proba, tuple):
            return torch.hstack([p.argmax(axis=1).reshape(-1, 1) for p in proba])
        return proba.argmax(axis=1)

    def _log_step(self, log_to, acc_prefix: str, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(y_pred=output, y_true=y)
        self.log(log_to, loss)
        self._try_cache_to(log_to, loss)
        if self._record:
            self._calc_accuracy(output=output, y_true=y, prefix=acc_prefix)
        return output

    @override
    def test_step(self, batch, batch_idx):
        _ = self._log_step("test_loss", "test", batch, batch_idx)

    @override
    def validation_step(self, batch, batch_idx):
        _ = self._log_step("val_loss", "val", batch, batch_idx)

    def reset_parameters(self):
        raise NotImplementedError()

    def register_optimizers(self, opt_fn: Callable):
        """Specify the optimizer to use for this model

        Parameters
        ----------
        opt_fn : Callable
            returns a Pytorch-compatible optimizer when called with named_parameters()
        """
        self._optimizer_fn = opt_fn

    def register_schedulers(
        self,
        scheduler_fn: Callable | None = None,
        lr_scheduler_config: None | dict = None,
    ):
        """Register a scheduler and/or scheduler config

        Parameters
        ----------
        scheduler_fn : Callable
            Function returning a Pytorch-compatible scheduler, taking the optimizer as
            the argument
        lr_scheduler_config : dict
            lr_scheduler_config as defined by Pytorch lightning
        """
        self._scheduler_fn = scheduler_fn
        self._scheduler_config = lr_scheduler_config

    @override
    def configure_optimizers(self) -> OptimizerConfig:
        if self._optimizer_fn is not None:
            optimizer = self._optimizer_fn(self.named_parameters())
        else:
            optimizer = optim.Adam(self.named_parameters(), lr=0.001)
        lr_scheduler_config = (
            self._scheduler_config.copy()
            if self._scheduler_config is not None
            else {"monitor": "train_loss"}
        )
        if self._scheduler_fn is None:
            lr_scheduler_config["scheduler"] = schedule.ReduceLROnPlateau(
                optimizer=optimizer, patience=40
            )
        else:
            lr_scheduler_config["scheduler"] = self._scheduler_fn(optimizer)
        return {"optimizer": optimizer, "lr_scheduler_config": lr_scheduler_config}

    def criterion(self, y_pred, y_true):
        """criterion.

        Parameters
        ----------
        y_pred :
            y_pred
        y_true :
            y_true
        """
        raise NotImplementedError()

    def l1(self) -> Tensor | Literal[0]:
        if self._l1_pars["lambda"] > 0:
            return l1(self, self._l1_pars["exclude"])
        return 0

    def l2(self) -> Tensor | Literal[0]:
        if self._l2_pars["lambda"] > 0:
            return l2(self, self._l2_pars["exclude"])
        return 0


# * Early stopping
#


class EarlyStopper:
    """A class implementing different early stopping techniques

    Parameters
    ----------
    trainer : Trainer class to apply early stopping with
    mode : type of early stopping to perform
    on_update : whether or not early stopping checks after each paramer update or
        each epoch (this also affects the "units" of `best_stop`)
    patience : tolerated number of iterations or epochs of no improvement
        in validation set error
    TODO: is it better to update on epochs or iterations? Seems that
        on_update = False should be better
    all : in the multi-task setting, the step is penalized unless
        improvements occur in ALL tasks of the validation set


    """

    def __init__(
        self,
        mode: Literal["simple"] = "simple",
        patience: int = 40,
        on_update: bool = True,
        higher_better: bool = True,
        all: bool = False,
    ) -> None:
        """__init__.

        Parameters
        ----------
        mode : Literal["simple"]
            mode
        patience : int
            patience
        on_update : bool
            on_update
        higher_better : bool
            higher_better
        all : bool
            all

        Returns
        -------
        None

        """
        self._mode: str = mode
        self._patience: int = patience
        self._all: bool = all
        self._on_update: bool = on_update
        self._higher_better: bool = higher_better  # if true, higher scores
        # mean improvement e.g. if using with accuracy
        self._tracker: int
        self._best_vset: Tensor
        self.best_stop: int

    def _reset(self) -> None:
        """_reset.

        Parameters
        ----------

        Returns
        -------
        None

        """
        self._tracker = 0
        self._best_vset = -torch.inf if self._higher_better else torch.inf
        self.best_stop = 0

    def _should_stop(self, score: Tensor, step: int) -> bool:
        """Check the current validation score

        Return
        ------
        True if the model has failed to improve the validation score for n > `_patience`
            rounds
        """
        if self._mode == "simple":
            passed: bool = False
            if self._higher_better:
                bools = score > self._best_vset
            else:
                bools = score < self._best_vset
            if len(bools) == 1:
                passed = bools
            else:
                if (self._all and all(bools)) or any(bools):
                    passed = True

            if not passed:
                self._tracker += 1
            else:
                self._tracker = 0
                self.best_stop = step
                self._best_vset = score
            b = self._tracker > self._patience
        else:
            raise NotImplementedError("Early stopping mode not supported yet")
        if b:
            print(f"Early stopping complete with best step {self.best_stop}")
        return b


# * Regularization


def l1(model: nn.Module, exclude: Sequence[str] = ()) -> Tensor | Literal[0]:
    """Compute l1 regularization

    Parameters
    ----------
    model : nn.Module
        model to apply l1 to
    exclude : Sequence[str]
        Sequence of parameter names to exclude from l1 calculation

    Returns
    -------
    Tensor

    """
    with torch.no_grad():
        return sum(
            torch.sum(torch.abs(v))
            for k, v in model.named_parameters()
            if k not in exclude
        )


def l2(model: nn.Module, exclude: Sequence[str] = ()) -> Tensor | Literal[0]:
    """Compute l2 regularization

    Parameters
    ----------
    model : nn.Module
        model to apply l2 to
    exclude : Sequence[str]
        Sequence of parameter names to exclude from l2 calculation

    Returns
    -------
    Tensor

    Notes
    -----
    Prefer to use the "weight_decay" parameter in the optimizer over this
    """
    with torch.no_grad():
        return (
            sum(
                torch.sum(torch.pow(v, 2))
                for k, v in model.named_parameters()
                if k not in exclude
            )
            / 2
        )
