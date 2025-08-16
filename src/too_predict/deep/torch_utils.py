#!/usr/bin/env ipython

from __future__ import annotations

import itertools
import math
import time
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Iterator, Literal, override

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
from lightning.pytorch.loggers import CometLogger, TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities.types import OptimizerConfig
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from too_predict.utils import if_none
from torch import Tensor
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
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


def iter_cols(x: Tensor | np.ndarray | tuple) -> Iterable:
    """Iterate over columnes of x"""
    if isinstance(x, Tensor):
        to_iter = torch.unbind(x, dim=1)
    elif isinstance(x, np.ndarray):
        to_iter = [x[:, i] for i in range(x.shape[1])]
    else:
        to_iter = iter(x)
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
            n_classes = tuple([n_uniques(y)])
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
        ).to(self.device)
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


@dataclass
class ModuleConfig:
    def __init__(
        self,
        record_metrics: bool = True,
        task_names: Sequence[str] | None = None,
        task_weights: Tensor | Sequence | None = None,
        l1_pars: dict | None = None,
        l2_pars: dict | None = None,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
        scheduler_config: dict | None = None,
        cache: str | None | Sequence = None,
        record_norm: bool = False,
        dropout_p: float = 0.2,
        scaler: TorchScaler | None = None,
        init_device: str = "cpu",
        targets: tuple[Tensor] | None = None,
        outlayer_type: Literal["softmax", "regression"] = "softmax",
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        task_weights : Tensor | Sequence | None
            task_weights
        l1_pars : dict
            Parameters for l1 regularization, dict of two keys: {"lambda", "exclude"}
            "lambda" is the regularization constant, "exclude" denotes named parameters
            to ignore from the calculation
        l2_pars : dict
            Parameters for l2 regularization, dict of two keys: {"lambda", "exclude"}
        log_norm : bool
            Whether to log the gradient norm
        scaler : TorchScaler
            Fitted (ideally on entire train set) scaler that will apply transformation
            to each batch prior to training
        kwargs : Model-specific kwargs, stored in a dict
        targets : Tensor of targets that the model will observe during training
        """
        self.l1_pars: dict = if_none(l1_pars, {"lambda": 0, "exclude": set()})
        self.l2_pars: dict = if_none(l1_pars, {"lambda": 0, "exclude": set()})
        self.record_norm: bool = record_norm
        self.record: bool = record_metrics
        self._init_device: torch.device = torch.device(init_device)
        self.task_weights: Tensor = None
        if task_weights is not None and not isinstance(task_weights, Tensor):
            self.task_weights = torch.tensor(task_weights).to(self._init_device)
        self.optimizer_fn: Callable | None = optimizer_fn
        self.scheduler_fn: Callable | None = scheduler_fn
        self.scheduler_config: dict | None = scheduler_config
        self.dropout_p: float = dropout_p
        self.cache: str | Sequence | None = cache
        self.scaler: TorchScaler | None = scaler
        self.outlayer_type: str = outlayer_type
        if targets is not None:
            self.out_bias: tuple[Tensor, ...] | None = get_initial_out_bias(
                self.outlayer_type, targets
            )
        else:
            self.out_bias = None
        self.kwargs: dict = kwargs

    @property
    def init_device(self) -> torch.device:
        return self._init_device

    @init_device.setter
    def init_device(self, value: str | torch.device):
        if isinstance(value, str):
            self._init_device = torch.device(value)
        else:
            self._init_device = value


class MultiModule(L.LightningModule):
    """MultiModule."""

    def __init__(
        self,
        in_features: int,
        n_classes_per_task: list[int],
        conf: ModuleConfig | None = None,
        task_names: Sequence | None = None,
    ) -> None:
        """
        Parameters
        ----------
        in_features : int
            number of incoming features
        n_classes_per_task : list[int]
            n_classes_per_task

        """
        super().__init__()
        self.in_features: int = in_features
        self.n_tasks: int = len(n_classes_per_task)
        self.conf: ModuleConfig = ModuleConfig() if conf is None else conf
        self.n_classes: Sequence[int] = n_classes_per_task
        self._accs: nn.ModuleList | None = None
        self.supervised: bool = True
        if self.conf.record:
            self._accs = nn.ModuleList(
                [Accuracy(task="multiclass", num_classes=n) for n in n_classes_per_task]
            )
        if task_names is None:
            self.task_names: Sequence[str] = [str(i) for i in range(self.n_tasks)]
        else:
            self.task_names = task_names

        # Cache results after iterations or validation for custom callbacks
        self.cache: dict[str, tuple[bool, list]] = {
            "train_loss": (False, []),
            "train_acc": (False, []),
            "val_acc": (False, []),
            "val_loss": (False, []),
            "test_loss": (False, []),
            "test_acc": (False, []),
        }
        if isinstance(self.conf.cache, str):
            self.set_cache(self.conf.cache)
        elif self.conf.cache is not None:
            for c in self.conf.cache:
                self.set_cache(c)

    def init_out_bias(
        self, targets: tuple[Tensor] | Dataset | DataLoader | None = None
    ) -> None:
        """Initialize biases of output layers based on training data"""
        if self.conf.out_bias is not None:
            biases = [b.clone() for b in self.conf.out_bias]
        else:
            biases = get_initial_out_bias(self.conf.outlayer_type, targets)
        for i, bias in enumerate(biases):
            layer = self.get_outlayer(i)
            layer.bias = nn.Parameter(bias).to(self.device)

    def get_outlayer(self, i: int) -> nn.Module:
        """Return the module's ith outlayer"""
        raise NotImplementedError()

    @classmethod
    def new(
        cls,
        in_features: int,
        n_classes_per_task: list[int],
        conf: ModuleConfig | None = None,
        **kwargs,
    ):
        return cls(in_features, n_classes_per_task, conf=conf, **kwargs)

    @override
    def forward(self, X):
        "Should return unnormalized logits. predict_proba returns probabilities"
        raise NotImplementedError()

    def set_cache(
        self, value: Literal["train_loss", "val_acc", "val_loss", "train_acc"]
    ):
        if value not in self.cache:
            raise ValueError(f"Value to cache must be one of {self.cache.keys()}")
        self.cache[value] = (True, [])

    def _try_cache_to(self, target: str, value: Tensor) -> None:
        """Record ``value`` to the cache if it has been set for recording"""
        if self.cache[target][0]:
            self.cache[target][1].append(value.detach())

    def cache_clear(self, target) -> None:
        self.cache[target][1].clear()

    def _calc_accuracy(
        self,
        output: Tensor | tuple[Tensor],
        y_true: Tensor,
        prefix: str,
    ) -> None:
        output = self._to_proba(output)
        if isinstance(output, tuple):
            preds: Tensor = torch.hstack(
                [p.argmax(axis=1).reshape(-1, 1) for p in output]
            )
        else:
            preds = output.argmax(axis=1)
        if isinstance(output, tuple):
            accs = []
            for i, (name, y_true, pred) in enumerate(
                zip(self.task_names, iter_cols(y_true), iter_cols(preds))
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
        proba = self._to_proba(self(X))
        if isinstance(proba, tuple):
            return tuple(p.detach() for p in proba)
        return proba.detach()

    def maybe_scale(self, x):
        if self.conf.scaler is not None:
            return self.conf.scaler.transform(x)
        return x

    @override
    def training_step(self, batch, batch_idx):
        if self.supervised:
            x, y = batch
        else:
            x, y = batch, None
        x = self.maybe_scale(x)
        output = self(x)
        loss = self.criterion(y_pred=output, y_true=y, context="train")
        self.log("train_loss", loss)
        if self.conf.record:
            self._calc_accuracy(output=output, y_true=y, prefix="train")
        self._try_cache_to("train_loss", loss)
        if self.conf.record_norm:
            norm = (
                sum(
                    p.grad.norm(2).item() ** 2
                    for p in self.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            self.log("gradient_norm", norm)
        return loss

    @staticmethod
    def _to_proba(forward_out: tuple | Tensor, log: bool = False) -> tuple | Tensor:
        "Convert logits from ``forward`` into normalized probabilities with softmax"
        if log:
            fn: Callable = nn.functional.log_softmax
        else:
            fn = nn.functional.softmax
        if isinstance(forward_out, tuple):
            return tuple([fn(p) for p in forward_out])
        return fn(forward_out)

    @override
    def predict_step(self, batch, batch_idx=None, dataloader_idx=0) -> Tensor:
        try:
            x, _ = batch
        except ValueError:
            x = batch
        if isinstance(x, DataLoader):
            x = x.dataset[:][0]
        elif isinstance(x, Dataset):
            x = x[:]
        x = self.maybe_scale(x)
        proba = self._to_proba(self(x))
        if isinstance(proba, tuple):
            return torch.hstack([p.argmax(axis=1).reshape(-1, 1) for p in proba])
        return proba.argmax(axis=1)

    def _log_step(self, log_to, acc_prefix: str, batch, batch_idx):
        x, y = batch
        x = self.maybe_scale(x)
        output = self(x)
        loss = self.criterion(y_pred=output, y_true=y, context=acc_prefix)
        self.log(log_to, loss)
        self._try_cache_to(log_to, loss)
        if self.conf.record:
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
        self.conf.optimizer_fn = opt_fn

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
        self.conf.scheduler_fn = scheduler_fn
        self.conf.scheduler_config = lr_scheduler_config

    @override
    def configure_optimizers(self) -> OptimizerConfig:
        if self.conf.optimizer_fn is not None:
            optimizer = self.conf.optimizer_fn(self.named_parameters())
        else:
            optimizer = optim.Adam(self.named_parameters(), lr=0.001)
        lr_scheduler_config = (
            self.conf.scheduler_config.copy()
            if self.conf.scheduler_config is not None
            else {"monitor": "train_loss"}
        )
        if self.conf.scheduler_fn is None:
            lr_scheduler_config["scheduler"] = schedule.ReduceLROnPlateau(
                optimizer=optimizer, patience=40
            )
        else:
            lr_scheduler_config["scheduler"] = self.conf.scheduler_fn(optimizer)
        return {"optimizer": optimizer, "lr_scheduler_config": lr_scheduler_config}

    def criterion(self, y_pred, y_true, context: str | None = None):
        """criterion.

        Parameters
        ----------
        context : str
            Optional string denoting when the criterion was calculated e.g. "train"
            "validation" etc.
        """
        raise NotImplementedError()

    def l1(self) -> Tensor | Literal[0]:
        if self.conf.l1_pars["lambda"] > 0:
            return l1(self, self.conf.l1_pars["exclude"])
        return 0

    def l2(self) -> Tensor | Literal[0]:
        if self.conf.l2_pars["lambda"] > 0:
            return l2(self, self.conf.l2_pars["exclude"])
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


# * Logging


def lightning_logger(
    exp_name: str,
    prefix_with_date: bool = True,
    platform: Literal["comet", "wandb", "tensorboard"] = "comet",
    **kwargs,
) -> CometLogger | WandbLogger | TensorBoardLogger:
    if prefix_with_date:
        d = date.today().isoformat() if prefix_with_date else ""
        exp_name = f"{d}-{exp_name}"
    if platform == "comet":
        logger = CometLogger(**kwargs)
        logger.experiment.set_name(exp_name)
    elif platform == "wandb":
        kwargs.update({"name": exp_name})
        logger = WandbLogger(**kwargs)
    elif platform == "tensorboard":
        if "save_dir" in kwargs:
            kwargs["name"] = ""
        kwargs.update({"version": exp_name})
        logger = TensorBoardLogger(**kwargs)
    return logger


def tflog2pandas(path: Path | str) -> pd.DataFrame:
    """Extract metrics from tensorboard logdir as a df"""
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    if isinstance(path, Path):
        path = str(path)
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags: Sequence = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data.pivot(columns="metric", values="value")


# * Normalization and scaling


class TorchScaler:
    def __init__(self) -> None:
        pass

    def fit(self, x: Tensor) -> None:
        raise NotImplementedError

    def transform(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def fit_transform(self, x: Tensor) -> Tensor:
        self.fit(x)
        return self.transform(x)


class TorchStandardScaler(TorchScaler):
    def __init__(self) -> None:
        super().__init__()
        self.mean: Tensor
        self.std: Tensor

    @override
    def fit(self, x: Tensor):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    @override
    def transform(self, x):
        x -= self.mean
        x /= self.std + 1e-7
        return x


class TorchMinMaxScaler(TorchScaler):
    def __init__(self) -> None:
        super().__init__()
        self.min: Tensor
        self.diff: Tensor

    @override
    def fit(self, x: Tensor) -> None:
        self.min = x.min(0).values
        self.diff = x.max(0).values - self.min

    @override
    def transform(self, x: Tensor) -> Tensor:
        x_std = (x - self.min) / self.diff
        return x_std * self.diff + self.min


# * Samplers


class BootstrapSequential(SequentialSampler):
    def __init__(self, data_source, n: int) -> None:
        super().__init__(data_source)
        self.n: int = n

    @override
    def __iter__(self) -> Iterator[int]:
        length = len(self.data_source)
        indices_iter: Iterator = iter(range(length))
        if self.n > length:
            diff = self.n - length
            extension = []
            while diff > 0:
                extension.append(iter(range(0, min(diff, length))))
                diff -= length
            indices_iter = itertools.chain(indices_iter, *extension)
        return indices_iter

    @override
    def __len__(self) -> int:
        return self.n


class BootstrapSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        n_batches: int,
        data_source=None,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__(data_source)
        """Custom sampler supporting batch sizes larger than training set through
            bootstrapping

        Parameters
        ----------
        batch_size : Batch size.
        n_batches : Number of batches to produce
        data_source : Dataset
            Dataset to get batches from
        shuffle : bool
            Whether to shuffle data. If False, data are sampled in the same
            order for every batch
        pre_shuffle : bool
            If True and shuffle is True, then the indices are shuffled once, and subsequent
            batches use the same ordering
        generator : Generator
            Generator object for randomization
        """
        if not shuffle:
            inner = BootstrapSequential(
                data_source=data_source, n=batch_size * n_batches
            )
        else:
            inner = RandomSampler(
                data_source=data_source,
                replacement=True,
                num_samples=batch_size * n_batches,
            )
        self._sampler: BatchSampler = BatchSampler(
            inner, batch_size=batch_size, drop_last=drop_last
        )

    @override
    def __iter__(self):
        return iter(self._sampler)


def update_batch_strategy(
    loader_kwargs: dict,
    dataset: Dataset,
    default_batch_size=512,
    trainer_kwargs: dict | None = None,
    grad_accumulation: bool = False,
    grad_accumulation_batch_size: int = 256,
) -> tuple[dict, dict]:
    """R

    Parameters
    ----------
    grad_accumulation : whether to `accumulate_grad_batches` in the Trainer instead
        of passing true batches
    grad_accumulation_batch_size : Actual batch size used by the Trainer. The number
        of accumulation rounds is thus given by
        default_batch_size(or loader_kwargs['batch_size']) // grad_accumulation_batch_size

    Returns
    -------
    A tuple of [dataloader_kwargs, trainer_kwargs] to pass to the DataLoader and Trainer
        respectively
    """
    loader_kwargs = loader_kwargs.copy()
    if grad_accumulation and not isinstance(trainer_kwargs, dict):
        raise ValueError(
            "If you intend to use gradient accumulation to increase batch size, `trainer_kwargs` must be given!"
        )
    elif isinstance(trainer_kwargs, dict):
        trainer_kwargs = trainer_kwargs.copy()
    else:
        trainer_kwargs = {}

    if "batch_size" not in loader_kwargs:
        loader_kwargs["batch_size"] = default_batch_size

    use_whole: bool = loader_kwargs.get("batch_size") == -1
    old_batch_size = loader_kwargs["batch_size"]

    if not use_whole and (
        old_batch_size > len(dataset)
        or (
            ((len(dataset) / loader_kwargs["batch_size"]) <= 1)
            and not grad_accumulation
        )
    ):
        del loader_kwargs["batch_size"]
        loader_kwargs["batch_sampler"] = BootstrapSampler(
            batch_size=old_batch_size,
            n_batches=loader_kwargs.pop("n_batches", 5),
            data_source=dataset,
            shuffle=loader_kwargs.pop("shuffle", True),
            drop_last=loader_kwargs.pop("drop_last", False),
        )

    if use_whole:
        if not grad_accumulation:
            loader_kwargs["batch_size"] = len(dataset)
        else:
            loader_kwargs["batch_size"] = grad_accumulation_batch_size
            length = len(dataset)
            remainder = length % grad_accumulation_batch_size
            n_acc = length // grad_accumulation_batch_size
            if n_acc <= 1:
                raise ValueError(
                    "grad_accumulation_batch_size is too large! Set a smaller value"
                )
            if remainder > 0:
                loader_kwargs["batch_size"] += remainder // (n_acc + 1)
            trainer_kwargs["accumulate_grad_batches"] = n_acc
    elif grad_accumulation:
        if not loader_kwargs.get("batch_sampler", False):
            loader_kwargs["batch_size"] = grad_accumulation_batch_size
        n_acc = old_batch_size // grad_accumulation_batch_size
        trainer_kwargs["accumulate_grad_batches"] = (
            n_acc if (old_batch_size % grad_accumulation_batch_size == 0) else n_acc + 1
        )
    return loader_kwargs, trainer_kwargs


# * Initialization


def get_initial_out_bias(
    mode: Literal["softmax", "regression"],
    targets: tuple[Tensor, ...] | Dataset | DataLoader,
):
    if isinstance(targets, DataLoader):
        targets = targets.dataset
    if isinstance(targets, Dataset):
        targets = targets[:][1]
    if isinstance(targets, Tensor):
        targets = iter_cols(targets)
    if mode == "softmax":
        tmp = []
        for t in targets:
            _, counts = t.unique(return_counts=True)
            tmp.append(counts / counts.sum())
        return tuple(tmp)
    elif mode == "regression":
        return tuple([t.mean() for t in targets])


def init_lazy(model: L.LightningModule, loader: DataLoader) -> None:
    x, _ = next(iter(loader))
    model(x)
