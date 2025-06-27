#!/usr/bin/env ipython

from __future__ import annotations

import math
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
import torch.nn.init as init
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


class Module(nn.Module):
    def __init__(self, n_tasks: int = 1) -> None:
        super().__init__()
        self.n_tasks: int = n_tasks

    @override
    def forward(self, X):
        raise NotImplementedError()

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

    def reset_parameters(self):
        raise NotImplementedError()

    def predict_proba(self, X) -> np.ndarray | tuple:
        X = torch.tensor(X) if isinstance(X, np.ndarray) else X
        proba = self(X)
        if isinstance(proba, tuple):
            return tuple(p.detach().numpy() for p in proba)
        return proba.detach().numpy()

    def get_optimizers(self) -> Optimizer:
        return optim.Adam(self.named_parameters())

    def criterion(self, y_pred, y_true):
        raise NotImplementedError()


# ** Subclass for multi-label classifier


class MultiModule(Module):
    def __init__(
        self,
        in_features: int,
        n_classes_per_task: list[int],
        task_weights: Tensor | Sequence | None = None,
    ) -> None:
        super().__init__(n_tasks=len(n_classes_per_task))
        self.task_weights: Tensor = None
        if task_weights is not None and not isinstance(task_weights, Tensor):
            self.task_weights = torch.tensor(task_weights)
        elif task_weights is not None:
            self.task_weights = task_weights

    @override
    def forward(self, X):
        return super().forward(X)

    @override
    def predict(self, X: Tensor | np.ndarray | DataLoader | Dataset) -> np.ndarray:
        return super().predict(X)

    @override
    def predict_proba(self, X) -> np.ndarray | tuple:
        return super().predict_proba(X)

    @override
    def reset_parameters(self):
        return super().reset_parameters()

    @override
    def get_optimizers(self) -> Optimizer:
        return super().get_optimizers()

    @override
    def criterion(self, y_pred, y_true):
        return super().criterion(y_pred, y_true)


# * Trainer


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
        at_batch_level: bool | int = True,
    ) -> None:
        self._evaluate: Callable
        self._n_epochs: int = n_epochs
        self.optimizer: Optimizer = (
            optimizer if optimizer is not None else model.get_optimizers()
        )
        self._es: EarlyStopper | None = None
        self._at_batch_level: bool | int = at_batch_level
        self.scheduler: schedule.LRScheduler | None = None
        self.model: Module = model

        # Obtain score function
        if score_fn is not None:
            self._train_score_key: str = f"train_{score_fn_name}"
            self._test_score_key: str = f"test_{score_fn_name}"
            self._evaluate = score_fn
        else:
            self._train_score_key = f"train_{score_metric}"
            self._test_score_key = f"test_{score_metric}"
            if score_metric == "accuracy":
                self._evaluate = met.accuracy_score
            elif score_metric == "balanced_accuracy":
                self._evaluate = met.balanced_accuracy_score
            elif score_metric == "f1":
                self._evaluate = met.f1_score
            elif score_metric == "precision":
                self._evaluate = met.precision_score
            elif score_metric == "mean_squared_error":
                self._evaluate = met.mean_squared_error
            elif score_metric == "recall":
                self._evaluate = met.recall_score

        # Training metric attributes
        self._record_train_score: bool = record_train_score
        self._record_test_score: bool = record_test_score
        self._batch_tracker: int = 0
        self._metrics: dict
        self._train_keys: list
        self._test_keys: list
        self._output_names: Sequence | None

        if self.model.n_tasks > 1 and output_names is None:
            self._output_names = range(self.model.n_tasks)
        else:
            self._output_names = output_names

    def _init_metrics(self):
        self._metrics = {"epoch": []}
        self._train_keys = []
        self._test_keys = []

        if self._at_batch_level:
            self._metrics["minibatch"] = []
            self._metrics["loss"] = []
            self._batch_tracker = 0
        else:
            self._metrics["avg_loss"] = []
        if self.model.n_tasks == 1:
            if self._record_train_score:
                self._metrics[self._train_score_key] = []
            if self._record_test_score:
                self._metrics[self._test_score_key] = []
        if self._output_names is not None:
            for name in self._output_names:
                if self._record_train_score:
                    key = f"{name}_{self._train_score_key}"
                    self._train_keys.append(key)
                    self._metrics[key] = []
                if self._record_test_score:
                    key = f"{name}_{self._test_score_key}"
                    self._test_keys.append(key)
                    self._metrics[key] = []

    def _record(self, X, y, single_key: str, multi_key: list[str]) -> Tensor:
        self.model.eval()
        y_pred = self.model.predict(X)
        if self._train_keys:
            score = torch.empty(len(self._train_keys))
            for i, k in enumerate(multi_key):
                s = self._evaluate(y_pred[:, i], y[:, i])
                self._metrics[k].append(s)
                score[i] = s
        else:
            score = self._evaluate(y_pred, y)
            self._metrics[single_key].append(score)
        self.model.train()
        return score

    def _should_record_batch(self) -> bool:
        if isinstance(self._at_batch_level, bool):
            return self._at_batch_level
        elif self._batch_tracker == self._at_batch_level:
            self._batch_tracker = 0
            return True
        self._batch_tracker += 1
        return False

    def _train_minibatch(
        self,
        train_x: Tensor,
        train_y: Tensor,
        vx: Tensor,
        vy: Tensor,
        validate: bool,
        epoch: int,
        iter: int,
        losses: list,
    ) -> Tensor | None:
        self.optimizer.zero_grad()
        out = self.model(train_x)
        loss: torch.Tensor = self.model.criterion(y_pred=out, y_true=train_y)
        loss.backward()

        v_score: Tensor | None = None
        should_record_batch = self._should_record_batch()
        if self._record_train_score and should_record_batch:
            _ = self._record(
                train_x,
                train_y,
                multi_key=self._train_keys,
                single_key=self._train_score_key,
            )
        if validate and self._record_test_score and should_record_batch:
            v_score = self._record(
                vx,
                vy,
                multi_key=self._test_keys,
                single_key=self._test_score_key,
            )
        if should_record_batch:
            self._metrics["epoch"].append(epoch)
            self._metrics["minibatch"].append(iter)
            self._metrics["loss"].append(loss.detach().numpy())
        else:
            losses.append(loss)
        self.optimizer.step()
        return v_score

    def register_early_stop(self, es: EarlyStopper) -> None:
        self._es = es
        self._at_batch_level = es._on_update

    def __call__(
        self, loader: DataLoader, validation: Dataset | None = None
    ) -> pd.DataFrame:
        self._init_metrics()
        self.model.reset_parameters()
        if self._es and validation is None:
            raise ValueError("Can't perform early stopping without a validation set!")
        if self._es:
            self._es._reset()
        self.model.train()

        if validation is None and self._record_test_score:
            raise ValueError("Can't record test score without validation set!")

        if validation is not None:
            validate: bool = True
            valid_x, valid_y = validation[:]
        else:
            valid_x, valid_y = None, None
            validate = False

        stop: bool = False
        n_updates: int = 0
        for i in range(self._n_epochs):
            losses = []
            for j, (X, y) in enumerate(loader):
                v_score = self._train_minibatch(
                    train_x=X,
                    train_y=y,
                    vx=valid_x,
                    vy=valid_y,
                    validate=validate,
                    losses=losses,
                    iter=j,
                    epoch=i,
                )
                if self._es and self._es._on_update:
                    if self._es._should_stop(v_score, n_updates):
                        stop = True
                        break
                n_updates += 1
            if self.scheduler is not None:
                self.scheduler.step()
            if not self._at_batch_level:  # Per-epoch metrics
                with torch.no_grad():
                    self._metrics["epoch"].append(i)
                    self._metrics["avg_loss"].append(np.mean(losses))
                if self._record_train_score:
                    x_train, y_train = loader.dataset[:]
                    self._record(
                        x_train,
                        y_train,
                        multi_key=self._train_keys,
                        single_key=self._train_score_key,
                    )
                if validate and self._record_test_score:
                    v_score = self._record(
                        valid_x,
                        valid_y,
                        multi_key=self._test_keys,
                        single_key=self._test_score_key,
                    )
                    if self._es and not self._es._on_update:
                        stop = self._es._should_stop(v_score, i)
            if stop:
                break

        self.model.eval()
        return pd.DataFrame(self._metrics)


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
        all: bool = False,
    ) -> None:
        self._mode: str = mode
        self._patience: int = patience
        self._all: bool = all
        self._on_update: bool = on_update
        self._tracker: int
        self._best_vset: Tensor  # Want to maximize this score
        self.best_stop: int

    def _reset(self) -> None:
        self._tracker = 0
        self._best_vset = -torch.inf
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
            return self._tracker > self._patience

        raise NotImplementedError("Early stopping mode not supported yet")


# * Utility functions


def reset_sequential(mod: nn.Module) -> None:
    def reset(m):
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Linear)
            or isinstance(m, Module)
        ):
            m.reset_parameters()

    mod.apply(reset)


def iter_cols(x: Tensor | np.ndarray) -> Iterable:
    if isinstance(x, Tensor):
        to_iter = torch.unbind(x, dim=1)
    else:
        to_iter = [x[:, i] for i in range(x.shape[1])]
    return to_iter


# def optimize():
#
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
    X: Dataset | DataLoader | torch.Tensor | np.ndarray | ad.AnnData,
    y: torch.Tensor | np.ndarray | None | pd.DataFrame = None,
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

    if (
        isinstance(y, tuple)
        and isinstance(next(iter(y)), str)
        and isinstance(X, ad.AnnData)
    ):
        y = X.obs.loc[:, y]
    if isinstance(X, ad.AnnData):
        X = ut.xarray_if_sparse(X)

    if isinstance(X, Dataset):
        return _for_dataset(X)
    elif isinstance(X, DataLoader):
        return _for_dataset(X.dataset)
    elif isinstance(y, pd.DataFrame):
        return X.shape[1], tuple([len(y[s].unique()) for s in y])
    elif isinstance(y, np.ndarray) and len(y.shape) > 1:
        return X.shape[1], tuple([n_uniques(y[:, i]) for i in range(y.shape[1])])
    return X.shape[1], len(set(y))
