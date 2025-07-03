#!/usr/bin/env ipython

from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import reduce
from typing import Literal, override

import pandas as pd
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as schedule
import torchmetrics.functional.classification as tmet
from sortedcontainers import SortedList
from too_predict.deep.torch_utils import EarlyStopper, Module
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset

# * Utilities


class LargestCollection:
    """A fixed length collection of objects that only keeps the largest values given to it
    Is sorted
    """

    def __init__(self, length: int, key: Callable | None = None) -> None:
        self._data: SortedList = SortedList(key=key)
        self._key: Callable = key if key is not None else lambda x: x
        self._length: int = length

    @override
    def __repr__(self) -> str:
        return repr(self._data)

    def __getitem__(self, x: int):
        return self._data[x]

    def __iter__(self):
        return iter(self._data)

    def would_push(self, item) -> bool:
        "Return True if item can be successfully added into the collection, but do not add it"
        if len(self._data) < self._length:
            return True
        return self._key(item) > self._key(self.min())

    def min(self):
        return self._data[0]

    def max(self):
        return self._data[-1]

    def push(self, item) -> bool:
        "Returns True if `item` was successfully pushed into the collection"
        if len(self._data) < self._length:
            self._data.add(item)
            return True
        if item < self._data[0]:
            return False
        _ = self._data.pop(0)
        self._data.add(item)
        return True


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
            "f1",
            "precision",
            "recall",
        ] = "accuracy",
        score_fn: Callable[[Tensor, Tensor], float] | None = None,
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
        self._avg_model: AverageModel | None = None
        self._avg_model_fn: Callable | None = None
        self._avg_mode: str | None = None

        # Obtain score function
        if score_fn is not None:
            self._train_score_key: str = f"train_{score_fn_name}"
            self._test_score_key: str = f"test_{score_fn_name}"
            self._evaluate = score_fn
        else:
            self._train_score_key = f"train_{score_metric}"
            self._test_score_key = f"test_{score_metric}"
            if score_metric == "accuracy":
                self._evaluate = tmet.accuracy
            elif score_metric == "f1":
                self._evaluate = tmet.f1_score
            elif score_metric == "precision":
                self._evaluate = tmet.precision
            elif score_metric == "recall":
                self._evaluate = tmet.recall

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
        self._n_classes: Sequence

    def _init_metrics(self):
        """_init_metrics."""
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
        """Record model's performance and optionally loss on X, y"""
        self.model.eval()
        y_pred = self.model.predict(X)
        if self._train_keys:
            score = torch.empty(len(self._train_keys))
            for i, k in enumerate(multi_key):
                s = self._evaluate(
                    preds=y_pred[:, i],
                    target=y[:, i],
                    task="multiclass",
                    num_classes=self._n_classes[i],
                )
                self._metrics[k].append(s)
                score[i] = s
        else:
            score = self._evaluate(
                preds=y_pred,
                target=y,
                task="multiclass",
                num_classes=self._n_classes[0],
            )
            self._metrics[single_key].append(score)
        self.model.train()
        return score

    def _should_record_batch(self) -> bool:
        """_should_record_batch.

        Parameters
        ----------

        Returns
        -------
        bool

        """
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
        """_train_minibatch.

        Parameters
        ----------
        train_x : Tensor
            train_x
        train_y : Tensor
            train_y
        vx : Tensor
            vx
        vy : Tensor
            vy
        validate : bool
            validate
        epoch : int
            epoch
        iter : int
            iter
        losses : list
            losses

        Returns
        -------
        Tensor | None

        """
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
            self._metrics["loss"].append(loss.detach())
        else:
            losses.append(loss.detach())
        self.optimizer.step()
        if self._avg_mode == "EMA" and epoch == 0:  # To prevent it making a deepcopy
            # of unitialized params
            self._avg_model = self._avg_model_fn(self.model)
        if self._avg_mode == "EMA" and epoch > 0:
            self._avg_model.update_parameters(self.model)
        return v_score

    def register_average(self, **kwargs):
        self._avg_model_fn = lambda x: AverageModel(x, **kwargs)
        self._avg_mode = kwargs["mode"]

    def register_early_stop(self, es: EarlyStopper) -> None:
        """Register early stopper"""
        self._es = es
        self._at_batch_level = es._on_update

    def deregister_early_stop(self) -> None:
        """Disable early stopping"""
        self._es = None

    def __call__(
        self,
        loader: DataLoader,
        n_classes: Sequence[int],
        validation: Dataset | None = None,
    ) -> pd.DataFrame:
        """__call__.

        Parameters
        ----------
        loader : DataLoader
            loader
        validation : Dataset | None
            validation

        Returns
        -------
        pd.DataFrame

        """
        self._init_metrics()
        self._n_classes = n_classes
        self.model.reset_parameters()
        if self._avg_mode is not None:
            self._avg_model = self._avg_model_fn(self.model)
        if self._es and validation is None:
            raise ValueError("Can't perform early stopping without a validation set!")
        if self._es:
            self._es._reset()
        self.model.train()

        if validation is None and self._record_test_score:
            raise ValueError("Can't record test score without validation set!")
        elif validation is None and self._avg_mode == "best_epochs":
            raise ValueError("Need a validation set to record best epochs!")

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
                    self._metrics["avg_loss"].append(torch.mean(torch.tensor(losses)))
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
                    if self._avg_mode == "best_epochs":
                        self._avg_model.update_parameters(
                            self.model, score=torch.mean(torch.tensor(v_score))
                        )
                    if self._es and not self._es._on_update:
                        stop = self._es._should_stop(v_score, i)
            if stop:
                break

        if self._avg_model is not None:
            self._avg_model.finalize(self.model)
        self.model.eval()
        metrics = pd.DataFrame(self._metrics)
        mapping = {k: float for k in metrics.select_dtypes(object).columns}
        return metrics.astype(mapping)


# * Model averaging


class AverageModel:
    def __init__(
        self,
        model: nn.Module,
        mode: Literal["EMA", "best_epochs"],
        n_best: int | None = None,
        decay: float = 0.999,
    ) -> None:
        self._best_epochs: LargestCollection | None = None
        self._model: AveragedModel | None = None
        if mode == "EMA":
            self._model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))
        elif mode == "best_epochs":
            self._best_epochs = LargestCollection(length=n_best, key=lambda x: x[0])
        self.mode: Literal["EMA", "best_epochs"] = mode

    def update_parameters(
        self, model: nn.Module, score: Sequence | float | None = None
    ) -> None:
        if self.mode == "EMA":
            self._model.update_parameters(model)
        elif self.mode == "best_epochs":
            if self._best_epochs.would_push((score, None)):  # Make sure
                # copying is necessary before doing so
                self._best_epochs.push((score, deepcopy(model.state_dict())))

    def finalize(self, model: nn.Module) -> None:
        """Average collected parameters and transfer them to `model`"""
        if self.mode == "EMA":
            source = self._model.state_dict()
            new_dict = {k: source[f"module.{k}"] for k in model.state_dict().keys()}
            model.load_state_dict(new_dict)
        else:
            parameters = []
            scores = []
            for score, p in self._best_epochs:
                parameters.append(p)
                scores.append(score)
            print(f"Taking average of epochs with scores: {scores}")
            with torch.no_grad():
                averaged = reduce(
                    lambda x, y: {k: (y[k] + x[k]) / 2 for k in x.keys()},
                    parameters,
                )
                model.load_state_dict(averaged)
