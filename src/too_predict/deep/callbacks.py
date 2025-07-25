#!/usr/bin/env ipython

from collections.abc import Callable
from copy import deepcopy
from functools import reduce
from typing import Literal, override

import lightning as L
import torch
from sortedcontainers import SortedList
from too_predict.deep.torch_utils import MultiModule
from torch.optim.lr_scheduler import LRScheduler

"""
References
[1] Smith, S. L., Kindermans, P.-J., Ying, C., & Le, Q. V. (2018). Don’t Decay the Learning Rate, Increase the Batch Size (No. arXiv:1711.00489). arXiv. https://doi.org/10.48550/arXiv.1711.00489
"""

# * Utilities
#


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


# * Callbacks


class AverageBest(L.Callback):
    """Callback to average the weights of the n best epochs of a model's training

    Parameters
    ----------

    n_best : number of states to average over
    """

    def __init__(
        self,
        n_best: int,
        target: Literal["train_loss", "val_acc", "val_loss", "train_acc"] = "val_acc",
        decay: float = 0.999,
    ) -> None:
        super().__init__()
        self._best_epochs: LargestCollection | None = None
        self._n_best: int = n_best
        self._target: Literal["train_loss", "val_acc", "val_loss", "train_acc"] = target

    def _get_best_score_and_push(self, module: MultiModule, target: str) -> None:
        cached = module._cache[target][1]
        if cached:
            score = module._cache[target][1][-1]
            if self._best_epochs.would_push((score, None)):  # Make sure
                # copying is necessary before doing so
                self._best_epochs.push((score, deepcopy(module.state_dict())))
            module.cache_clear(target)

    @override
    def on_train_epoch_end(self, trainer, pl_module):
        self._get_best_score_and_push(pl_module, self._target)

    @override
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._target.startswith("val"):
            return
        self._get_best_score_and_push(pl_module, self._target)

    @override
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._best_epochs = LargestCollection(length=self._n_best, key=lambda x: x[0])

    @override
    def on_fit_end(self, trainer, pl_module) -> None:
        """Average collected parameters and transfer them to `model`"""
        parameters = []
        scores = []
        for score, p in self._best_epochs:
            parameters.append(p)
            scores.append(score)
        with torch.no_grad():
            try:
                averaged = reduce(
                    lambda x, y: {k: (y[k] + x[k]) / 2 for k in x.keys()},
                    parameters,
                )
                pl_module.load_state_dict(averaged)
            except RuntimeError:
                print(
                    "WARNING: AverageBest failed to take average, reverting to original weights..."
                )
                pass


class BatchSizeScaler(L.Callback):
    """Implements the increasing batch size strategy described in [1]

    Parameters
    ----------
    factor : Scaling factor by which to increase the batch size by
    total_iters : The number of epochs before the batch size getes scaled
    max : Maximum batch size before scaling stops completely
    scheduler_fn : Function taking the model's optimizer as a single argument
        and returns a scheduler
    callback_metric : Trainer metric to pass to scheduler on step. If not provided,
        scheduler is assumed not to require a metric
    """

    def __init__(
        self,
        factor: int = 5,
        max: int | None = None,
        total_iters: int = 10,
        scheduler_fn: Callable | None = None,
        callback_metric: str = "",
    ) -> None:
        super().__init__()
        self._factor: int = factor
        self._max: int | None = max
        self._total_iters: int = total_iters
        self._stopped: bool = False
        self._scheduler_fn: Callable | None = scheduler_fn
        self._scheduler: LRScheduler | None = None
        self._callback_metric: str = callback_metric

    @override
    def on_train_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        if self._max is None:
            self._max = len(trainer.train_dataloader.dataset) / 10
        # local import to avoid circular import
        from lightning.pytorch.strategies import DeepSpeedStrategy

        if self._scheduler_fn is not None:
            self._scheduler = self._scheduler_fn(trainer.model.optimizers())

        if isinstance(trainer.strategy, DeepSpeedStrategy):
            raise RuntimeError(
                f"The `{type(trainer.strategy).__name__}` does not support `accumulate_grad_batches` changing between epochs."
            )
        if trainer.accumulate_grad_batches != 1:
            raise ValueError(
                "You have set `accumulate_grad_batches` and are using the `GradientAccumulationScheduler` callback. Either remove `accumulate_grad_batches` from the Trainer or remove the callback."
            )
        return super().on_train_start(trainer, pl_module)

    @override
    def on_train_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        if self._stopped and self._scheduler is not None:
            if self._callback_metric:
                self._scheduler.step(
                    metrics=trainer.callback_metrics[self._callback_metric],
                    epoch=trainer.current_epoch,
                )
            else:
                self._scheduler.step(epoch=trainer.current_epoch)

    @override
    def on_train_epoch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        if self._stopped:
            return
        batch_size = trainer.train_dataloader.batch_size
        if trainer.accumulate_grad_batches * batch_size >= self._max:
            self._stopped = True
        elif (1 + trainer.current_epoch) % self._total_iters == 0:
            trainer.accumulate_grad_batches = (
                trainer.accumulate_grad_batches * self._factor
            )
