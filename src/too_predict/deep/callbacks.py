#!/usr/bin/env ipython

from collections.abc import Callable
from copy import deepcopy
from functools import reduce
from typing import Literal, override

import lightning as L
import torch
from sortedcontainers import SortedList
from too_predict.deep.torch_utils import MultiModule

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
        self._best_epochs = LargestCollection(length=self.n_best, key=lambda x: x[0])

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
