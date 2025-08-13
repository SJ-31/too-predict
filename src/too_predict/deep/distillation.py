#!/usr/bin/env python

from types import MethodType
from typing import override

import lightning as L
import torch
import torch.nn as nn
from too_predict.deep.nns import Baseline
from too_predict.deep.torch_utils import AnnDataset, MultiModule
from torch import Tensor


class TeacherResponse(torch.utils.data.Dataset):
    def __init__(
        self,
        data: AnnDataset,
        teacher: Baseline | MultiModule,
        is_fitted: bool = False,
        trainer: L.Trainer | None = None,
        **kwargs,
    ) -> None:
        self.data: AnnDataset = data
        self.teacher: Baseline | MultiModule = teacher
        self.input: Tensor = self.data[:][0]
        self.response: Tensor
        self.device: torch.device = data.device
        self.label_cols: tuple = self.data.label_cols
        if isinstance(self.teacher, Baseline):
            if not is_fitted:
                self.teacher.fit(self.data)
            proba = tuple(
                [
                    torch.tensor(p).to(self.device)
                    for p in self.teacher.predict_proba(self.input)
                ]
            )
        else:
            if not is_fitted:
                trainer.fit(model=self.teacher, train_dataloaders=self.data, **kwargs)
            proba = self.teacher.predict_proba(self.input)
        self.response = proba

    def get_targets(self):
        return self.data[:][1]

    @override
    def __getitem__(self, index):
        return self.input[index, :], tuple([r[index, :] for r in self.response])

    def __len__(self) -> int:
        return len(self.data)

    @property
    def shape(self) -> tuple:
        return self.data.X.shape


def distillation_loss(self, y_pred: tuple, y_true: tuple, context: str | None = None):
    total_loss: torch.Tensor = 0
    y_pred = self._to_proba(y_pred, log=True)
    if self.n_tasks > 1:
        for student_prob, teacher_prob in zip(y_pred, y_true):
            total_loss += nn.functional.kl_div(input=student_prob, target=teacher_prob)
    else:
        total_loss += nn.functional.kl_div(input=y_pred, target=y_true)
    total_loss += self.l2() + self.l1()
    return total_loss


def use_kd_criterion(model: MultiModule):
    "Swap ``model``'s criterion method for distillation loss"
    model.criterion = MethodType(distillation_loss, model)
    model.conf.record = False
    #  TODO: can't calculate accuracy while using distillation,
    #  but maybe some other metric would work
