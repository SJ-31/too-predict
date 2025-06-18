#!/usr/bin/env python

from collections.abc import Callable
from pathlib import Path
from typing import override

import anndata as ad
import numpy as np
import optuna
import optuna.artifacts as oa
import too_predict.deep.torch_utils as d_ut
import too_predict.filter as fil
import too_predict.optimization as topt
import too_predict.transformer as tt
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
from too_predict.deep.evaluation import holdout
from torch.optim import Optimizer


class DlTrialSetup(topt.TrialSetup):
    def __init__(
        self, trial: optuna.Trial | None = None, trial_params: dict | None = None
    ) -> None:
        super().__init__(trial, trial_params)

    def _suggest_optimizer(self) -> Callable:
        name = self._suggest_param_or_default("optimizer")
        lr = self._suggest_param_or_default("lr")
        if name == "Adam":
            betas = self._suggest_param_or_default("betas")
            weight_decay = self._suggest_param_or_default("weight_decay")
            amsgrad = self._suggest_param_or_default("amsgrad")
            return lambda x: optim.Adam(
                x, lr=lr, betas=betas, weight_decay=weight_decay, amsgrad=amsgrad
            )
        if name == "SGD":
            momentum = self._suggest_param_or_default("momentum")
            weight_decay = self._suggest_param_or_default("weight_decay")
            return lambda x: optim.SGD(
                x, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"{name} not implemented!")

    def _suggest_scheduler(self) -> Callable:
        name = self._suggest_param_or_default("scheduler")
        if name == "ReduceLROnPlateau":
            patience = self._suggest_param_or_default("patience")
            factor = self._suggest_param_or_default("factor")
            return lambda x: schedule.ReduceLROnPlateau(
                x, factor=factor, patience=patience
            )
        elif name == "CyclicLR":
            mode = self._suggest_param_or_default("mode")
            return lambda x: schedule.CyclicLR(x, mode=mode)
        elif name == "PolynomialLR":
            power = self._suggest_param_or_default("power")
            total_iters = self._suggest_param_or_default("total_iters")
            return lambda x: schedule.PolynomialLR(
                x, power=power, total_iters=total_iters
            )
        else:
            raise ValueError(f"{name} not implemented!")

    @override
    def __call__(self, opts: dict | None = None) -> tuple:
        self.user_opts: dict = opts
        module = self._suggest_param_or_default("module")
        optimizer: Optimizer = self._suggest_optimizer(module)
        scheduler = self._suggest_scheduler(optimizer)
        transformer: tt.Transformer = self._suggest_param_or_default("transformer")
        filter: fil.Filter = self._suggest_param_or_default("filter")
        return module, optimizer, scheduler, transformer, filter


class DlOptimizer(topt.BaseOptimizer):
    def __init__(
        self,
        label_col: Sequence[str] = ("tumor_type", "Sample_Type"),
        save_cv: bool = True,
        ignore_duplicated: bool = True,
        journal_file: Path | str | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        super().__init__(
            None,
            label_col,
            False,
            save_cv,
            ignore_duplicated,
            None,
            journal_file,
            artifact_dir,
        )

    @override
    def _objective(
        self,
        trial: optuna.Trial,
        adata: ad.AnnData,
        split_fns: dict | None = None,
        split_masks: dict | None = None,
        cv_splits=5,
        opts: dict | None = None,
        artifact_store: oa.FileSystemArtifactStore | None = None,
        **kwargs,
    ):
        setup = DlTrialSetup(trial=trial, **kwargs)
        module, optimizer, scheduler, transformer, filter = setup(opts=opts)
        n_epochs = setup._suggest_param_or_default("n_epochs")
        trainer = d_ut.Trainer(
            model=module,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=n_epochs,
            tol=8,
        )
        if filter != -1:
            adata = filter.fit_transform(adata)
        if transformer != -1:
            adata = transformer.fit_transform(adata)
        result: dict = holdout(
            trainer=trainer,
            adata=adata,
            to_encode=self.label_col,
            split_fns=split_fns,
            split_masks=split_masks,
            minimal=True,
            verbose=False,
        )
        return np.mean(result.values())
