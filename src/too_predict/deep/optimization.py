#!/usr/bin/env python

from collections.abc import Callable, Sequence
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
from too_predict.deep.evaluation import cross_validate, holdout


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
        module_fn = self._suggest_param_or_default("module")
        optimizer_fn: Callable = self._suggest_optimizer()
        scheduler_fn: Callable = self._suggest_scheduler()
        transformer: tt.Transformer = self._suggest_param_or_default("transformer")
        filter: fil.Filter = self._suggest_param_or_default("filter")
        return module_fn, optimizer_fn, scheduler_fn, transformer, filter


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
            score_fn=None,
            label_col=label_col,
            save_model=False,
            save_cv=save_cv,
            ignore_duplicated=ignore_duplicated,
            journal_file=journal_file,
            artifact_dir=artifact_dir,
        )

    @override
    def _objective(
        self,
        trial: optuna.Trial,
        adata: ad.AnnData,
        do_splits: bool = True,
        do_cv: bool = True,
        split_fns: dict | None = None,
        split_masks: dict | None = None,
        cv_splits=5,
        opts: dict | None = None,
        artifact_store: oa.FileSystemArtifactStore | None = None,
        **kwargs,
    ):
        if not do_splits and not do_cv:
            raise ValueError("One of do_splits or do_cv must be true!")
        setup = DlTrialSetup(trial=trial, **kwargs)
        module_fn, optimizer_fn, scheduler_fn, transformer, filter = setup(opts=opts)
        n_epochs = setup._suggest_param_or_default("n_epochs")
        n_features, n_classes = d_ut.data_spec(adata, y=self.label_col)
        module: d_ut.MultiModule = module_fn(
            in_features=n_features, n_classes_per_task=n_classes
        )
        optimizer = optimizer_fn(module.named_parameters())
        scheduler = scheduler_fn(optimizer)
        trainer = d_ut.Trainer(
            model=module,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=n_epochs,
            tol=8,
            record_test_score=False,
        )
        if filter != -1:
            adata = filter.fit_transform(adata)
        if transformer != -1:
            adata = transformer.fit_transform(adata)
        vals = []
        if do_splits:
            result: dict = holdout(
                trainer=trainer,
                adata=adata,
                n_classes=n_classes,
                to_encode=self.label_col,
                split_fns=split_fns,
                split_masks=split_masks,
                minimal=True,
                verbose=False,
            )
            vals.append(np.mean(result.values()))

        if do_cv:
            cv_results = cross_validate(
                trainer=trainer,
                adset=d_ut.AnnDataset(adata, to_encode=self.label_col),
                n_classes=n_classes,
                n_splits=cv_splits,
            )
            vals.append(np.mean(cv_results.values[:, 1:]))
        return np.mean(vals)
