#!/usr/bin/env python

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import override

import anndata as ad
import lightning as L
import numpy as np
import optuna
import optuna.artifacts as oa
import too_predict.deep.torch_utils as d_ut
import too_predict.filter as fil
import too_predict.optimization as topt
import too_predict.transformer as tt
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
from sklearn.model_selection import train_test_split
from too_predict.deep.evaluation import cross_validate, holdout
from too_predict.deep.nns import Disyak
from too_predict.deep.trainer import Trainer
from too_predict.utils import train_test_split_ad

# * HPO


class DlTrialSetup(topt.TrialSetup):
    def __init__(
        self,
        trial: optuna.Trial | None = None,
        trial_params: dict | None = None,
        **kwargs,
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

    def _suggest_module(self) -> Callable:
        module = self._suggest_param_or_default("module")
        if isinstance(module, Callable):  # No need to adjust module parameters
            return module
        elif isinstance(module, str):
            if module == "Disyak":
                dropout = self._suggest_param_or_default("dropout")
                l1_pars = self._suggest_param_or_default("l1_pars")
                l2_pars = self._suggest_param_or_default("l2_pars")
                task_weights = self._suggest_param_or_default("task_weights")
                n_hidden = self._suggest_param_or_default("n_hidden")
                return lambda in_features, n_classes_per_task: Disyak(
                    in_features=in_features,
                    n_classes_per_task=n_classes_per_task,
                    dropout_p=dropout,
                    l1_pars=l1_pars,
                    l2_pars=l2_pars,
                    task_weights=task_weights,
                    n_hidden=n_hidden,
                )
            else:
                raise ValueError("Module name not recognized!")
        else:
            raise ValueError("Parameter for module must be string or callable")

    @override
    def __call__(self, opts: dict | None = None) -> tuple:
        self.user_opts: dict = opts
        module_fn = self._suggest_module()
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
        """
        Additional Parameters via **kwargs:
        -----------------------------------
        at_batch_level : bool, optional
            If True, records metrics at the batch level during training. Default is False.
        early_stop : EarlyStopping or compatible object, optional
            An early stopping criterion to be registered with the trainer. If provided, training may stop early based on the monitored metric.
        average_model_kwargs : dict, optional
            Keyword arguments to configure model weight averaging during training. Passed to `trainer.register_average(**average_model_kwargs)`.
        test_size : float, optional
            Proportion of the dataset to allocate to the test split in holdout validation. Default is 0.1.
        intermediate_out : Path, optional
            Directory path where intermediate outputs (e.g., cross-validation results) will be saved. If provided, a subdirectory named after the trial number will be created.
        save_intermediate : bool, optional
            If True, saves intermediate results during cross-validation to the specified `intermediate_out` directory. Default is False.
        verbose : bool, optional
            If True, enables verbose output during cross-validation. Default is False.
        batch_size : int, optional
            Mini-batch size to use during cross-validation. Default is 32.
        """

        if not do_splits and not do_cv:
            raise ValueError("One of do_splits or do_cv must be true!")
        setup = DlTrialSetup(trial=trial, **kwargs)
        n_features, n_classes = d_ut.data_spec(adata, y=self.label_col)
        opts["n_classes"] = n_classes
        module_fn, optimizer_fn, scheduler_fn, transformer, filter = setup(opts=opts)
        n_epochs = setup._suggest_param_or_default("n_epochs")
        module: d_ut.MultiModule = module_fn(
            in_features=n_features, n_classes_per_task=n_classes
        )
        module.register_optimizers(opt_fn=optimizer_fn)
        module.register_schedulers(scheduler_fn=scheduler_fn)
        for cache in kwargs.get("set_cache", []):
            module.set_cache(cache)
        callbacks = []
        callbacks.extend(kwargs.get("callbacks", []))
        log_root: Path | None = kwargs.get("intermediate_out", None)
        if log_root is not None:
            log_root = log_root.joinpath(str(trial.number))
            log_root.mkdir(exist_ok=True)
        trainer = L.Trainer(
            max_epochs=n_epochs,
            enable_checkpointing=False,  # Do not change this
            callbacks=callbacks,
            default_root_dir=log_root,
        )
        if filter != -1:
            adata = filter.fit_transform(adata)
        if transformer != -1:
            adata = transformer.fit_transform(adata)
        train, test = train_test_split_ad(adata, test_size=kwargs.get("test_size", 0.1))
        vals = []
        if do_splits:
            result: dict = holdout(
                trainer=trainer,
                adata=train,
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
                model=module,
                trainer=trainer,
                adset=d_ut.AnnDataset(train, to_encode=self.label_col),
                validation=d_ut.AnnDataset(test, to_encode=self.label_col),
                n_classes=n_classes,
                n_splits=cv_splits,
                verbose=kwargs.get("verbose", False),
                batch_size=kwargs.get("batch_size", 512),
            )
            if log_root is not None:
                cv_results.to_csv(log_root.joinpath("fold_summary.csv"), index=False)
            vals.append(np.mean(cv_results.values[:, 1:]))
        mean_accs = tuple(np.mean(cv_results.loc[:, label]) for label in self.label_col)
        return mean_accs
