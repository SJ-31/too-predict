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
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
from lightning.pytorch.loggers import Logger
from too_predict.deep.callbacks import BatchSizeScaler
from too_predict.deep.evaluation import cross_validate, holdout
from too_predict.deep.nns import Disyak, HardSharer
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

    def _suggest_scheduler(self) -> tuple[bool, Callable]:
        name = self._suggest_param_or_default("scheduler")
        if name == "ReduceLROnPlateau":
            patience = self._suggest_param_or_default("patience")
            factor = self._suggest_param_or_default("factor")
            return False, lambda x: schedule.ReduceLROnPlateau(
                x, factor=factor, patience=patience
            )
        elif name == "CyclicLR":
            mode = self._suggest_param_or_default("mode")
            return False, lambda x: schedule.CyclicLR(x, mode=mode)
        elif name == "PolynomialLR":
            power = self._suggest_param_or_default("power")
            total_iters = self._suggest_param_or_default("total_iters")
            return False, lambda x: schedule.PolynomialLR(
                x, power=power, total_iters=total_iters
            )
        elif name == "BatchSizeScaler":
            factor = self._suggest_param_or_default("bs_factor")
            total_iters = self._suggest_param_or_default("bs_interval")
            return True, lambda x: BatchSizeScaler(
                factor=factor,
                total_iters=total_iters,
                scheduler_fn=lambda x: schedule.StepLR(x, step_size=10),
            )
        else:
            raise ValueError(f"{name} not implemented!")

    def _suggest_module(
        self,
    ) -> tuple[d_ut.MultiModule, dict, d_ut.ModuleConfig | None]:
        model = self._suggest_param_or_default("module")
        if isinstance(model, d_ut.MultiModule):  # No need to adjust module parameters
            return model, {}, None
        elif isinstance(model, str):
            conf = d_ut.ModuleConfig(
                dropout=self._suggest_param_or_default("dropout"),
                l1_pars=self._suggest_param_or_default("l1_pars"),
                l2_pars=self._suggest_param_or_default("l2_pars"),
                task_weights=self._suggest_param_or_default("task_weights"),
            )
            if model == "Disyak":
                n_hidden = self._suggest_param_or_default("n_hidden")
                kwargs = dict(
                    n_hidden=n_hidden,
                )
                return Disyak, kwargs, conf
            else:
                raise ValueError("Module name not recognized!")
        else:
            raise ValueError("Parameter for module must be string or callable")

    @override
    def __call__(self, opts: dict | None = None) -> tuple:
        self.user_opts: dict = opts
        model_cls, model_kwargs, model_config = self._suggest_module()
        model_kwargs["scaler"] = self._suggest_param_or_default("scaler")
        optimizer_fn: Callable = self._suggest_optimizer()
        is_callback, scheduler_fn = self._suggest_scheduler()
        model_kwargs["optimizer_fn"] = optimizer_fn
        if not is_callback:
            model_config.scheduler_fn = scheduler_fn
        else:
            self.user_opts["callbacks"].append(scheduler_fn)
        transformer: tt.Transformer = self._suggest_param_or_default("transformer")
        filter: fil.Filter = self._suggest_param_or_default("filter")
        return model_cls, model_kwargs, model_config, transformer, filter


class DlOptimizer(topt.BaseOptimizer):
    def __init__(
        self,
        label_col: Sequence[str] = ("tumor_type", "Sample_Type"),
        save_cv: bool = True,
        ignore_duplicated: bool = True,
        storage_file: Path | str | None = None,
        artifact_dir: Path | None = None,
        log_fn: Callable[[str], Logger] | None = None,
    ) -> None:
        super().__init__(
            score_fn=None,
            label_col=label_col,
            save_model=False,
            save_cv=save_cv,
            ignore_duplicated=ignore_duplicated,
            storage_file=storage_file,
            artifact_dir=artifact_dir,
        )
        self.log_fn: Callable = log_fn

    @override
    def _objective(
        self,
        trial: optuna.Trial,
        adata: ad.AnnData,
        do_splits: bool = True,
        do_cv: bool = True,
        split_fns: dict | None = None,
        split_masks: dict | None = None,
        device: str = "cpu",
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
        model_cls, model_kwargs, model_config, transformer, filter = setup(opts=opts)
        model_config: d_ut.ModuleConfig
        n_epochs = setup._suggest_param_or_default("n_epochs")
        matmul_precision: float | None = setup._suggest_param_or_default(
            "matmul_precision"
        )
        if matmul_precision:
            torch.set_float32_matmul_precision(matmul_precision)
        model_config.cache = kwargs.get("set_cache")
        log_root: Path | str | None = kwargs.get("intermediate_out", None)
        if log_root is not None:
            if isinstance(log_root, str):
                log_root = Path(log_root)
            log_root = log_root.joinpath(
                str(trial.number)
            )  # This is supposed to be unique...
            log_root.mkdir(exist_ok=True)

        if filter != -1:
            adata = filter.fit_transform(adata)
        if transformer != -1:
            adata = transformer.fit_transform(adata)
        train, test = train_test_split_ad(adata, test_size=kwargs.get("test_size", 0.1))
        vals = []
        trainer_params = {
            "max_epochs": n_epochs,
            "precision": setup._suggest_param_or_default("precision"),
        }
        if do_splits:
            result: dict = holdout(
                model_cls=model_cls,
                model_config=model_kwargs,
                trainer_kwargs=trainer_params,
                data=train,
                in_features=n_features,
                n_classes=n_classes,
                logger_fn=lambda x: self.log_fn(f"{trial.number}-holdout_{x}"),
                to_encode=self.label_col,
                split_fns=split_fns,
                device=device,
                split_masks=split_masks,
                minimal=True,
                verbose=False,
            )
            vals.append(np.mean(result.values()))
        train_set = d_ut.AnnDataset(
            train,
            to_encode=self.label_col,
            device=device,
        )
        valid_set = d_ut.AnnDataset(test, device=device)

        callbacks = kwargs.get("callbacks", []).extend(opts.get("callbacks", []))
        if do_cv:
            if log_root is not None:
                cv_out = log_root.joinpath("cv")
                cv_out.mkdir(exist_ok=True)
            else:
                cv_out = None
            cv_results = cross_validate(
                model_cls=model_cls,
                model_kwargs=model_kwargs,
                model_config=model_config,
                trainer_kwargs=trainer_params,
                callbacks=callbacks,
                save_path=cv_out,
                adset=train_set,
                logger_fn=lambda x: self.log_fn(f"{trial.number}-cv_fold_{x}"),
                validation=valid_set,
                device=device,
                in_features=n_features,
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
