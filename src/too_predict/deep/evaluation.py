#!/usr/bin/env ipython

import math
from collections.abc import Callable, Sequence
from pathlib import Path

import anndata as ad
import lightning as L
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import too_predict.deep.torch_utils as d_ut
import too_predict.evaluation as te
import too_predict.utils as ut
import torch
from lightning.pytorch.loggers import Logger
from too_predict.deep.distillation import TeacherResponse, use_kd_criterion
from too_predict.deep.metrics import (
    multitask_acc,
    multitask_all_metrics,
    multitask_metrics2df,
)
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split


def train_test_split_torch(
    dataset: Dataset,
    train_size: float | int | None = None,
    test_size: float | int = 0.25,
    valid: float | int | None | bool = None,
    as_dataloader: bool = True,
    **kwargs,
) -> tuple[DataLoader, ...] | tuple[Dataset, ...]:
    """Get train, test and optionally validation splits from dataset

    Parameters
    ----------
    train_size : count or proportion of train dataset
    test_size : count or proportion of test dataset
    valid_size : if a boolean True, valid_size is interpreted as
        valid_size = 1 - train_size - test_size (if both are given)
        valid_size = test_size (if train_size is not given)

    Returns
    -------
    Tuple of DataLoaders
    """
    n_samples = len(dataset)
    if isinstance(train_size, int):
        train_size = math.floor(train_size / n_samples)
    if isinstance(test_size, int):
        test_size = math.floor(test_size / n_samples)
    if (
        not isinstance(valid, bool)
        and valid is not None
        and not isinstance(valid, float)
    ):
        valid = math.floor(valid / n_samples)

    if not test_size:
        raise ValueError("A test split must be provided!")

    # Only test size provided, valid True
    if isinstance(valid, bool) and valid and train_size is None:
        valid = test_size
        lengths = (1 - test_size - valid, test_size, valid)
    # Only test provided, valid float
    elif isinstance(valid, float) and train_size is None:
        lengths = (1 - test_size - valid, test_size, valid)
    # Test and train provided, valid True
    elif isinstance(valid, bool) and valid and train_size:
        valid = 1 - train_size - test_size
        lengths = (train_size, test_size, valid)
    # Test size provided, valid False|None
    elif test_size and train_size is None and not valid:
        lengths = (1 - test_size, test_size)
    # Test and train provided, valid False|None
    elif train_size and test_size and not valid:
        lengths = (train_size, test_size)
    # All provided
    else:
        lengths = (train_size, test_size, valid)

    print(f"Split proportions: {lengths}")
    if np.sum(lengths) > 1:
        raise ValueError("Invalid split sizes provided!")

    if as_dataloader:
        return tuple(DataLoader(d, **kwargs) for d in random_split(dataset, lengths))
    return tuple(d for d in random_split(dataset, lengths))


def train_test_wrapper_torch(
    module_cls: d_ut.MultiModule,
    train_test: Sequence[Dataset],
    to_encode: Sequence[str] | str,
    n_classes: Sequence[int],
    in_features: int,
    module_kwargs: dict | None = None,
    trainer_kwargs: dict | None = None,
    loader_kwargs: dict | None = None,
    logger_fn: None | Callable = None,
    validation: Dataset | None = None,
    module_config: d_ut.ModuleConfig | None = None,
    scaler: d_ut.TorchScaler | None = None,
    device: str = "cpu",
    grad_accumulation: bool = False,
    grad_accumulation_batch_size: int = 256,
    minimal: bool = False,
    set_label: str = "model",
):
    """
    Parameters
    ----------
    module_cls : d_ut.MultiModule
        A model class compatible with PyTorch Lightning that provides a
        `.new()` constructor for instantiation.
    maybe_split : Callable or Sequence
        If `pre_split` is False, either a callable that splits `adata`
        into train/test sets or a sequence of index arrays.
        If `pre_split` is True, directly provide `(train, test)` `AnnData`
        objects.
    to_encode : str or sequence of str
        Names of observation columns in `AnnData` to be used as labels
        for supervised learning tasks.
    n_classes : sequence of int
        Number of classes per task, matching the length of `to_encode`.
    in_features : int
        Number of input features for the model.
    module_kwargs : dict, optional
        Additional keyword arguments passed to `module_cls.new()`.
    trainer_kwargs : dict, optional
        Keyword arguments forwarded to `pytorch_lightning.Trainer`.
    loader_kwargs : dict, optional
        Keyword arguments forwarded to `torch.utils.data.DataLoader`.
    logger_fn : callable, optional
        Callable that accepts a model label and returns a logger instance.
    validation : ad.AnnData, optional
    module_config : d_ut.ModuleConfig, optional
        Configuration object for the model.
    scaler : d_ut.TorchScaler, optional
        Optional feature scaler applied before training.
    device : str, default="cpu"
        Device string (`"cpu"`, `"cuda"`, etc.).
    grad_accumulation : bool, default=False
        Whether to use gradient accumulation during training.
    grad_accumulation_batch_size : int, default=256
        Effective batch size if gradient accumulation is enabled.
    minimal : bool, default=False
        If True, return only train/test accuracies. If False, return a
        full set of metrics.
    set_label : str, default="model"
        Label used for saving outputs and logging.

    Returns
    -------
    dict
        If `minimal=False`, returns a dictionary of multitask evaluation
        metrics (`multitask_all_metrics`).
    (float, float)
        If `minimal=True`, returns `(test_accuracy, train_accuracy)`.

    Notes
    -----
    - Requires `pytorch_lightning`, `torch`, and `anndata`.
    - Splits can be saved to disk for reproducibility if `save_split_path`
      is provided.
    - For multitask learning, multiple labels in `to_encode` are supported
    """
    loader_kwargs = ut.if_none(loader_kwargs, {})
    module_config = d_ut.ModuleConfig() if module_config is None else module_config
    module_kwargs = ut.if_none(module_kwargs, {})
    if logger_fn is not None:
        trainer_kwargs["logger"] = logger_fn(set_label)
    x_train, x_test = train_test
    v_loader = DataLoader(validation) if validation is not None else None
    updated_loader_kwargs, updated_train_kwargs = d_ut.update_batch_strategy(
        loader_kwargs=loader_kwargs,
        dataset=x_train,
        default_batch_size=512,
        trainer_kwargs=trainer_kwargs,
        grad_accumulation=grad_accumulation,
        grad_accumulation_batch_size=grad_accumulation_batch_size,
    )
    trainer = L.Trainer(**updated_train_kwargs)
    train_l = DataLoader(x_train, **updated_loader_kwargs)
    x_test_tensor, y_true = x_test[:]
    if scaler is not None:
        scaler.fit(train_l.dataset[:][0])
        module_config.scaler = scaler
    module_config.init_device = device
    with trainer.init_module():
        model = module_cls.new(
            in_features=in_features,
            n_classes_per_task=n_classes,
            conf=module_config,
            **module_kwargs,
        )
        if isinstance(x_train, TeacherResponse):
            use_kd_criterion(model)
    trainer.fit(model=model, train_dataloaders=train_l, val_dataloaders=v_loader)
    model.to(device)
    if not minimal:
        proba = model.predict_proba(x_test_tensor)
        res: dict = multitask_all_metrics(
            y_true=y_true,
            scores=proba,
            task_names=to_encode,
            n_classes=n_classes,
        )
        return res

    acc_kwargs = {"n_classes": n_classes, "task_names": to_encode}
    test_acc = multitask_acc(
        y_true=y_true, predictions=model.predict_step(x_test[:]), **acc_kwargs
    )
    train_acc = multitask_acc(
        y_true=x_train[:][1],
        predictions=model.predict_step(x_train[:]),
        **acc_kwargs,
    )
    return test_acc, train_acc


# TODO: give this params for working with distillation
def holdout(
    data: ad.AnnData | dict[str, tuple[ad.AnnData, ad.AnnData]],
    split_fns: dict[str, Callable[[ad.AnnData], tuple[ad.AnnData, ad.AnnData]]]
    | None = None,
    split_masks: dict[str, tuple] | None = None,
    outdir: Path | None = None,
    **kwargs,
) -> dict:
    """Wrapper function for doing the classic holdout method (train-test-split) with
    torch module

    Parameters
    ---------
    Same parameters as original holdout
    split_fn : A dictionary of function that splits adata into a tuple of train, test
    split_indices : A dictionary of mapping test set names to (train, test) boolean indices
    validation : float | None
        If not none, the percentage of the test split to use for validation data in
        early stopping and reporting training metrics
    logger_fn : Optional function taking name of holdout set and returning Lightning
        logger

    Returns
    ------
    A dictionary containing model evaluation results for each task i.e.
        split->task->results_dict
    If `minimal`, is a dictionary of split->dict[task->accuracy]
    """
    if (split_fns is None and split_masks is None) and isinstance(data, ad.AnnData):
        raise ValueError("Either split_fns or split_indices must be given!")
    elif (split_fns is None and split_masks is None) and isinstance(data, dict):
        pre_split: bool = True
    else:
        pre_split = False

    result: dict = {}
    if split_fns is None and split_masks is None:
        iter_over = data
    elif split_fns:
        iter_over = {k: split(data) for k, split in split_fns.items()}
    else:
        iter_over = {k: (data[s[0], :], data[s[1], :]) for k, s in split_masks.items()}
    for set_label, val in iter_over.items():
        result[set_label] = train_test_wrapper_torch(
            set_label=set_label,
            train_test=val,
            adata=data if not pre_split else None,
            **kwargs,
        )
        if outdir is not None and not kwargs.get("minimal"):
            cur_outdir = outdir.joinpath(set_label)
            cur_outdir.mkdir(exist_ok=True)
            for task in kwargs["to_encode"]:
                te.write_cross_val(
                    result[set_label][task], outdir=cur_outdir, prefix=f"{task}_"
                )
    if outdir is not None and kwargs.get("minimal"):
        tasks = list(next(iter(result.values()))[0].keys())
        to_df = {"set": result.keys()}
        for t in tasks:
            to_df[f"{t}_test_acc"] = [v[0][t] for v in result.values()]
            to_df[f"{t}_train_acc"] = [v[1][t] for v in result.values()]
        df = pd.DataFrame(to_df)
        df = d_ut.tensor_cols_to_float(df)
        df.to_csv(outdir.joinpath("accuracy.csv"), index=False)
    return result


def cross_validate(
    model_cls: d_ut.MultiModule,
    trainer_kwargs: dict,
    adset: d_ut.AnnDataset | TeacherResponse,
    n_classes: Sequence[int],
    in_features: int,
    model_config: d_ut.ModuleConfig | None = None,
    model_kwargs: dict | None = None,
    random_state: int | None = ut.RANDOM_STATE,
    callbacks: list[Callable[[], L.Callback]] | None = None,
    n_splits: int = 5,
    validation: Dataset | None = None,
    verbose: bool = False,
    logger_fn: Callable[[int], Logger] | None = None,
    save_path: Path | None = None,
    scaler: d_ut.TorchScaler | None = None,
    with_train_acc: bool = True,
    device: str = "cpu",
    init_bias: bool = True,
    grad_accumulation: bool = False,
    grad_accumulation_batch_size: int = 256,
    minimal: bool = True,
    **kwargs,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Run cross-validation

    Parameters
    ----------
    trainer_kwargs : Keyword arguments passed to Lightning trainer
    logger_fn : Optional function taking in the fold number and returning a Lightning
        logger
    kwargs : key-word arguments passed to DataLoader. Set batch_size == -1 to use the entire
        training set in each cross-validation loop

    Returns
    -------
    Dataframe summarizing cv accuracies
    """
    if verbose:
        print("Beginning cross validation...")
    cv = ms.KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    splits = cv.split(adset)
    dfs: list = []

    model_config = model_config if model_config else d_ut.ModuleConfig()
    tasks = adset.label_cols
    cms: dict = {t: [] for t in tasks} if not minimal else {}
    if init_bias:
        model_config.out_bias = d_ut.get_initial_out_bias(
            model_config.outlayer_type, adset
        )
    if scaler is not None:
        model_config.scaler = scaler
    if isinstance(adset, TeacherResponse):
        distillation: bool = True
        labels: Tensor | None = adset.get_targets()
    else:
        distillation = False
        labels = None
    model_kwargs = ut.if_none(model_kwargs, {})
    for fold, (train_idx, test_idx) in enumerate(splits):
        if verbose:
            print(f"fold {fold} started")
        train_set = Subset(adset, train_idx)
        updated_kwargs, updated_trainer_kwargs = d_ut.update_batch_strategy(
            loader_kwargs=kwargs,
            dataset=train_set,
            trainer_kwargs=trainer_kwargs,
            grad_accumulation=grad_accumulation,
            grad_accumulation_batch_size=grad_accumulation_batch_size,
        )
        train: DataLoader = DataLoader(train_set, **updated_kwargs)
        test: Dataset = Subset(adset, test_idx)
        val_loader = DataLoader(validation) if validation is not None else None

        if save_path is not None:
            root = save_path.joinpath(f"fold_{fold}")
            root.mkdir(exist_ok=True)
            updated_trainer_kwargs["default_root_dir"] = root

        if logger_fn is not None:
            updated_trainer_kwargs["logger"] = logger_fn(fold)

        if scaler is not None:
            scaler.fit(train.dataset[:][0])

        if callbacks is not None:
            updated_trainer_kwargs["callbacks"] = [c() for c in callbacks]

        model_config.init_device = device
        trainer = L.Trainer(**updated_trainer_kwargs)
        with trainer.init_module():
            model = model_cls.new(
                in_features=in_features,
                n_classes_per_task=n_classes,
                conf=model_config,
                **model_kwargs,
            )
            if distillation:
                use_kd_criterion(model)
            if init_bias:
                d_ut.init_lazy(model, loader=train)
                model.init_out_bias()  # Targets provided with whole dataset above
        trainer.fit(model=model, train_dataloaders=train, val_dataloaders=val_loader)
        model = model.to(torch.device(device))

        if not isinstance(adset, TeacherResponse):
            test_y = test[:][1]
            train_y = train.dataset[:][1]
        else:
            test_y = labels[test_idx, :]
            train_y = labels[train_idx, :]

        if minimal:
            acc = multitask_acc(
                y_true=test_y,
                predictions=model.predict_step(test[:]),
                task_names=tasks,
                n_classes=n_classes,
                as_df=True,
            )
            dfs.append(acc.assign(fold=fold, context="test"))
            if with_train_acc:
                train_acc = multitask_acc(
                    y_true=train_y,
                    predictions=model.predict_step(train.dataset[:]),
                    task_names=tasks,
                    n_classes=n_classes,
                    as_df=True,
                )
                dfs.append(train_acc.assign(fold=fold, context="train"))
        else:
            test_metrics = multitask_all_metrics(
                y_true=test_y,
                scores=model.predict_proba(test[:][0]),
                task_names=tasks,
                n_classes=n_classes,
            )
            dfs.append(
                multitask_metrics2df(test_metrics).assign(context="test", fold=fold)
            )
            for t in tasks:
                cms[t].append(test_metrics[t]["cm"])
            if with_train_acc:
                train_metrics = multitask_all_metrics(
                    y_true=train_y,
                    scores=model.predict_proba(train.dataset[:][0]),
                    task_names=tasks,
                    n_classes=n_classes,
                )
                dfs.append(
                    multitask_metrics2df(train_metrics).assign(
                        context="train", fold=fold
                    )
                )
    if verbose:
        print("Cross validation completed")
    if minimal:
        return pd.DataFrame(pd.concat(dfs))
    return pd.DataFrame(pd.concat(dfs)), cms


# * Test functions


def init_test(model: d_ut.MultiModule, loader: DataLoader) -> Tensor | tuple:
    d_ut.init_lazy(model, loader)
    biases = d_ut.get_initial_out_bias(model.conf.outlayer_type, loader)
    model.conf.out_bias = biases
    model.init_out_bias()
    return model.predict_proba(loader), biases


def random_softmax_loss(
    model: d_ut.MultiModule,
    trainer: L.Trainer,
    train: DataLoader,
    test: DataLoader,
):
    x, y = train.dataset[:]
    rand_indices = torch.randperm(y.shape[0])
    y = y[rand_indices]
    shuffled_train = TensorDataset(x, y)
    if train.batch_size:
        new_loader = DataLoader(shuffled_train, batch_size=train.batch_size)
    elif train.batch_sampler:
        new_loader = DataLoader(shuffled_train, batch_sampler=train.batch_sampler)
    else:
        new_loader = DataLoader(shuffled_train)
    trainer.fit(model, train_dataloaders=new_loader)
    train_result = trainer.test(model, dataloaders=train)
    test_result = trainer.test(model, dataloaders=test)
    return train_result, test_result
