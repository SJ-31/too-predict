#!/usr/bin/env ipython

import os
from pathlib import Path

import anndata as ad
import pandas as pd
import too_predict._train_utils as tt
import too_predict.deep.torch_utils as d_ut
import too_predict.utils as ut
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
from lightning.pytorch.callbacks import EarlyStopping
from too_predict._train_utils import get_callback_fn, get_model_fn
from too_predict.deep.callbacks import AverageBest
from too_predict.deep.evaluation import cross_validate

try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = ut.DummySnake(rule="my_rule", configfile="my_config")

torch.set_default_dtype(torch.float32)


LABELS = smk.config["multi_labels"]
DL_CONFIG = smk.config["defaults"]["dl"]
N_REPEATS = smk.config["cv_n_repeats"]
TEST: bool = smk.config["test"]

if (mlp := DL_CONFIG["matmul_precision"].lower()) != "none":
    torch.set_float32_matmul_precision(mlp)


FILTER, TRANSFORM = tt.default_filter_transform(smk.config)
# TODO: the transformation NEEDS to be a hyperparameter that you optimize for

MODELS = smk.config["models"]["dl"]


def opt_fn(pars):
    return optim.Adam(pars, **DL_CONFIG["optimizer"])


def get_scheduler(optimizer):
    return schedule.ReduceLROnPlateau(optimizer, **DL_CONFIG["schedule"])


def cross_val(in_file: str):
    adata = ad.read_h5ad(in_file, backed=True)
    model = Path(in_file).stem
    model_kwargs = MODELS[model].get("params", {})
    model_fn = get_model_fn(model, model_kwargs)
    n_features, n_classes = d_ut.data_spec(adata, y=LABELS)
    train, valid = ut.train_test_split_ad(
        adata, test_size=0.1, random_state=ut.RANDOM_STATE
    )
    train_set = d_ut.AnnDataset(
        train, to_encode=LABELS, device="cpu" if TEST else DL_CONFIG["device"]
    )
    valid_set = d_ut.AnnDataset(
        valid, to_encode=LABELS, device="cpu" if TEST else DL_CONFIG["device"]
    )
    outdir = Path(smk.params["outdir"]).joinpath(model)
    trainer_kwargs = DL_CONFIG["trainer"].copy()
    callbacks = []
    for callback, dct in DL_CONFIG["callbacks"]:
        if dct.get("enabled"):
            callbacks.append(get_callback_fn(callback, dct["params"]))
    trainer_kwargs["fast_dev_run"] = smk.config["dev_run"]
    if TEST:
        trainer_kwargs["log_every_n_steps"] = 1
        trainer_kwargs["max_epochs"] = 10
    model_kwargs = {
        "n_classes_per_task": n_classes,
        "in_features": n_features,
        "cache": "val_acc",
        "scheduler_fn": get_scheduler,
        "optimizer_fn": opt_fn,
    }
    dfs = []
    for i in range(N_REPEATS):
        cv: pd.DataFrame = cross_validate(
            model_fn=model_fn,
            model_kwargs=model_kwargs,
            callbacks=callbacks,
            logger_fn=lambda x: d_ut.lightning_logger(
                f"fold_{x}_repeat_{i}",
                platform="tensorboard",
                save_dir=outdir.joinpath("tensorboard"),
            ),
            trainer_kwargs=trainer_kwargs,
            adset=train_set,
            n_classes=n_classes,
            validation=valid_set,
            device=DL_CONFIG["device"],
            **DL_CONFIG["cv"],
        )
        cv.loc[:, "repeat"] = i
        dfs.append(cv)
    pd.concat(dfs).to_csv(outdir.joinpath("cv_results.csv"), index=False)


if smk.rule == "preprocess":
    if TEST:
        adata = ut.training_data_internal_test(minimal=True)
    else:
        adata = ut.training_data_internal()
    filter, transform = tt.default_filter_transform(smk.config)
    for model in smk.output:
        if Path(model).exists():
            continue
        name = Path(model).stem
        spec = smk.config["models"]["dl"].get(name)
        if spec.get("filter"):
            adata = filter.fit_transform(adata)
        if spec.get("transform"):
            adata = transform.fit_transform(adata)
        adata.write_h5ad(model)
if smk.rule == "cross_validate":
    for f in smk.input:
        cross_val(f)
