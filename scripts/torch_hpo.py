#!/usr/bin/env ipython

from pathlib import Path

import anndata as ad
import joblib
import too_predict._train_utils as tt
import too_predict.deep.torch_utils as d_ut
import too_predict.utils as ut
import torch
from too_predict.deep.optimization import DlOptimizer

try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = ut.DummySnake(
        rule="choose_optimization",
        configfile="env.yaml",
        input="../data/tests/adatas/optuna/optimze.h5ad",
        output={
            "trial_df": "../data/output/tests/optuna_df.csv",
            "study_obj": "../data/output/tests/optuna.pkl",
        },
        params={
            "storage_file": "../data/output/tests/optuna.db",
            "artifact_dir": "../data/output/tests/",
        },
    )
    smk.config["test"] = True

torch.set_default_dtype(torch.float32)

LABELS = smk.config["multi_labels"]
DL_CONFIG: dict = smk.config["defaults"]["dl"]
N_REPEATS = smk.config["cv_n_repeats"]
TEST: bool = smk.config["test"]


FILTER, TRANSFORM = tt.default_filter_transform(smk.config)

MODELS = smk.config["models"]["dl"]


if smk.rule == "preprocess":
    if TEST:
        adata = ut.training_data_internal_test(minimal=True)
    else:
        adata = ut.training_data_internal()
    filter, transform = tt.default_filter_transform(smk.config)
    if DL_CONFIG["filter"]:
        adata = filter.fit_transform(adata)
    adata = transform.fit_transform(adata)
    adata.write_h5ad(str(smk.output))
elif smk.rule == "main":
    hpo_task = smk.config["hpo_task"]
    default_opts = DL_CONFIG["hpo"].copy()
    default_opts["matmul_precision"] = DL_CONFIG["matmul_precision"]
    default_opts["precision"] = DL_CONFIG["trainer"]["precision"]
    default_opts["lr"] = DL_CONFIG["optimizer"]["lr"]
    if hpo_task == "adam":
        changes = {
            "l1_pars": [{"lambda": 0.001}, {"lambda": 0.01}],
            "betas": [(0.9, 0.999), (0.7, 0.888)],
            "amsgrad": [True, False],
            "weight_decay": [0, 0.01, 0.001, 0.0001],
            "momentum": [0, 0.9],
        }
    elif hpo_task == "scheduling":
        changes = {
            "scheduler": ["ReduceLROnPlateau", "PolynomialLR", "BatchSizeScaler"],
            "mode": ["triangular", "triangular2"],
            "bs_factor": 5,
            "bs_total_iters": 10,
        }
    elif hpo_task == "task_weights":
        changes = {
            "task_weights": [(1, 2), (1, 4), (1, 8)],
        }
    elif hpo_task == "precision":
        changes = {
            "precision": ["32-true", "16-mixed"],
            "matmul_precision": ["high", "medium", "highest"],
        }
    else:
        raise ValueError("No task specified!")

    default_opts.update(changes)
    adata = ad.read_h5ad(str(smk.input), backed=True)
    date = smk.params["date"]
    searcher = DlOptimizer(
        label_col=LABELS,
        storage_file=smk.params["storage_file"],
        artifact_dir=smk.params["artifact_dir"],
        log_fn=lambda x: d_ut.lightning_logger(
            x,
            platform="tensorboard",
            save_dir=str(smk.output["log"]),
        ),
    )
    searcher.make_objective(
        adata=adata,
        opts=default_opts,
        do_splits=smk.config["do_holdout"],
        do_cv=smk.config["do_cv"],
        cv_splits=DL_CONFIG["cv"]["n_splits"],
        device=DL_CONFIG["device"],
        verbose=TEST != "",
        set_cache=["val_acc"],
        callbacks=tt.smk_callbacks(DL_CONFIG),
        batch_size=DL_CONFIG["cv"]["batch_size"],
    )
    path = Path(smk.params["storage_file"]).resolve()
    study = searcher.run_study(
        study_name="optimizer_selection",
        directions=["maximize", "maximize"],
        storage=f"sqlite:///{path}",
    )
    joblib.dump(study, smk.output["study_obj"])
    df = study.trials_dataframe()
    df.to_csv(smk.output["df"], index=False)
