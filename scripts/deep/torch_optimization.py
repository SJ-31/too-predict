#!/usr/bin/env ipython

from pathlib import Path

import joblib
import pandas as pd
import too_predict.utils as ut
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pyhere import here
from too_predict._train_utils import ADDITIONAL_SPLITS
from too_predict.deep.callbacks import AverageBest
from too_predict.deep.logistic import MultiLevel
from too_predict.deep.optimization import DlOptimizer
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.transformer import Transformer

if str(Path.home()) == "/home/shannc":
    OPTUNA_JOURNALS = here("data", "tests", "optuna_journaldir")
    OPTUNA_STORAGE = here("data", "tests", "optuna_artifacts")
else:
    OPTUNA_JOURNALS = here("remote", "repos", "too-predict", "optuna_journals")
    OPTUNA_STORAGE = here("remote", "repos", "too-predict", "optuna_artifactstore")
OUTDIR = here("data", "output", "optimization")
REF, FEAT = ut.ref_feature_lists_internal()

torch.set_default_dtype(torch.float32)

if torch.cuda.is_available():
    torch.set_default_device("cuda:0")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", default=False, help="", action="store_true")
    args = vars(parser.parse_args())  # convert to dict
    return args


if __name__ == "__main__":
    args = parse_args()
    TEST = "_test" if args["test"] else ""
else:
    TEST = ""


@ut.SaveOrLoad(
    out={
        "trial_df": here(OUTDIR, f"torch_choose_optimization{TEST}.csv"),
        "study_obj": here(OUTDIR, f"torch_choose_optimization{TEST}.pkl"),
    },
    read_fn={"trial_df": pd.read_csv, "study_obj": ut.load_pickle},
    logdir=here("log"),
)
def choose_optimization(dct, adata) -> None:
    journal_file = here(OPTUNA_JOURNALS, f"torch_select_optimizer{TEST}.log")
    artifact_dir = here(OPTUNA_STORAGE, f"torch_optimization{TEST}")
    if TEST == "_test":
        filter = -1
        n_epochs = 5
    else:
        filter = Filter(
            features=FEAT["edgeR_median_lfc_feature_list_3000"],
            inplace=False,
            feature_col="GENEID",
        )
        n_epochs = 1000
    sample_opts = {
        # Module arguments
        "module": "Disyak",
        "dropout": [0.2, 0.5],
        "l2_pars": "none",
        "task_weights": torch.tensor([1, 1.2]),
        "l1_pars": [{"lambda": 0.001}, {"lambda": 0.01}],
        "n_hidden": [1000, 2000, None],
        # Optimization
        "n_epochs": n_epochs,
        "optimizer": "Adam",
        "betas": [(0.9, 0.999), (0.7, 0.888)],  # Adam
        "amsgrad": [True, False],  # Adam
        "weight_decay": [0, 0.01, 0.001, 0.0001],  # Adam, SGD
        "lr": 0.001,  # All optimizers
        "momentum": [0, 0.9],  # SGD
        # Extras
        "transformer": Transformer("clr", impute_fn=Imputer("plus_one"), inplace=False),
        # Try without filtering first
        # [2025-06-18 Wed] Ask Aj though
        "filter": filter,
        # Scheduling
        "scheduler": ["ReduceLROnPlateau", "PolynomialLR"],
        # PolynomialLR
        "power": 0.5,
        "total_iters": 5,
        # ReduceLROnPlateau
        "patience": 4,
        "factor": 0.1,
        # CyclicLR
        "mode": ["triangular", "triangular2"],
    }
    searcher = DlOptimizer(
        label_col=("tumor_type", "Sample_Type"),
        journal_file=journal_file,
        artifact_dir=artifact_dir,
    )
    cv_output = OUTDIR.joinpath("cv_output")
    cv_output.mkdir(exist_ok=True)
    OUTDIR.mkdir(exist_ok=True)
    searcher.make_objective(
        adata=adata,
        opts=sample_opts,
        split_fns=ADDITIONAL_SPLITS,
        do_splits=False,
        do_cv=True,
        cv_splits=3,
        save_intermediate=True,
        intermediate_out=cv_output,
        verbose=TEST != "",
        set_cache=["val_acc"],
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=40, mode="min"),
            AverageBest(n_best=10, target="val_acc"),
        ],
        batch_size=1024,
    )
    study = searcher.run_study(
        study_name="optimizer_selection", directions=["maximize", "maximize"]
    )
    joblib.dump(study, dct["study_obj"])
    df = study.trials_dataframe()
    df.to_csv(dct["trial_df"], index=False)
    return


@ut.SaveOrLoad(
    out={
        "study_obj": here(OUTDIR, "torch_choose_epochs.pkl"),
        "trial_df": here(OUTDIR, "torch_choose_epochs.csv"),
    },
    read_fn={"trial_df": pd.read_csv, "study_obj": ut.load_pickle},
    logdir=here("log"),
)
def choose_epochs(dct, adata) -> None:
    print("Choosing epochs...")
    journal_file = here(OPTUNA_JOURNALS, "torch_n_epochs.log")
    artifact_dir = here(OPTUNA_STORAGE, "torch_epochs")
    sample_opts = {
        "module": lambda **kwargs: MultiLevel(**kwargs),
        "optimizer": "Adam",
        "betas": [(0.9, 0.999)],
        "amsgrad": False,
        "weight_decay": 1,
        "lr": 1e-4,  # All optimizers
        "transformer": -1,
        "filter": -1,
        "n_epochs": [1000, 500],
        "scheduler": "ReduceLROnPlateau",
        "patience": 4,
        "factor": 0.1,
    }
    searcher = DlOptimizer(
        label_col=("tumor_type", "Sample_Type"),
        journal_file=journal_file,
        artifact_dir=artifact_dir,
    )
    searcher.make_objective(
        adata=adata,
        opts=sample_opts,
        split_fns=ADDITIONAL_SPLITS,
        do_splits=False,
        at_batch_level=False,
        test_size=0.1,
    )
    study = searcher.run_study(
        study_name="best_n_epochs", directions=["maximize", "maximize"]
    )
    joblib.dump(study, dct["study_obj"])
    df = study.trials_dataframe()
    df.to_csv(dct["trial_df"], index=False)
    return


if __name__ == "__main__":
    args = parse_args()
    if args["test"]:
        print("Using test set")
        adata = ut.training_data_internal_test(minimal=True)
    else:
        adata = ut.training_data_internal()
    choose_optimization(adata)  # TODO: run these
