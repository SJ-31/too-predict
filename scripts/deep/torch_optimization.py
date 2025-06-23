#!/usr/bin/env ipython

import joblib
import pandas as pd
import too_predict.utils as ut
from pyhere import here
from too_predict._train_utils import ADDITIONAL_SPLITS
from too_predict.deep.optimization import DlOptimizer

OPTUNA_JOURNALS = here("remote", "repos", "too-predict", "optuna_journaldir")
OPTUNA_STORAGE = here("remote", "repos", "too-predict", "optuna_artifactstore")
OUTDIR = here("data", "output", "optimization")


@ut.SaveOrLoad(
    out={
        "trial_df": here(OUTDIR, "torch_choose_optimization.csv"),
        "study_obj": here(OUTDIR, "torch_choose_optimization.pkl"),
    },
    read_fn={"trial_df": pd.read_csv, "study_obj": ut.load_pickle},
    logdir=here("log"),
)
def choose_optimization(dct, adata) -> None:
    journal_file = here(OPTUNA_JOURNALS, "torch_select_optimizer.log")
    artifact_dir = here(OPTUNA_STORAGE, "torch_optimization")
    sample_opts = {
        "optimizer": "Adam",
        "betas": [(0.9, 0.999), (0.7, 0.888)],  # Adam
        "amsgrad": [True, False],  # Adam
        "weight_decay": (1, 0, -2),  # Adam, SGD
        "lr": (),  # All optimizers
        "momentum": (),  # SGD
        "transformer": -1,
        # Try without filtering first
        # [2025-06-18 Wed] Ask Aj though
        "filter": -1,
        #
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
    searcher.make_objective(adata=adata, opts=sample_opts, split_fns=ADDITIONAL_SPLITS)
    study = searcher.run_study(study_name="optimizer_selection")
    joblib.dump(study, dct["study_obj"])
    df = study.trials_dataframe()
    df.write_csv(dct["trial_df"], index=False)
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
    journal_file = here(OPTUNA_JOURNALS, "torch_n_epochs.log")
    artifact_dir = here(OPTUNA_STORAGE, "torch_epochs")
    sample_opts = {
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
    searcher.make_objective(adata=adata, opts=sample_opts, split_fns=ADDITIONAL_SPLITS)
    study = searcher.run_study(study_name="best_n_epochs")
    joblib.dump(study, dct["study_obj"])
    df = study.trials_dataframe()
    df.write_csv(dct["trial_df"], index=False)
    return


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    parser.add_argument("-t", "--test", default=False, help="", action="store_true")
    args = vars(parser.parse_args())  # convert to dict
    return args


if __name__ == "__main__":
    args = parse_args()
    if args["test"]:
        adata = ut.training_data_internal_test(minimal=True)
    else:
        adata = ut.training_data_internal()
    # choose_optimization(adata) # TODO: run these
    # choose_epochs(adata) # TODO: run these
