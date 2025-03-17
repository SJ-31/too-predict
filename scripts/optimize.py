#!/usr/bin/env ipython

import joblib
import optuna
from too_predict.optimization import get_artifact_store, get_options, nested_optuna, run
from too_predict.utils import get_data, training_data_internal

#
#
trial_options = get_options(None)  # Default options for now


#
def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--n_inner", default=3)
    parser.add_argument("-r", "--run_name", default="run", help="", action="store")
    parser.add_argument("-l", "--label_col", default="tumor_type")
    parser.add_argument("-o", "--n_outer", default=10)
    args = vars(parser.parse_args())
    return args


# See https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
# for how to parallelize

if __name__ == "__main__":
    args = parse_args()
    lc = args["label_col"]
    adata = training_data_internal(label=lc)
    journal_dir = get_data(".optuna_journals").joinpath(args["run_name"])
    journal_dir.mkdir(exist_ok=True, parents=True)
    artifact_dir = get_data(".optuna_artifactstore").joinpath(args["run_name"])
    artifact_dir.mkdir(exist_ok=True, parents=True)

    nested_optuna(
        adata,
        score_fn="???",
        n_outer=args["n_outer"],
        n_inner=args["n_inner"],
        label_col=lc,
        save_model=True,
        save_cv=True,
        journal_dir=journal_dir,
        artifact_dir=artifact_dir,
    )
