#!/usr/bin/env ipython

import joblib
import optuna
from too_predict.optimization import get_options, nested_optuna, run
from too_predict.utils import get_data, training_data_internal

#
#
trial_options = get_options(None)  # Default options for now


#
def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--n_inner", default=3)
    parser.add_argument("-l", "--label_col", default="tumor_type")
    parser.add_argument("-o", "--n_outer", default=10)
    args = vars(parser.parse_args())  # convert to dict
    return args


if __name__ == "__main__":
    args = parse_args()
    lc = args["label_col"]
    adata = training_data_internal(label=lc)
    journaldir = get_data(".optuna_journals")
    nested_optuna(
        adata,
        score_fn="???",
        n_outer=args["n_outer"],
        n_inner=args["n_inner"],
        label_col=lc,
        save_model=True,
        save_cv=True,
        storagedir=journaldir,
    )
