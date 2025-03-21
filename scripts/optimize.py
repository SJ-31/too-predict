#!/usr/bin/env ipython

from functools import partial

import joblib
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from too_predict.optimization import (
    Optimizer,
    get_artifact_store,
    get_options,
    ignore_duplicated,
    nested_optuna,
    objective,
)
from too_predict.utils import get_data, training_data_internal

#
#
trial_options = get_options(None)  # Default options for now


#
def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--n_inner", default=10)
    parser.add_argument("-r", "--run_name", default="run", help="", action="store")
    parser.add_argument("-l", "--label_col", default="tumor_type")
    parser.add_argument("-o", "--n_outer", default=5)
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

    n_inner = args["n_inner"]
    pruner = HyperbandPruner(min_resource=1, max_resource=n_inner, reduction_factor=4)

    # [2025-03-17 Mon] With this set, maybe you can do more inner cv  loops with aggressive
    # pruning. Will be 2 brackets with the above args and n_inner = 10
    #
    def get_sampler(seed):
        return TPESampler(
            multivariate=True, constant_liar=True, consider_endpoints=True, seed=seed
        )

    # Their default gamma is
    # def default_gamma(x: int) -> int:
    # return min(int(np.ceil(0.1 * x)), 25)
    # Low noise in objective due to CV
    search = Optimizer(
        score_fn="???",
        label_col=lc,
        save_cv=True,
        save_model=True,
        journal_dir=journal_dir,
        artifact_dir=artifact_dir,
    )

    results = search.nested(
        adata,
        n_outer=args["n_outer"],
        n_inner=n_inner,
        pruner=pruner,
        sampler_fn=get_sampler,
    )
