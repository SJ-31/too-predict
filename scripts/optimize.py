#!/usr/bin/env ipython

import joblib
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from pyhere import here
from too_predict._train_utils import ADDITIONAL_SPLITS
from too_predict.model import PredBase, XGBEstimator
from too_predict.optimization import Optimizer
from too_predict.utils import get_data, training_data_internal


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--n_inner", default=10)
    parser.add_argument("-r", "--run_name", default="run", help="", action="store")
    parser.add_argument("-l", "--label_col", default="tumor_type")
    parser.add_argument("-o", "--n_outer", default=5)
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-c", "--cores", default=8, type=int)
    args = vars(parser.parse_args())
    return args


def optimize_hp(label_col, journal_dir, artifact_dir):
    searcher = Optimizer(
        score_fn="???",
        label_col=label_col,
        save_cv=True,
        save_model=True,
        journal_file=journal_dir,
        artifact_dir=artifact_dir,
    )

    results = searcher.nested(
        adata,
        n_outer=args["n_outer"],
        n_inner=n_inner,
        pruner=pruner,
        sampler_fn=get_sampler,
    )


def choose_feature_set_run(adata, label_col):
    optuna_storage = here("remote", "repos", "too-predict")
    fs_journal_file = here(optuna_storage, "optuna_journals", "feature_selection.log")
    fs_artifact_dir = here(optuna_storage, "optuna_artifactstore", "feature_selection")
    fs_artifact_dir.mkdir(exist_ok=True)
    user_opts = {
        "imputation": "plus_one",
        "transformation": "clr",
        "clr_subset": "none",
        "feature_set": [
            "edgeR_15_per_type_ovp_tissue_enriched",
            "edgeR_30_per_type_ovp_tissue_enriched",
            "edgeR_50_per_type_ovp_tissue_enriched",
            "edgeR_70_per_type_ovp_tissue_enriched",
            "edgeR_70_per_type_ovp_",
            "edgeR_30_per_type_ovp_",
            "edgeR_15_per_type_ovp_",
        ],
        "classifier": PredBase(XGBEstimator(max_depth=3)),
    }
    searcher = Optimizer(
        label_col=label_col,
        save_cv=True,
        save_model=False,
        journal_file=fs_journal_file,
        artifact_dir=fs_artifact_dir,
    )
    searcher.make_objective(
        adata=adata, cv_splits=-1, opts=user_opts, split_fns=ADDITIONAL_SPLITS
    )
    study = searcher.run_study(study_name="choosing_feature_set")
    joblib.dump(
        study, here("data", "output", "feature_selection", "best_feature_sets.pkl")
    )


# See https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
# for how to parallelize

if __name__ == "__main__":
    args = parse_args()
    lc = args["label_col"]
    adata = training_data_internal(label=lc)
    # TODO: reconfigure this stuff when it comes to hp optimization
    # journal_dir = get_data(".optuna_journals").joinpath(args["run_name"])
    # journal_dir.mkdir(exist_ok=True, parents=True)
    # artifact_dir = get_data(".optuna_artifactstore").joinpath(args["run_name"])
    # artifact_dir.mkdir(exist_ok=True, parents=True)

    # n_inner = args["n_inner"]
    # pruner = HyperbandPruner(min_resource=1, max_resource=n_inner, reduction_factor=4)

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
    # optimize_hp(label_col=lc, journal_dir=journal_dir, artifact_dir=artifact_dir)
    with joblib.parallel_backend("loky", n_jobs=args["cores"]):
        choose_feature_set_run(adata=adata, label_col=lc)
