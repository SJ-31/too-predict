#!/usr/bin/env ipython

from functools import partial

import joblib
import numpy as np
import optuna
import optuna.artifacts as oa
import optuna.storages.journal as oj
import pandas as pd
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from imblearn.ensemble import BalancedRandomForestClassifier
from optuna.samplers import TPESampler
from pyhere import here
from sklearn.ensemble import RandomForestClassifier
from too_predict.evaluation import holdout, summarize_studies
from too_predict.filter import Filter
from too_predict.imbalance import Balancer, spaced_resample
from too_predict.imputer import Imputer
from too_predict.model import PredBase, XGBEstimator
from too_predict.optimization import ignore_duplicated
from too_predict.transformer import Transformer
from too_predict.utils import (
    RANDOM_STATE,
    ref_feature_lists_internal,
    training_data_internal,
    training_data_internal_test,
    write_pickle,
)

SPLITS = {
    "CHULA": lambda x: (
        x[~x.obs["Project_ID"].str.contains("CHULA"), :],
        x[x.obs["Project_ID"].str.contains("CHULA"), :],
    ),
}

store_dir = here("remote", "repos", "too-predict", "optuna_artifactstore", "balancing")
store_dir.mkdir(exist_ok=True)
journal_path = here(
    "remote", "repos", "too-predict", "optuna_journals", "balancing_journal.log"
)

ARTIFACT_STORE: oa.FileSystemArtifactStore = oa.FileSystemArtifactStore(store_dir)
JOURNAL = oj.JournalStorage(oj.JournalFileBackend(str(journal_path)))

REFS, FEATURES = ref_feature_lists_internal()

UNDERSAMPLING = ("RandomUnderSampler", "EditedNearestNeighbours", "NearMiss")
OVERSAMPLING = ("SMOTEENN", "SMOTETomek", "SMOTE")

# OVERSAMPLING = ("SMOTE", "SVMSMOTE", "BorderLineSMOTE", "RandomOverSampler")
# Finished above on 2025-3-24


def objective(
    trial: optuna.Trial, label_class: str = "tumor_type", test: bool = False
) -> float | None:
    if test:
        adata = training_data_internal_test(label=label_class)
    else:
        adata = training_data_internal(label=label_class)
    is_duplicated, val = ignore_duplicated(trial)
    if is_duplicated:
        return val
    filter = Filter(
        feature_col="GENEID", features=FEATURES["edgeR_median_lfc_feature_list_1000"]
    )
    type = trial.suggest_categorical("sampling_type", ("oversample", "undersample"))
    if type == "oversample":
        method_name = trial.suggest_categorical("oversample_name", OVERSAMPLING)
        strategy = trial.suggest_categorical(
            "oversampling_strategy", ("minority", "not majority", "targeted")
        )
    else:
        method_name = trial.suggest_categorical("undersample_name", UNDERSAMPLING)
        strategy = trial.suggest_categorical(
            "undersampling_strategy", ("not minority", "targeted")
        )
    hist_spec = None
    if strategy == "targeted" and type == "oversample":
        hist_spec = {
            "PAAD": 3,
            "LIHC": 3,
            "CHOL": 2.1,  # Aggressive because so few
        }  # Oversample routinely misclassified classes
    elif strategy == "targeted" and method_name in {"EditedNearestNeighbours"}:
        hist_spec = {
            "COAD-READ": 1
        }  # Undersample the classes that are commonly mistaken
    if strategy == "targeted":
        strategy = lambda y: spaced_resample(
            y, targets=hist_spec, undersample=type == "undersample", n_bins=40
        )

    bkwargs = {"method": method_name, "sampling_strategy": strategy}
    print(bkwargs)

    if method_name not in {"TomekLinks", "EditedNearestNeighbours", "NearMiss"}:
        bkwargs["random_state"] = RANDOM_STATE
    if method_name == "EditedNearestNeighbours":
        bkwargs["kind_sel"] = "mode"
    if method_name == "NearMiss":
        bkwargs["version"] = 3
    if method_name == "InstanceHardnessThreshold":
        bkwargs["estimator"] = RandomForestClassifier()  # Would use XGB, but want speed

    balancer = Balancer(**bkwargs)

    transformer = Transformer("clr", impute_fn=Imputer("plus_one"), inplace=False)
    adata = filter.fit_transform(adata)
    adata = transformer.fit_transform(adata)

    classifier_name = trial.suggest_categorical("classifier", ["XGB"])
    if classifier_name == "XGB":
        model = PredBase(model=XGBEstimator())
    else:
        model = PredBase(model=BalancedRandomForestClassifier())

    results = holdout(model, adata, SPLITS, label_col=label_class, balancer=balancer)
    # write_pickle(balancer, "balancer.pkl")
    # artifact_id = oa.upload_artifact(
    #     artifact_store=ARTIFACT_STORE, file_path="balancer.pkl", study_or_trial=trial
    # )
    # trial.set_user_attr("artifact_id", artifact_id)
    miss_counts = results["misses"].loc[:, label_class].value_counts().to_dict()
    trial.set_user_attr("misses", miss_counts)
    value = results["misc"]["balanced_acc"][0]  # [2025-03-26 Wed] Changed to b acc
    return value


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-c", "--cores", default=8)
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-s", "--use_saved", default=False, action="store_true")
    parser.add_argument("-d", "--dask", default=False, action="store_true")
    parser.add_argument("-l", "--label_class", default="tumor_type")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dask:
        cluster = SLURMCluster(cores=int(args.cores), memory=f"{args.memory} GB")
        client = Client(cluster)
    backend = "dask" if args.dask else "loky"
    par_args = {"wait_for_workers_timeout": 0} if args.dask else {"n_jobs": args.cores}

    with joblib.parallel_backend(backend):
        sampler = TPESampler(seed=RANDOM_STATE, multivariate=True, constant_liar=True)
        obj = partial(objective, test=args.test, label_class=args.label_class)

        if not args.use_saved:
            try:
                study = optuna.load_study(
                    storage=JOURNAL, study_name="try_balancing", sampler=sampler
                )
            except (KeyError, FileNotFoundError, ValueError):
                study = optuna.create_study(
                    storage=JOURNAL,
                    study_name="try_balancing",
                    direction="maximize",
                    sampler=sampler,
                )
                study.optimize(obj, catch=(RuntimeError,))
                print("Study complete")
                print(f"Best value: {study.best_value}")
                print(f"Best params: {study.best_params}")
                joblib.dump(study, here(store_dir, "study.pkl"))
        else:
            study = optuna.load_study(
                storage=JOURNAL, study_name="try_balancing", sampler=sampler
            )
            print(f"Best value: {study.best_value}")
            print(f"Best params: {study.best_params}")
            df = summarize_studies(study, "kappa")
            df.to_csv(here("data", "output", "balancing_results.csv"), index=False)
