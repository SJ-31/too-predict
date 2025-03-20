#!/usr/bin/env ipython

from functools import partial

import joblib
import optuna
import optuna.artifacts as oa
import optuna.storages.journal as oj
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from imblearn.ensemble import BalancedRandomForestClassifier
from optuna.samplers import TPESampler
from pyhere import here
from too_predict.evaluation import holdout
from too_predict.filter import Filter
from too_predict.imbalance import Balancer
from too_predict.imputer import Imputer
from too_predict.model import PredBase, XGBEstimator
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

store_dir = here("data", ".optuna_artifactstore", "balancing")
store_dir.mkdir(exist_ok=True)
journal_path = here("data", ".optuna_journals", "balancing_journal.log")

ARTIFACT_STORE: oa.FileSystemArtifactStore = oa.FileSystemArtifactStore(store_dir)
JOURNAL = oj.JournalStorage(oj.JournalFileBackend(str(journal_path)))

REFS, FEATURES = ref_feature_lists_internal()


def objective(
    trial: optuna.Trial, label_class: str = "tumor_type", test: bool = False
) -> float:
    if test:
        adata = training_data_internal_test(label=label_class)
    else:
        adata = training_data_internal(label=label_class)
    filter = Filter(
        feature_col="GENEID", features=FEATURES["edgeR_median_lfc_feature_list_1000"]
    )
    type = trial.suggest_categorical("sampling_type", ("oversample", "undersample"))
    if type == "oversample":
        method_name = trial.suggest_categorical(
            "oversample_name",
            (
                "KMeansSMOTE",
                "SVMSMOTE",
                "ADASYN",
                "BorderLineSMOTE",
                "RandomOverSampler",
            ),
        )
        strategy = trial.suggest_categorical(
            "oversampling_strategy", ("minority", "not majority")
        )
    else:
        method_name = trial.suggest_categorical(
            "undersample_name", ("TomekLinks", "RandomUnderSampler")
        )
        strategy = trial.suggest_categorical(
            "undersampling_strategy", ("majority", "not minority")
        )
    bkwargs = {"method": method_name, "sampling_strategy": strategy}
    if method_name != "TomekLinks":
        bkwargs["random_state"] = RANDOM_STATE

    balancer = Balancer(**bkwargs)

    transformer = Transformer("clr", impute_fn=Imputer("plus_one"), inplace=False)
    adata = filter.fit_transform(adata)
    adata = transformer.fit_transform(adata)

    classifier_name = trial.suggest_categorical("classifier", ["XGB", "BalancedRF"])
    if classifier_name == "XGB":
        model = PredBase(model=XGBEstimator())
    else:
        model = PredBase(model=BalancedRandomForestClassifier())

    results = holdout(model, adata, SPLITS, label_col=label_class, balancer=balancer)
    write_pickle(balancer, "balancer.pkl")
    artifact_id = oa.upload_artifact(
        artifact_store=ARTIFACT_STORE, file_path="balancer.pkl", study_or_trial=trial
    )
    trial.set_user_attr("artifact_id", artifact_id)
    value = results["misc"]["kappa"][0]
    print(results["misc"])
    print(value)
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
            except (KeyError, FileNotFoundError):
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
            study = joblib.load(here(store_dir, "study.pkl"))
