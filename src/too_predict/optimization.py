#!/usr/bin/env ipython

# Study: optimization based on an objective function
# Trial: a single execution of the objective function

from functools import partial
from pathlib import Path

import optuna
import sklearn.linear_model as sl
import yaml
from sklearn.ensemble import RandomForestClassifier

from too_predict.evaluation import cross_validate
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.model import AlrBase, PredBase, SimPred, XGBEstimator
from too_predict.simulation import IMPLEMENTED_SIMULATION
from too_predict.transformer import Transformer
from too_predict.utils import (
    get_data,
    ref_feature_lists_internal,
    training_data_internal,
    training_data_internal_test,
)

REFS, FEATURES = ref_feature_lists_internal(add_all=False)


def get_options(file: str | Path | None = None) -> dict:
    if file is None:
        file = get_data("optimization_options.yaml")
    with open(file, "r") as f:
        loaded = yaml.safe_load(f)
    return loaded


def get_classifier(classifier_name, trial: optuna.Trial):
    # TODO: need to fill out these options
    match classifier_name:
        case "XGBEstimator":
            # trial.suggest_categorical()
            classifier = XGBEstimator()
        case "SGD":
            classifier = sl.SGDClassifier()
        case _:
            raise ValueError(f"Classifier {classifier_name} is not implemented!")
    return classifier


def get_transformation(transform_name, trial, opts: dict | None = None) -> dict:
    transform_kwargs = {}
    opts = {} if opts is None else opts
    match transform_name:
        case "clr":
            transform_kwargs["feature_col"] = "GENEID"
            clr_subset = opts.get("clr_subset", list(REFS.keys()) + [None])
            ref_set = trial.suggest_categorical("clr_subset", clr_subset)
            if ref_set is not None:
                transform_kwargs["features"] = REFS[ref_set]
        case "alr":
            alr_ref = opts.get("alr_references", REFS.keys())
            ref_set = trial.suggest_categorical("alr_references", alr_ref)
            transform_kwargs["references"] = REFS[ref_set]
        case "dirichlet":
            transform_kwargs["n_instances"] = trial.suggest_int(
                "n_dirichlet_instances", low=5, high=15, step=5
            )
    return transform_kwargs


def objective(
    trial: optuna.Trial,
    opts: dict | None = None,
    label_col: str = "tumor_type",
    test=False,
):
    transform: bool = True
    if opts is None:
        opts = get_options(None)

    imputation = trial.suggest_categorical("imputation", opts["imputation"])
    transform_name = trial.suggest_categorical("transformation", opts["transformation"])
    features = FEATURES[trial.suggest_categorical("feature_set", opts["feature_set"])]
    classifier_name = trial.suggest_categorical("classifier", opts["classifier"])

    transform_kwargs: dict = get_transformation(transform_name, trial)
    model = get_classifier(classifier_name, trial)
    if transform_name == "alr":
        classifier = AlrBase(
            model,
            references=transform_kwargs["references"],
            imputation=imputation,
        )
        del transform_kwargs["references"]
        transform = False
    elif transform_name in IMPLEMENTED_SIMULATION:
        classifier = SimPred(model=model, method=transform_name, **transform_kwargs)
    else:
        classifier = PredBase(model=model)

    if test:
        adata = training_data_internal_test()
        adata = adata[:50, :]
        cv_splits = 2
    else:
        adata = training_data_internal()
        cv_splits = 5

    adata = Filter(feature_col="GENEID", features=features).fit_transform(adata)
    if transform:
        adata = Transformer(
            transform_name,
            impute_fn=Imputer(imputation),
            inplace=False,
            **transform_kwargs,
        ).fit_transform(adata)
    cv_results: dict = cross_validate(
        classifier, adata, label_col=label_col, n_splits=cv_splits
    )
    kappa = cv_results["misc"]["kappa"]
    return kappa.mean()


def run(n_trials=100, **kwargs) -> optuna.Study:
    study: optuna.Study = optuna.create_study(direction="maximize")
    obj = partial(objective, **kwargs)
    study.optimize(obj, n_trials=n_trials)
    print(study.best_trial)
    print("Study completed")
    return study
