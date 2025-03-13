#!/usr/bin/env ipython

# Study: optimization based on an objective function
# Trial: a single execution of the objective function

import optuna
import sklearn.linear_model as sl
from sklearn.ensemble import RandomForestClassifier

from too_predict.evaluation import cross_validate
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.model import AlrBase, PredBase, SimPred, XGBEstimator
from too_predict.simulation import IMPLEMENTED_SIMULATION
from too_predict.transformer import Transformer
from too_predict.utils import (
    ref_feature_lists_internal,
    training_data_internal,
    training_data_internal_test,
)

REFS, FEATURES = ref_feature_lists_internal(add_all=False)


def get_classifier(classifier_name, trial: optuna.Trial):
    match classifier_name:
        case "XGBEstimator":
            # trial.suggest_categorical()
            classifier = XGBEstimator()
        case "SGD":
            classifier = sl.SGDClassifier()
        case _:
            raise ValueError(f"Classifier {classifier_name} is not implemented!")
    return classifier


def get_transformation(transform_name, trial) -> dict:
    transform_kwargs = {}
    match transform_name:
        case "clr":
            transform_kwargs["feature_col"] = "GENEID"
            ref_set = trial.suggest_categorical(
                "clr_subset", list(REFS.keys()) + [None]
            )
            if ref_set is not None:
                transform_kwargs["features"] = REFS[ref_set]
        case "alr":
            ref_set = trial.suggest_categorical("alr_references", list(REFS.keys()))
            transform_kwargs["references"] = REFS[ref_set]
        case "dirichlet":
            transform_kwargs["n_instances"] = trial.suggest_int(
                "n_dirichlet_instances", low=6, high=16, step=2
            )
    return transform_kwargs


def objective(trial: optuna.Trial, label_col: str = "tumor_type", test=False):
    transform: bool = True
    imputation = trial.suggest_categorical(
        "imputation", ["plus_one", "replace_one", "none"]
    )
    transform_name = trial.suggest_categorical(
        "transformation", ["clr", "tmm", "tpm", "dirichlet", "alr"]
    )
    features = FEATURES[trial.suggest_categorical("feature_set", FEATURES.keys())]
    classifier_name = trial.suggest_categorical("classifier", ["XGBEstimator", "SGD"])

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
    study.optimize(lambda trial: objective(trial, **kwargs), n_trials=n_trials)
    print(study.best_trial)
    return study
