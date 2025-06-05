#!/usr/bin/env ipython

import pickle
from functools import partial

import optuna
import optuna.artifacts as oa
import optuna.storages.journal as oj
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from too_predict.model import AlrEstimator, PredBase
from too_predict.optimization import Setup, Optimizer
from too_predict.utils import (
    RANDOM_STATE,
    get_data,
    load_pickle,
    training_data_internal_test,
)

# #  --- CODE BLOCK ---
#
adata = training_data_internal_test()[:100]
jlog = get_data("tests/optuna_journaldir", must_exist=False)
adir = get_data("tests/optuna_artifacts")
cv_results = get_data("tests/cv_results", must_exist=False)


def test_objective():
    study: optuna.Study = optuna.create_study(direction="maximize")
    O = Optimizer()
    obj = partial(
        O._objective,
        adata=adata,
        cv_splits=2,
        artifact_store=oa.FileSystemArtifactStore(adir),
    )
    study.optimize(obj, n_trials=3)
    print(study.best_trial)
    print("Study completed")
    return study


sampler = TPESampler(seed=RANDOM_STATE)


def test_pickle():
    study = test_objective()
    oa.download_artifact(
        artifact_store=oa.FileSystemArtifactStore(adir),
        artifact_id=study.best_trial.user_attrs["artifact_id"],
        file_path="best_model.pickle",
    )
    pipeline: Pipeline = load_pickle("best_model.pickle")
    pipeline.fit(adata, y="tumor_type")
    ppred = pipeline.predict(adata)
    oa.download_artifact(
        artifact_store=oa.FileSystemArtifactStore(adir),
        artifact_id=study.best_trial.user_attrs["cv_id"],
        file_path=cv_results,
    )
    res = load_pickle(cv_results)

    study.best_trial.user_attrs
    cons = Setup(trial=None, trial_params=study.best_params)
    transform, model, _ = cons()

    cpred = model.fit(transform(adata), y="tumor_type")
    assert (ppred == cpred).all()


def test_nested():
    search = Optimizer(
        save_model=True, save_cv=True, journal_dir=jlog, artifact_dir=adir
    )
    best = search.nested(adata, n_outer=3, n_inner=2)
    print(best)


test_nested()
