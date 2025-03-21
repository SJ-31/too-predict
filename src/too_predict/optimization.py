#!/usr/bin/env ipython

# Study: optimization based on an objective function
# Trial: a single execution of the objective function

import pickle
from functools import partial
from pathlib import Path
from typing import Callable

import anndata as ad
import numpy as np
import optuna
import optuna.artifacts as oa
import optuna.storages.journal as oj
import sklearn.linear_model as sl
import sklearn.metrics as sm
import sklearn.model_selection as ms
import sklearn.svm as sv
import yaml
from optuna.pruners import BasePruner, HyperbandPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.trial import TrialState
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from too_predict import transformer
from too_predict.evaluation import cross_validate, prc_auc_score, write_cross_val
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.model import AlrBase, PredBase, SimPred, XGBEstimator
from too_predict.simulation import IMPLEMENTED_SIMULATION
from too_predict.transformer import Transformer
from too_predict.utils import (
    RANDOM_STATE,
    get_data,
    ref_feature_lists_internal,
    training_data_internal,
    training_data_internal_test,
    write_pickle,
)

REFS, FEATURES = ref_feature_lists_internal(add_all=False)


def get_options(file: str | Path | None = None) -> dict:
    if file is None:
        file = get_data("optimization_options.yaml")
    with open(file, "r") as f:
        loaded = yaml.safe_load(f)
    return loaded


class Constructor:
    """Helper class to transform data and get predictor from params

    Returns
    -------
    tuple of [adata transformation function, unfitted model with params]


    Notes
    -----
    Used either to build a model for the objective function,
        or to get a model from optuna study params
        (ideally use optuna artifactstore, but use this as backup)
    """

    def __init__(
        self, trial: optuna.Trial | None = None, trial_params: dict | None = None
    ) -> None:
        self.for_trial = trial is not None
        self.trial: optuna.Trial | None = trial
        self.params: dict | None = trial_params

    def _get_gradient_booster(self, name: str):
        if self.for_trial:
            learning_rate = self.trial.suggest_categorical(
                "learning_rate", [0.01, 0.1, 0.2, 0.3]
            )  # Synonymous with shrinkage rate, eta
            l2_reg = self.trial.suggest_float(
                "l2_regularization", 0, 5, step=1
            )  # lambda
            max_depth = self.trial.suggest_int("max_depth", 3, 15, step=5)
            if name == "XGBEstimator":
                l1_reg = self.trial.suggest_float(
                    "l1_regularization", 0, 5, step=1
                )  # alpha
                minimum_loss = self.trial.suggest_int(
                    "minimum_loss", 0, 5, step=1
                )  # gamma
        else:
            minimum_loss = self.params.get("minimum_loss")
            learning_rate = self.params.get("learning_rate")
            max_depth = self.params.get("max_depth")
            l1_reg = self.params.get("l1_regularization")
            l2_reg = self.params.get("l2_regularization")
        if name == "XGBEstimator":
            return XGBEstimator(
                gamma=minimum_loss,
                learning_rate=learning_rate,
                max_depth=max_depth,
                reg_lambda=l2_reg,
                reg_alpha=l1_reg,
                random_state=RANDOM_STATE,
            )
        else:
            return HistGradientBoostingClassifier(
                random_state=RANDOM_STATE,
                scoring="loss",
                early_stopping=True,
                learning_rate=learning_rate,
                l2_regularization=l2_reg,
                max_depth=max_depth,
            )

    def _get_svm(self):
        if self.for_trial:
            c = self.trial.suggest_float("C", low=0.2, high=1, step=0.2)
            loss_fn = self.trial.suggest_categorical("loss", ["hinge", "squared_hinge"])
        else:
            c = self.params.get("C")
            loss_fn = self.params.get("loss")
        return sv.LinearSVC(C=c, loss=loss_fn, random_state=RANDOM_STATE)

    def _get_classifier(self, name):
        match name:
            case "XGBEstimator" | "HistGradientBoostingClassifier":
                return self._get_gradient_booster(name)
            case "SVM":
                return self._get_svm()
            case _:
                raise ValueError(f"Classifier {name} is not implemented!")

    def _get_transformation(self, transform_name, opts: dict | None = None) -> dict:
        transform_kwargs = {}
        opts = {} if opts is None else opts
        match transform_name:
            case "clr":
                transform_kwargs["feature_col"] = "GENEID"
                clr_subset = opts.get("clr_subset", list(REFS.keys()) + [None])
                ref_set = (
                    self.trial.suggest_categorical("clr_subset", clr_subset)
                    if self.for_trial
                    else self.params.get("clr_subset")
                )
                if ref_set is not None:
                    transform_kwargs["features"] = REFS[ref_set]
            case "alr":
                alr_ref = opts.get("alr_references", REFS.keys())
                ref_set = (
                    self.trial.suggest_categorical("alr_references", alr_ref)
                    if self.for_trial
                    else self.params.get("alr_references")
                )
                alr_n_refs = (
                    self.trial.suggest_int("alr_n_references", low=5, high=20, step=5)
                    if self.for_trial
                    else self.params.get("alr_n_references")
                )
                transform_kwargs["n_refs"] = alr_n_refs
                transform_kwargs["references"] = REFS[ref_set]
            case "dirichlet":
                transform_kwargs["n_instances"] = (
                    self.trial.suggest_int(
                        "n_dirichlet_instances", low=5, high=15, step=5
                    )
                    if self.for_trial
                    else self.params.get("n_dirichlet_instances")
                )
        return transform_kwargs

    def __call__(
        self, opts: dict | None = None
    ) -> tuple[Callable[[ad.AnnData], ad.AnnData], PredBase, Pipeline]:
        transform: bool = True
        # Get parameters
        if self.for_trial:
            imputation = self.trial.suggest_categorical(
                "imputation", opts["imputation"]
            )
            transform_name = self.trial.suggest_categorical(
                "transformation", opts["transformation"]
            )
            features = FEATURES[
                self.trial.suggest_categorical("feature_set", opts["feature_set"])
            ]
            classifier_name = self.trial.suggest_categorical(
                "classifier", opts["classifier"]
            )
        else:
            imputation = self.params.get("imputation")
            transform_name = self.params.get("transformation")
            features = FEATURES[self.params.get("feature_set")]
            classifier_name = self.params.get("classifier")

        # Make classifier from params
        model = self._get_classifier(classifier_name)
        transform_kwargs: dict = self._get_transformation(transform_name)
        if transform_name == "alr":
            classifier = AlrBase(
                model,
                references=transform_kwargs["references"],
                imputation=imputation,
                n_refs=transform_kwargs["n_refs"],
            )
            features += transform_kwargs["references"]
            del transform_kwargs["references"]
            transform = False
        elif transform_name in IMPLEMENTED_SIMULATION:
            classifier = SimPred(model=model, method=transform_name, **transform_kwargs)
        else:
            classifier = PredBase(model=model)

        # Return transformation function and classifier
        filter = Filter(feature_col="GENEID", features=features)
        pipeline_lst = [("filter", filter)]
        if transform:
            transformer = Transformer(
                transform_name,
                impute_fn=Imputer(imputation),
                inplace=False,
                **transform_kwargs,
            )
            pipeline_lst.append(("transformer", transformer))
        else:
            transformer = None
        pipeline_lst.append(("classifier", classifier))

        def transform_fn(X: ad.AnnData) -> ad.AnnData:
            X = filter.fit_transform(X)
            if transform:
                X = transformer.fit_transform(X)
            return X

        return transform_fn, classifier, Pipeline(pipeline_lst)


def ignore_duplicated(
    trial: optuna.Trial, states=(TrialState.COMPLETE, TrialState.PRUNED)
) -> tuple[bool, float | None]:
    consider = trial.study.get_trials(deepcopy=False, states=states)
    for t in reversed(consider):
        if t.params == trial.params:
            return True, t.value
    return False, 0.0


class Optimizer:
    def __init__(
        self,
        score_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        label_col: str = "tumor_type",
        save_model: bool = True,
        save_cv: bool = True,
        ignore_duplicated: bool = True,
        group_col: None | str = None,
        journal_dir: Path | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        self.label_col = label_col
        self.save_model = save_model
        self.group_col: None | str = group_col
        self.save_cv = save_cv
        self.journal_dir = journal_dir
        self.artifact_dir = artifact_dir
        self.ignore_duplicated = ignore_duplicated
        self.score_fn = score_fn

    def objective(
        self,
        trial: optuna.Trial,
        adata: ad.AnnData,
        cv_splits=5,
        opts: dict | None = None,
        artifact_store: oa.FileSystemArtifactStore | None = None,
    ):
        if ignore_duplicated:
            is_duplicated, val = ignore_duplicated(trial)
            if is_duplicated:
                return val
        cons = Constructor(trial=trial, trial_params=None)
        transform, classifier, pipeline = cons(
            opts=opts if opts is not None else get_options()
        )
        if self.save_model:
            write_pickle(pipeline, "save.pickle")
            artifact_id = oa.upload_artifact(
                artifact_store=artifact_store,
                file_path="save.pickle",
                study_or_trial=trial,
            )
            trial.set_user_attr("artifact_id", artifact_id)

        adata = transform(adata)
        cv_results: dict = cross_validate(
            classifier,
            adata,
            label_col=self.label_col,
            n_splits=cv_splits,
            trial=trial,
            get_report_val=lambda x: x["kappa"],
        )
        if self.save_cv:
            write_pickle(cv_results, "cv_results.pickle")
            cv_id = oa.upload_artifact(
                artifact_store=artifact_store,
                file_path="cv_results.pickle",
                study_or_trial=trial,
            )
            trial.set_user_attr("cv_id", cv_id)
        kappa = cv_results["misc"]["kappa"]
        return kappa.mean()

    def nested(
        self,
        adata: ad.AnnData,
        n_outer: int,
        n_inner: int,
        pruner: BasePruner | None = None,
        sampler_fn: Callable[[int], BaseSampler] | None = None,
    ):
        outer_results = []
        if not self.group_col:
            cv = ms.StratifiedKFold(
                n_splits=n_outer, shuffle=True, random_state=RANDOM_STATE
            )
            outer_splits = cv.split(adata, adata.obs[self.label_col])
        else:
            cv = ms.StratifiedGroupKFold(
                n_splits=n_outer, random_state=RANDOM_STATE, shuffle=True
            )
            outer_splits = cv.split(
                adata, adata.obs[self.label_col], groups=adata.obs[self.group_col]
            )
        a_store = None
        if self.artifact_dir is None and self.save_model:
            raise ValueError("Must supply artifact store if `save_model` is True!")
        for fold, (train_i, test_i) in enumerate(outer_splits):
            # Search hyperparameter space in inner loop
            x_train = adata[train_i]
            x_test, y_test = adata[test_i], adata.obs[self.label_col].iloc[test_i]
            sampler: BaseSampler = (
                TPESampler(seed=fold) if sampler_fn is None else sampler_fn(fold)
            )  # Sampler function takes seed as param
            study_kwargs = {
                "study_name": "optimize_predictions",
                "direction": "maximize",
                "sampler": sampler,
            }
            if pruner is not None:
                study_kwargs["pruner"] = pruner
            obj_kwargs = {"adata": x_train, "cv_splits": n_inner}
            if self.journal_dir is not None:  # This enables parallelization
                out = self.journal_dir.joinpath(f"fold_{fold}.log")
                jfile = oj.JournalFileBackend(str(out))
                study_kwargs["storage"] = oj.JournalStorage(jfile)
            if self.artifact_dir is not None:
                a_store_dir = self.artifact_dir.joinpath(f"fold_{fold}")
                a_store_dir.mkdir(exist_ok=True, parents=True)
                a_store = oa.FileSystemArtifactStore(a_store_dir)
                obj_kwargs["artifact_store"] = a_store

            study = optuna.create_study(**study_kwargs)
            obj = partial(self.objective, **obj_kwargs)
            study.optimize(obj)

            # Test optimal hyperparameters with inner test set
            best_params = study.best_params
            if self.save_model:
                oa.download_artifact(
                    artifact_store=a_store,
                    artifact_id=study.best_trial.user_attrs["artifact_id"],
                    file_path="best_model.pickle",
                )
                pipeline: Pipeline = pickle.load("best_model.pickle")
                pipeline.fit(x_train, y=x_train.obs[self.label_col])
                y_hat = pipeline.predict(x_test)
            else:
                cons = Constructor(trial=None, trial_params=best_params)
                transform, model, _ = cons()
                model.fit(transform(x_train))
                y_hat = model.predict(transform(x_test))
            score = (
                self.score_fn(y_test, y_hat)
                if self.score_fn is not None
                else sm.accuracy_score(y_test, y_hat)
            )
            outer_results.append((fold, best_params, score))

        return outer_results
