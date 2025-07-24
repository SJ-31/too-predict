#!/usr/bin/env ipython

# Study: optimization based on an objective function
# Trial: a single execution of the objective function
import pickle
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, override

import anndata as ad
import numpy as np
import optuna
import optuna.artifacts as oa
import optuna.storages.journal as oj
import sklearn.metrics as sm
import sklearn.model_selection as ms
import sklearn.svm as sv
import yaml
from optuna.pruners import BasePruner, HyperbandPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.trial import TrialState
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from too_predict.evaluation import cross_validate, holdout
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.model import AlrBase, PredBase, SimPred, XGBEstimator
from too_predict.simulation import IMPLEMENTED_SIMULATION
from too_predict.transformer import Transformer
from too_predict.utils import (
    RANDOM_STATE,
    get_data,
    ref_feature_lists_internal,
    write_pickle,
)

REFS, FEATURES = ref_feature_lists_internal(add_all=False)


def get_options(file: str | Path | None = None) -> dict:
    if file is None:
        file = get_data("optimization_options.yaml")
    with open(file, "r") as f:
        loaded = yaml.safe_load(f)
    return loaded


class TrialSetup:
    def __init__(
        self, trial: optuna.Trial | None = None, trial_params: dict | None = None
    ) -> None:
        self.user_opts: dict
        self.for_trial: bool = trial is not None
        self.trial: optuna.Trial | None = trial
        self.params: dict | None = trial_params
        if trial is None and trial_params is None:
            raise ValueError("One of trial or trial_params must be provided!")

    def _suggest_param_or_default(self, param_name: str) -> Any:
        """
            Read a default value of a hyperparameter from user options, or suggest a
            selection to the optuna trial
            If the value of `param_name` in user_opts is a tuple, the first and
        second values are interpreted as start, end of a range and the last
        as the step
            If a sequence and the first item is "literal" or "lit", return the second item
               regardless
            If a list, interpreted as categorical options
        """
        val = self.user_opts.get(param_name)
        if isinstance(val, Sequence) and len(val) > 0 and val[0] in {"literal", "lit"}:
            return val[1]
        if isinstance(val, str) and val.lower() == "none":
            return None
        if val is None:
            raise ValueError(f"Missing default value for {param_name}!")
        if isinstance(val, list):
            return self.trial.suggest_categorical(param_name, val)
        elif isinstance(val, tuple):
            if len(val) != 3:
                raise ValueError("Values for tuple params must be (start, stop, step)")
            start, stop, step = val
            range_kwargs = {}
            if str(step) == "True":
                range_kwargs["log"] = True
            else:
                range_kwargs["step"] = step
            if isinstance(start, float):
                return self.trial.suggest_float(param_name, start, stop, **range_kwargs)
            else:
                return self.trial.suggest_int(param_name, start, stop, **range_kwargs)
        return val

    def __call__(self, opts: dict | None = None) -> tuple:
        raise NotImplementedError()


def ignore_duplicated(
    trial: optuna.Trial, states=(TrialState.COMPLETE, TrialState.PRUNED)
) -> tuple[bool, float | None]:
    consider = trial.study.get_trials(deepcopy=False, states=states)
    for t in reversed(consider):
        if t.params == trial.params:
            return True, t.value
    return False, 0.0


# * Base optimizer


class BaseOptimizer:
    def __init__(
        self,
        score_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        label_col: str = "tumor_type",
        save_model: bool = True,
        save_cv: bool = True,
        ignore_duplicated: bool = True,
        storage_file: Path | str | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        self.label_col: str = label_col
        self.save_model: bool = save_model
        self.save_cv: bool = save_cv
        self.storage: str | None = (
            str(storage_file) if isinstance(storage_file, Path) else storage_file
        )
        self.artifact_dir: Path | None = artifact_dir
        self.ignore_duplicated: bool = ignore_duplicated
        self.score_fn: Callable = score_fn
        self.objective: Callable[[optuna.Trial], float]

    def make_objective(self, **kwargs):
        if "artifact_store" not in kwargs and self.artifact_dir is not None:
            kwargs["artifact_store"] = oa.FileSystemArtifactStore(self.artifact_dir)
        self.objective = partial(self._objective, **kwargs)

    def _objective(
        self,
        trial: optuna.Trial,
        adata: ad.AnnData,
        do_splits: bool = True,
        do_cv: bool = True,
        split_fns: dict | None = None,
        split_masks: dict | None = None,
        device: str = "cpu",
        cv_splits=5,
        opts: dict | None = None,
        artifact_store: oa.FileSystemArtifactStore | None = None,
        **kwargs,
    ):
        raise NotImplementedError()

    def run_study(self, **kwargs) -> optuna.Study:
        defaults = {
            "study_name": "optimize",
            "direction": "maximize",
            "load_if_exists": True,
        }
        if "directions" in kwargs:
            del defaults["direction"]
        if "storage" not in kwargs and self.storage is not None:
            kwargs["storage"] = oj.JournalStorage(oj.JournalFileBackend(self.storage))
        defaults.update(kwargs)
        try:
            study = optuna.create_study(**defaults)
            study.optimize(self.objective)
        except ValueError:
            defaults = {
                k: v
                for k, v in defaults.items()
                if k not in {"direction", "directions", "load_if_exists"}
            }
            study = optuna.load_study(**defaults)
        return study


#
# * For shallow models


class Optimizer(BaseOptimizer):
    def __init__(
        self,
        score_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        label_col: str = "tumor_type",
        save_model: bool = True,
        save_cv: bool = True,
        group_col: None | str = None,
        ignore_duplicated: bool = True,
        journal_file: Path | str | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        super().__init__(
            score_fn,
            label_col,
            save_model,
            save_cv,
            ignore_duplicated,
            journal_file,
            artifact_dir,
        )
        self.group_col: str = group_col

    @override
    def _objective(
        self,
        trial: optuna.Trial,
        adata: ad.AnnData,
        do_splits: bool = True,
        do_cv: bool = True,
        split_fns: dict | None = None,
        split_masks: dict | None = None,
        cv_splits=5,
        opts: dict | None = None,
        artifact_store: oa.FileSystemArtifactStore | None = None,
        **kwargs,
    ):
        if self.ignore_duplicated:
            is_duplicated, val = ignore_duplicated(trial)
            if is_duplicated:
                return val

        setup = ShallowSetup(trial=trial, **kwargs)
        # Suggest values to the trial object, it'll track which values have been
        # seen
        transform, classifier, pipeline = setup(
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

        adata = transform(adata.copy())
        all_accs = []
        if cv_splits > 1:
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
            all_accs.append(cv_results["misc"]["balanced_acc"].mean())
        if split_fns is None and split_masks is None and cv_splits <= 1:
            raise ValueError("Not training task given!")
        split_res: dict = holdout(
            classifier, adata, split_fns=split_fns, split_masks=split_masks
        )
        all_accs.append(split_res["misc"]["balanced_acc"].mean())
        return np.mean(all_accs)

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
            if self.storage is not None:  # This enables parallelization
                out = self.storage.joinpath(f"fold_{fold}.log")
                jfile = oj.JournalFileBackend(str(out))
                study_kwargs["storage"] = oj.JournalStorage(jfile)
            if self.artifact_dir is not None:
                a_store_dir = self.artifact_dir.joinpath(f"fold_{fold}")
                a_store_dir.mkdir(exist_ok=True, parents=True)
                a_store = oa.FileSystemArtifactStore(a_store_dir)
                obj_kwargs["artifact_store"] = a_store

            study = optuna.create_study(**study_kwargs)
            obj = partial(self._objective, **obj_kwargs)
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
                cons = self.setup_fn()
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


class ShallowSetup(TrialSetup):
    """Helper class to transform data and get predictor from params

    Returns
    -------
    tuple of [adata transformation function, unfitted model with params]


    Notes
    -----
    Used either to build a model for the objective function,
        or to get a model from optuna study params (self.params)
        (ideally use optuna artifactstore, but use this as backup)

    The dictionary passed to this object in __call__ MUST have the following keys:
        imputation
        transformation
        feature_sets
        classifier
    Include key `<object>_default` if you don't want to modify any parameters
        e.g. classifier_default = True to use the defaults for that classifier

    Their values must either be
        a list of strings, which will then be passed to the optuna trial as categorial
           options
        or a tuple of [object, True] to indicate
            that the object be used as is. e.g. [PredBase(model), True]
    """

    def _suggest_gradient_booster(self, name: str):
        if self.for_trial and not self.user_opts.get("classifier_default"):
            learning_rate = self._suggest_param_or_default("learning_rate")
            # Synonymous with shrinkage rate, eta
            l2_reg = self._suggest_param_or_default("l2_regularization")  # lambda
            max_depth = self._suggest_param_or_default("max_depth")
            max_bin = self._suggest_param_or_default("max_bin")
            if name == "XGBEstimator":
                l1_reg = self._suggest_param_or_default("l1_regularization")  # alpha
                minimum_loss = self._suggest_param_or_default("minimum_loss")  # gamma
        else:
            minimum_loss = self.params.get("minimum_loss", 0)
            max_depth = self.params.get("max_depth", 3)
            l1_reg = self.params.get("l1_regularization", 1)
            l2_reg = self.params.get("l2_regularization", 0)
        learning_rate = self.params.get("learning_rate", 0.5)
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

    def _suggest_svm(self, use_default: bool):
        if not use_default:
            if self.for_trial:
                c = self._suggest_param_or_default("C")
                loss_fn = self._suggest_param_or_default("loss")
            else:
                c = self.params.get("C")
                loss_fn = self.params.get("loss")
            return sv.LinearSVC(C=c, loss=loss_fn, random_state=RANDOM_STATE)
        return sv.LinearSVC(random_state=RANDOM_STATE)

    def _suggest_classifier(self, name, features: list):
        if name in {"XGBEstimator", "HistGradientBoostingClassifier"}:
            model = self._suggest_gradient_booster(name)
        elif name == "SVM":
            model = self._suggest_svm()
        else:
            raise ValueError(f"Classifier {name} is not implemented yet!")

        tname = self.user_opts["transformation"]
        classifier: PredBase
        if tname == "alr":
            transform_kwargs: dict = self._suggest_transformation_kwargs(tname)
            classifier = AlrBase(
                model,
                references=transform_kwargs["references"],
                imputation=self._get("imputation", features),
                n_refs=transform_kwargs["n_refs"],
            )
            features.extend(transform_kwargs["references"])
        elif tname in IMPLEMENTED_SIMULATION:
            transform_kwargs = self._suggest_transformation_kwargs(tname)
            classifier = SimPred(model=model, method=tname, **transform_kwargs)
        else:
            classifier = PredBase(model=model)
        return classifier

    def _suggest_transformation(self, tname) -> Transformer | None:
        if tname == "alr":
            return None
        imputation = self._get("imputation", None)
        kwargs = self._suggest_transformation_kwargs(tname)
        return Transformer(
            tname,
            impute_fn=Imputer(imputation),
            inplace=False,
            **kwargs,
        )

    def _suggest_transformation_kwargs(self, tname) -> dict:
        transform_kwargs = {}
        if tname == "clr":
            transform_kwargs["feature_col"] = "GENEID"
            ref_set = (
                self._suggest_param_or_default("clr_subset")
                if self.for_trial
                else self.params.get("clr_subset")
            )
            if ref_set is not None and ref_set in REFS:
                transform_kwargs["features"] = REFS[ref_set]
        elif tname == "alr":
            ref_set = (
                self._suggest_param_or_default("alr_references")
                if self.for_trial
                else self.params.get("alr_references")
            )
            alr_n_refs = (
                self._suggest_param_or_default("alr_n_references")
                if self.for_trial
                else self.params.get("alr_n_references")
            )
            transform_kwargs["n_refs"] = alr_n_refs
            transform_kwargs["references"] = REFS[ref_set]
        elif tname == "dirichlet":
            transform_kwargs["n_instances"] = (
                self._suggest_param_or_default("n_dirichlet_instances")
                if self.for_trial
                else self.params.get("n_dirichlet_instances")
            )
        else:
            raise ValueError("Not recognized!")
        return transform_kwargs

    def _get(
        self,
        value: Literal["transformation", "classifier", "imputation"],
        features: list | None,
    ) -> Filter | Transformer | PredBase | list | None | str:
        vals = self.user_opts.get(value)
        if vals is None:
            raise ValueError(f"Key {value} not provided!")

        if value == "imputation":
            return self._suggest_param_or_default(value)
        read = self._suggest_param_or_default(value)
        if isinstance(read, str) and value == "transformation":
            return self._suggest_transformation(read)
        elif isinstance(read, str) and value == "classifier":
            return self._suggest_classifier(read, features=features)
        return read

    @override
    def __call__(self, opts: dict | None = None) -> tuple:
        transform: bool = True
        # Set up pipeline parameters
        if opts is None:
            raise ValueError("options dictionary must be provided if for trial!")
        self.user_opts = opts
        if self.for_trial:
            # Suggest for trial
            features = FEATURES[self._suggest_param_or_default("feature_set")]
            transformer = self._get("transformation", features)
            classifier = self._get("classifier", features)
        else:
            # Read parameters from dictionary
            features = FEATURES[self.params.get("feature_set")]
            transformer = self.params.get("transformation")
            classifier = self.params.get("classifier")

        filter = Filter(feature_col="GENEID", features=features)
        pipeline_lst = [("filter", filter)]
        if transformer is not None:
            pipeline_lst.append(("transformer", transformer))
        pipeline_lst.append(("classifier", classifier))

        def transform_fn(X: ad.AnnData) -> ad.AnnData:
            x: ad.AnnData = filter.fit_transform(X)
            if transform:
                x = transformer.fit_transform(x)
            return x

        return transform_fn, classifier, Pipeline(pipeline_lst)
