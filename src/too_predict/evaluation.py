#!/usr/bin/env ipython
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Literal

import anndata as ad
import numpy as np
import optuna
import pandas as pd
import sklearn.metrics as me
import sklearn.metrics as met
import sklearn.model_selection as ms
from rpy2.rinterface_lib.embedded import RRuntimeError
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset
from torchmetrics.functional.classification import accuracy

from too_predict.corrector import Corrector
from too_predict.imbalance import Balancer
from too_predict.transformer import Transformer
from too_predict.utils import RANDOM_STATE


def curve_multiclass(
    curve_fn: Callable,
    first: str,
    second: str,
    true_labels,
    pred_proba: pd.DataFrame | np.ndarray,
    s_fns: dict[str, Callable] | None = None,
    s_fn_all_classes: Callable | None = None,
    s_fn_all_classes_name: str = "",
    first_x: bool = True,
    classes=None,
):
    """
    Higher-order function for computing binary curve metrics e.g. ROC, precision-recall
    for a multi-class classification task
    using the One-vs-Rest (OvR) approach and return them in a pandas DataFrame.

    Parameters:
    -----------
    true_labels : np.ndarray | pd.Series | list
        The true class labels for each sample.

    pred_proba : np.ndarray | pd.DataFrame
        The predicted probabilities for each class, where each column corresponds to a class.

    curve_fn : callable function with signature Callable[true, score, pos_label] -> first, second, thresholds
        which are ndarrays

    classes : If pred_proba are the array the predicted probabilities,
        the names of the classes in their order of appearance

    s_fn : dict[str, Callable[true, score, labels]] a dict of functions
        supporting multi-class classification that returns an iterable.

    s_fn_all_classes : a function supporting multi-class classification returning
        a single value representing performance for all classes

    first_x : true if the array `first` from curve_fn is the x axis

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing:
        - x_axis (float)
        - y_axis (float)
        - 'thresholds' (float): The decision thresholds.
        - 'class' (str): The class for which the metrics metrics are computed.
        - summary_fn_name: Summary metric for the current class, if provided
    """
    if not isinstance(pred_proba, pd.DataFrame) and classes is not None:
        prob_df = pd.DataFrame(pred_proba, columns=classes)
    elif not isinstance(pred_proba, pd.DataFrame) and classes is None:
        raise ValueError("If `pred_proba` isn't a df, the classes must be provided!")
    else:
        prob_df = pred_proba
    summary_labels = {}
    if s_fns is not None:
        for name, fn in s_fns.items():
            summary = fn(true_labels, prob_df, prob_df.columns)
            summary_labels[name] = {
                c: v.round(3) for c, v in zip(prob_df.columns, summary)
            }
    if s_fn_all_classes is not None:
        summary_all = s_fn_all_classes(true_labels, prob_df, prob_df.columns)
    dfs = []
    for c in prob_df.columns:
        f, s, thresholds = curve_fn(true_labels, prob_df[c], c)
        if len(thresholds) < len(f):
            thresholds = np.concatenate([[np.inf], thresholds])
        tmp = pd.DataFrame({first: f, second: s, "thresholds": thresholds})
        tmp["auc"] = me.auc(f, s) if first_x else me.auc(s, f)
        tmp["class"] = c
        for s_fn_name, val_dict in summary_labels.items():
            tmp[s_fn_name] = val_dict[c]
        dfs.append(tmp)
    roc_df = pd.concat(dfs)
    if s_fn_all_classes is not None and s_fn_all_classes_name:
        roc_df[s_fn_all_classes_name] = summary_all
    return roc_df


def roc_multiclass(
    true_labels,
    pred_proba: pd.DataFrame | np.ndarray,
    classes=None,
    average="macro",  # If you equally value minority classes, set this to "macro"
    multi_class="ovo",
):
    return curve_multiclass(
        lambda true, score, labels: me.roc_curve(true, score, pos_label=labels),
        "fpr",
        "tpr",
        true_labels,
        pred_proba,
        classes=classes,
        s_fn_all_classes=lambda true, score, labels: me.roc_auc_score(
            true, score, average=average, multi_class=multi_class, labels=labels
        ),
        s_fn_all_classes_name=f"{average}_{multi_class}_auc",
    )


def prc_auc(y_true, y_score, **kwargs):
    p, r, _ = me.precision_recall_curve(y_true, y_score)
    return me.auc(r, p)


prc_auc_score: Callable = me.make_scorer(
    prc_auc,
    response_method=["decision_function", "predict_proba"],
    greater_is_better=True,
)


def precision_recall_multiclass(
    true_labels, pred_proba: pd.DataFrame | np.ndarray, classes=None, average="macro"
):
    return curve_multiclass(
        lambda true, score, labels: me.precision_recall_curve(
            true, score, pos_label=labels
        ),
        "precision",
        "recall",
        true_labels,
        pred_proba,
        classes=classes,
        first_x=False,
        s_fns={
            "average_precision": lambda true, score, _: me.average_precision_score(
                true, score, average=None
            )
        },
        s_fn_all_classes=lambda true, score, _: me.average_precision_score(
            true, score, average=average
        ),
        s_fn_all_classes_name=f"{average}_average_precision",
    )


def classification_report2df(
    report: dict, fold: int | None = None
) -> tuple[pd.DataFrame, float]:
    """Convert scikit-learn classification report (dict format) into a long dataframe
    :return: tuple[datframe of the major metrics, accuracy]
    """
    dct = {"class": [], "precision": [], "recall": [], "f1-score": [], "support": []}
    metrics = set(dct.keys()) - {"class"}
    for c in report.keys():
        if c == "accuracy":
            continue
        dct["class"].append(c)
        for r in metrics:
            dct[r].append(report[c][r])
    df = pd.DataFrame(dct)
    if fold:
        df["fold"] = fold
    return df, report["accuracy"]


def get_all_metrics(true, score, classes, average: str = "macro") -> dict:
    predictions = pd.DataFrame(score, columns=classes)
    pred_vals = predictions.idxmax(1)
    rep = me.classification_report(true, pred_vals, output_dict=True)

    cm = confusion_matrix_df(true, pred_vals, labels=classes)
    rep_df, acc = classification_report2df(rep)
    try:
        roc_df = roc_multiclass(true, predictions, average=average)
    except (IndexError, ValueError) as e:
        # [2025-03-10 Mon] can happen when true values don't contain
        # all classes
        # Or when using a model with decision_function
        print("WARNING: failed to calculate ROC, ignoring...")
        print(f"Exception: {e}")
        roc_df = None
    try:
        prec_recall_df = precision_recall_multiclass(true, predictions, average=average)
    except (IndexError, ValueError) as e:
        print("WARNING: failed to calculate PRC, ignoring...")
        print(f"Exception: {e}")
        prec_recall_df = None
    return {
        "cm": cm,
        "pred": list(pred_vals),
        "balanced_acc": me.balanced_accuracy_score(true, pred_vals),
        "acc": acc,
        "kappa": me.cohen_kappa_score(true, pred_vals),
        "jaccard": me.jaccard_score(true, pred_vals, average="macro"),
        "fowlkes_mallows": me.fowlkes_mallows_score(true, pred_vals),
        "mcc": me.matthews_corrcoef(true, pred_vals),
        "report": rep_df,
        "prec_recall": prec_recall_df,
        "roc": roc_df,
    }


def confusion_matrix_df(true, pred, **kwargs) -> pd.DataFrame:
    df = pd.DataFrame(me.confusion_matrix(true, pred, **kwargs))
    if "labels" not in kwargs:
        labels = np.unique(true)
        df.columns = labels
        df.index = labels
    else:
        df.columns = kwargs["labels"]
        df.index = kwargs["labels"]
    return df


def cross_validate(
    model,
    adata,
    label_col="tumor_type",
    group_col="",
    n_splits=5,
    random_state=RANDOM_STATE,
    balancer: Balancer | None = None,
    corrector: Corrector | None = None,
    transformer: Transformer | None = None,
    trial: optuna.Trial | None = None,
    get_report_val: Callable = lambda x: x["kappa"],
    record_dir: Path | None = None,
) -> dict:
    """Evaluate model performance with cross-validation

    Parameters
    ----------
    trial : optuna trial to report to, for use with objective function
    adata : adata object with filtered and possibly transformed count data
        Passing transformed data here is fine so long as transformations occur
        independently for each sample, otherwise pass a transformer object
    get_report_val : function that takes the results dictionary as input and extracts
        the metric to `report` to the optuna trial. Metric will be used by the pruner
    balancer : Balancer instance that will transform the training data
    record_dir : directory to record fold metadata
    """
    N = adata.copy()
    labels = N.obs[label_col]
    if not group_col:
        cv = ms.StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        splits = cv.split(N.X, labels)
    else:
        cv = ms.StratifiedGroupKFold(
            n_splits=n_splits, random_state=random_state, shuffle=True
        )
        splits = cv.split(N.X, labels, groups=N.obs[group_col])
    cm: dict = {}
    main_metrics = {"report": [], "prec_recall": [], "roc": []}
    others: dict = {
        "fold": [],
        "acc": [],
        "jaccard": [],
        "kappa": [],
        "balanced_acc": [],
        "fowlkes_mallows": [],
        "mcc": [],
    }
    misclassified: list = []
    for fold, (train_i, test_i) in enumerate(splits):
        x_train: ad.AnnData = N[train_i]
        if corrector is not None:
            x_train = corrector.fit_transform(x_train)
        if balancer is not None:  # Avoid data leakage
            x_train = balancer.fit_transform(x_train)  # Creates copy
        x_test: ad.AnnData = N[test_i]

        if transformer is not None:
            x_train = transformer.fit_transform(x_train)
            x_test = transformer.fit_transform(x_test)

        model.fit(x_train, y=label_col)

        y_true = labels.iloc[test_i]  # True values
        if record_dir is not None:
            x_train.obs.to_csv(
                record_dir.joinpath(f"train_set_{fold}.csv"), index=False
            )
            x_test.obs.to_csv(record_dir.joinpath(f"test_set_{fold}.csv"), index=False)

        if model.score_fn == "predict_proba":
            score = model.predict_proba(x_test)
        elif model.score_fn == "decision_function":
            score = model.decision_function(x_test)
        else:
            raise AttributeError("Model has no way of getting scores!")

        if model.had_inf and record_dir is not None:
            record_dir.joinpath("model_had_inf.log").touch(exist_ok=True)

        res: dict = get_all_metrics(true=y_true, score=score, classes=model.classes_)
        misses: pd.DataFrame = get_misses(x_test, y_true, res["pred"])
        if misses.shape[0] > 0:
            misses.loc[:, "fold"] = fold
            misclassified.append(misses)
        others["fold"].append(fold)

        for o in others.keys():
            if o != "fold":
                others[o].append(res[o])
        cm[fold] = res["cm"]
        for k, v in main_metrics.items():
            df = res[k]
            if df is not None:
                df["fold"] = fold
                v.append(df)

        if trial is not None:
            trial.report(get_report_val(res), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return {
        "cm": cm,
        "misc": pd.DataFrame(others),
        "misses": pd.concat(misclassified, ignore_index=True)
        if misclassified
        else pd.DataFrame(),
    } | {k: pd.concat(v) for k, v in main_metrics.items() if v}


def get_misses(adata: ad.AnnData, true: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    if isinstance(true, pd.Series):
        true = true.values
    if isinstance(pred, pd.Series):
        pred = pred.values
    if adata.shape[0] != true.shape[0]:
        raise ValueError("The shapes of the adata object and true values don't match!")
    copy = adata.obs.copy()
    copy["prediction"] = pred
    return copy.loc[true != pred, :]


def holdout(
    model,
    adata: ad.AnnData,
    split_fns: dict[str, Callable[[ad.AnnData], tuple[ad.AnnData, ad.AnnData]]]
    | None = None,
    balancer: Balancer | None = None,
    transformer: Transformer | None = None,
    corrector: Corrector | None = None,
    apply_correction_to: Literal["train", "test", "both"] = "train",
    label_col="tumor_type",
    save_split_path: Path | None = None,
    split_masks: dict[str, tuple] | None = None,
    verbose: bool = False,
    minimal: bool = False,
) -> dict:
    """Wrapper function for doing the classic holdout method (train-test-split)

    Parameters
    ---------
    split_fn: A dictionary of function that splits adata into a tuple of train, test
    split_indices: A dictionary of mapping test set names to (train, test) boolean indices

    Return
    ------
    A dictionary containing model evaluation results for each unique instance of
        `group_col`

    Notes
    -----
    - Only use in place of cross_validate with StratifiedGroupKFold
        when the group category to be evaluated is
        confounded with the target labels
    """
    if split_fns is None and split_masks is None:
        raise ValueError("Either split_fns or split_indices must be given!")

    split_is_fn: bool = split_fns is not None

    def helper(set_label, splitter, cur_adata):
        cur_adata = cur_adata.copy()
        n = len(cur_adata)
        if split_is_fn:
            x_train, x_test = splitter(cur_adata)
        else:
            x_train = cur_adata[splitter[0], :]
            x_test = cur_adata[splitter[1], :]
        if verbose:
            print(
                f"Train, test sizes for set {set_label}: {x_train.shape[0]}, {x_test.shape[0]}"
            )
        if save_split_path is not None:
            x_train.obs.to_csv(
                save_split_path.joinpath(f"{set_label}_train_obs.csv"), index=False
            )
            x_test.obs.to_csv(
                save_split_path.joinpath(f"{set_label}_test_obs.csv"), index=False
            )
        split_prop_tmp = np.array([len(x_train), len(x_test)]) / n
        split_prop = pd.DataFrame(
            {
                "train_prop": split_prop_tmp[0],
                "test_prop": split_prop_tmp[1],
                "train_size": len(x_train),
                "test_size": len(x_test),
            },
            index=[0],
        )
        if corrector is not None and apply_correction_to in {"both", "train"}:
            x_train = corrector.fit_transform(x_train)
        if corrector is not None and apply_correction_to in {"both", "test"}:
            x_test = corrector.fit_transform(x_test)
        if balancer is not None:
            x_train = balancer.fit_transform(x_train, y=label_col)
        if transformer is not None:
            x_train = transformer.fit_transform(x_train)
            x_test = transformer.fit_transform(x_test)
        model.fit(x_train, y=label_col)
        y_true = x_test.obs[label_col]
        if not minimal:
            proba = model.predict_proba(x_test)
            y_uniques = y_true.unique()
            res: dict = get_all_metrics(
                true=y_true, score=proba, classes=model.classes_
            )
            for k, v in res.items():
                if isinstance(v, pd.DataFrame) and v.shape[0] > 0:
                    if k == "cm":
                        continue
                    res[k] = v.loc[v["class"].isin(y_uniques), :]

            res["misses"] = get_misses(x_test, y_true, res["pred"])
            res["split_prop"] = split_prop
            return res
        pred = model.predict(x_test)
        return {"acc": met.accuracy_score(y_true=y_true, y_pred=pred)}

    dfs = {"report": [], "roc": [], "prec_recall": [], "split_prop": [], "misses": []}
    misc_tmp = {
        "test_set": [],
        "acc": [],
        "balanced_acc": [],
        "kappa": [],
        "jaccard": [],
        "fowlkes_mallows": [],
        "mcc": [],
    }
    cms = {}
    minimal_accs = {}
    splitters: dict = split_fns if split_fns is not None else split_masks
    for set_label, splitter in splitters.items():
        try:
            cur = helper(set_label, splitter, adata)
            if minimal:
                minimal_accs[set_label] = cur
        except RRuntimeError as e:
            print("Error in R runtime: ", e)
            print("ignoring...")
            continue
        cms[set_label] = cur["cm"]
        for m in misc_tmp.keys():
            if m == "test_set":
                continue
            misc_tmp[m].append(cur[m])
        misc_tmp["test_set"].append(set_label)
        for d in dfs.keys():
            df = cur[d]
            if df is not None:
                df["test_set"] = set_label
                dfs[d].append(df)
    if not minimal:
        concat = {
            d: pd.concat(v, ignore_index=True) for d, v in dfs.items() if len(v) > 0
        }
        concat["cm"] = cms
        concat["misc"] = pd.DataFrame(misc_tmp)
        return concat
    return minimal_accs


def write_cross_val(cv_results, outdir, prefix, cm_prefix: str = ""):
    """Helper function for saving cross-validation results"""
    for name, item in cv_results.items():
        if name != "cm" and isinstance(item, pd.DataFrame):
            item.to_csv(outdir.joinpath(f"{prefix}{name}.csv"), index=False)
        elif name == "cm":
            for lab, cm in item.items():
                cm.to_csv(outdir.joinpath(f"{prefix}-cm{cm_prefix}{lab}.csv"))


def fit_train_write(
    model,
    train: ad.AnnData,
    test: ad.AnnData,
    outdir: Path,
    name: str,
    y: str = "tumor_type",
) -> None:
    model.fit(train, y)
    proba = model.predict_proba(test)
    perf = get_all_metrics(test.obs[y], proba, model.classes_)
    test.obs.assign(prediction=perf["pred"]).to_csv(
        outdir.joinpath(f"{name}.csv"), index=False
    )
    write_metrics(outdir.joinpath(f"{name}.txt"), perf)


def write_metrics(
    path: Path, metrics: dict, selection=("acc", "balanced_acc", "kappa")
):
    txt = [f"{m}: {metrics.get(m)}" for m in selection]
    path.write_text("\n".join(txt))


def summarize_studies(study: optuna.Study, objective_name: str) -> pd.DataFrame:
    tmp = {}
    seen_params = set()
    for trial in study.trials:
        pdict = trial.params
        if set(pdict.keys()) <= seen_params:
            continue
        for p in pdict.keys():
            seen_params |= set(trial.params.keys())
            tmp[p] = []
    tmp["n"] = []
    vkey = f"objective_value-{objective_name}"
    tmp[vkey] = []
    for trial in study.trials:
        for param in seen_params:
            tmp[param].append(trial.params.get(param))
        tmp["n"].append(trial.number)
        tmp[vkey].append(trial.value)
    return pd.DataFrame(tmp)


def agg_lr_coefs(
    cv_results: dict,
    feature_names: Sequence | None = None,
    scale: bool = False,
    X: np.ndarray | None = None,
    agg_fn: Literal["std", "var"] = "std",
) -> pd.DataFrame:
    tmp = {}
    if "estimator" not in cv_results:
        raise ValueError("cross_validate must be run with return_estimator=True")
    classes = cv_results["estimator"][0].classes_
    for cls in classes:
        vals = []
        for j, est in enumerate(cv_results["estimator"]):
            coef = est.coef_[j, :]
            if scale and X is not None:
                coef = coef * est.transform(X).std(axis=0)
            elif scale and X is None:
                raise ValueError("X must be provided if scale")
            vals.append(coef)
        if agg_fn == "std":
            agg = np.array(vals).std(axis=0)
        elif agg_fn == "var":
            agg = np.array(vals).var(axis=0)
        tmp[cls] = agg
    return pd.DataFrame(tmp, index=feature_names)


# * Effective robustness


class Robustness:
    """Helper class to compute effective and relative robustness

    Parameters
    ----------
    train : Dataset to rain models on
    shifted_test : test dataset consisting of data from the shifted distribution
    standard_test : test dataset consisting of data from the same distribution as train
    n_classes : int
        number of classes in the dataset
    y_idx : for sklearn models, the index of the y array to use for training (in
        the case where the datasets are multitask)
    """

    def __init__(
        self,
        train: Dataset,
        shifted_test: Dataset,
        standard_test: Dataset,
        n_classes: int,
        beta_path: Path | None = None,
        y_idx: int | None = None,
    ) -> None:
        self._train: Dataset = train
        self._shifted_test: Dataset = shifted_test
        self._standard_test: Dataset = standard_test
        self.beta: LinearRegression | None | Path = None
        if beta_path is not None and beta_path.exists():
            with open(beta_path, "r") as f:
                self.beta = pickle.load(f)
        self._y_idx: None | int = y_idx
        self._n_classes: int = n_classes

        # Numpy ndarray copies of data
        self._has_numpy: bool = False
        self._train_x: np.ndarray | None = None
        self._train_y: np.ndarray | None = None
        self._shifted_x: np.ndarray | None = None
        self._shifted_y: np.ndarray | None = None
        self._standard_x: np.ndarray | None = None
        self._standard_y: np.ndarray | None = None

    def _validate_spec(
        self,
        model_fn: Callable,
        numpy: bool | None = None,
        train_fn: Callable | None = None,
        pretrained: bool | None = None,
    ) -> None:
        """Validate dictionary used to construct and train a model

        Parameters
        ----------
        model_fn : Callable returning a pytorch module or sklearn model. May or may not
            be fitted
        numpy : if given, model is taken to be an sklearn model and will be given the
            train and test data as numpy arrays
        train_fn : the training function for pytorch modules. Must take a dataset
        pretrained : if the model has been fit already. If true, then it will not
            be fit with ``self.train``
        """
        if numpy is None and train_fn is None:
            raise ValueError(
                "If the model isn't expected to take a numpy array, train_fn must be provided!"
            )

    def _fit(self, model, spec: dict) -> None:
        if not spec.get("pretrained"):
            if spec.get("numpy"):
                model.fit(self._train_x, self._train_y)
            else:
                train = spec["train_fn"]
                train(model, self._train)

    def _acc(
        self,
        model,
        spec: dict,
        test_dset: Dataset | None = None,
        test_x: np.ndarray | None = None,
        test_y: np.ndarray | None = None,
    ) -> float:
        if spec.get("numpy"):
            preds = model.predict(test_x)
            return met.accuracy_score(y_true=test_y, y_pred=preds)
        else:
            preds = model(test_dset[:][0])
            return accuracy(
                preds=preds,
                target=test_y,
                num_classes=self._n_classes,
                task="multiclass",
            ).item()

    def _acc_pair(
        self, model, spec: dict, with_standard: bool = True
    ) -> tuple[float, float]:
        """Helper function to compute the standard and shifted accuracy of the model
        defined in ```spec```
        """
        standard_acc = 0.0
        if spec.get("numpy"):
            if with_standard:
                standard_acc = self._acc(
                    model, test_x=self._standard_x, test_y=self._standard_y
                )
            shifted_acc = self._acc(
                model, test_x=self._shifted_y, test_y=self._shifted_x
            )
        else:
            if with_standard:
                standard_acc = self._acc(model, test_dset=self._standard_test)
            shifted_acc = self._acc(model, test_dset=self._shifted_test)
        return standard_acc, shifted_acc

    def effective_robustness(self, mspec: dict) -> float:
        self._validate_spec(**mspec)
        if not self.beta:
            self.beta = self.get_beta()
        model = mspec["model_fn"]()
        self._set_np([mspec])
        self._fit(model, mspec)
        standard_acc, shifted_acc = self._acc_pair(model, mspec)
        return shifted_acc - self.beta.predict(standard_acc)

    def relative_robustness(self, ispec: dict, mspec: dict) -> float:
        """Compute relative robustness

        Parameters
        ----------
        ispec : dict
            model spec dictionary with robustness intervention
        mspec : dict
            standard model spec dictionary

        Returns
        -------
        float : Relative Robustness score
        """
        self._validate_spec(**ispec)
        self._validate_spec(**mspec)
        self._set_np([ispec, mspec])
        imodel = ispec["model_fn"]()
        self._fit(imodel, ispec)
        _, intervention_acc = self._acc_pair(imodel, ispec, False)

        mmodel = mspec["model_fn"]()
        self._fit(mmodel, mspec)
        _, standard_acc = self._acc_pair(mmodel, mspec, False)
        return intervention_acc - standard_acc

    def _set_np(self, specs: Sequence[dict]) -> None:
        """Instantiate numpy arrays"""
        has_numpy: bool = any([m.get("numpy") for m in specs])
        if has_numpy and not self._has_numpy:
            self._train_x, self._train_y = [x.numpy() for x in self._train[:]]
            self._shifted_x, self._shifted_y = [
                x.numpy() for x in self._shifted_test[:]
            ]
            self._standard_x, self._standard_y = [
                x.numpy() for x in self._standard_test[:]
            ]
            if self._y_idx is not None:
                self._train_y = self._train_y[:, self._y_idx]
                self._shifted_y = self._shifted_y[:, self._y_idx]
                self._standard_y = self._standard_y[:, self._y_idx]
            self._has_numpy = True

    def get_beta(
        self, models: Sequence[dict], save_to: Path | str | None = None
    ) -> LinearRegression:
        """Compute accuracies on standard dataset to find model with which to calculate
            beta

        Parameters
        ----------
        models : Sequence
            A sequence of model dictionaries - see _validate_spec for the required keys
            The models specified here should be standard models with no robustness
            intervention
        save_to : Path
            Path to save fitted model to

        Returns
        -------
        LinearRegression fitted to compute baseline accuracies
        """
        self._set_np(models)
        vals: dict = {"name": [], "standard_acc": [], "shifted_acc": []}
        for i, spec in enumerate(models):
            self._validate_spec(spec)
            model = spec["model_fn"]()
            name = spec.get("name", f"model_{i}")
            self._fit(model, spec)
            shifted_acc, standard_acc = self._acc_pair(model, spec)
            vals["shifted_acc"].append(shifted_acc)
            vals["standard_acc"].append(standard_acc)
            vals["name"].append(name)
        beta = LinearRegression()
        beta.fit(vals["standard_acc"], y=vals["shifted_acc"])
        if save_to:
            if isinstance(save_to, str):
                save_to = Path(save_to)
            with open(save_to, "wb") as f:
                pickle.dump(beta, f)
            metrics = pd.DataFrame(vals)
            metric_file = save_to.parent.joinpath(f"{save_to.stem}.csv")
            metrics.to_csv(metric_file, index=False)
        return beta
