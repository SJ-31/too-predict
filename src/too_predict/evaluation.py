#!/usr/bin/env ipython
from typing import Callable

import numpy as np
import pandas as pd
import sklearn.metrics as me


def curve_multiclass(
    curve_fn: Callable,
    first: str,
    second: str,
    true_labels,
    pred_proba: pd.DataFrame | np.ndarray,
    s_fn: Callable = None,
    s_fn_name: str = "",
    s_fn_all_classes: Callable = None,
    s_fn_all_classes_name: str = "",
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

    s_fn : Callable[true, score, labels], a function supporting multi-class classification
        that returns an iterable

    s_fn_all_classes : a function supporting multi-class classification returning
        a single value representing performance for all classes

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
    if s_fn is not None:
        summary = s_fn(true_labels, prob_df, prob_df.columns)
        summary_labels: dict = {
            c: f"{c}: {v.round(3)}" for c, v in zip(prob_df.columns, summary)
        }
    if s_fn_all_classes is not None:
        summary_all = s_fn_all_classes(true_labels, prob_df, prob_df.columns)
    dfs = []
    for c in prob_df.columns:
        f, s, thresholds = curve_fn(true_labels, prob_df[c], c)
        if len(thresholds) < len(f):
            thresholds = np.concatenate([[np.inf], thresholds])
        tmp = pd.DataFrame({first: f, second: s, "thresholds": thresholds})
        tmp["class"] = c
        if s_fn and s_fn_name:
            tmp[s_fn_name] = summary_labels[c]
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
        s_fn=lambda true, score, labels: me.roc_auc_score(
            true, score, average=None, multi_class="ovr", labels=labels
        ),
        s_fn_name="class_auc",
        s_fn_all_classes=lambda true, score, labels: me.roc_auc_score(
            true, score, average=average, multi_class=multi_class, labels=labels
        ),
        s_fn_all_classes_name=f"{average}_{multi_class}_auc",
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
        s_fn=lambda true, score, _: me.average_precision_score(
            true, score, average=None
        ),
        s_fn_name="average_precision",
        s_fn_all_classes=lambda true, score, _: me.average_precision_score(
            true, score, average=average
        ),
        s_fn_all_classes_name=f"{average}_average_precision",
    )


def classification_report2df(
    report: dict, fold: int = None
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


def get_all_metrics(true, proba, classes) -> dict:
    predictions = pd.DataFrame(proba, columns=classes)
    pred_vals = predictions.idxmax(1)
    rep = me.classification_report(true, pred_vals, output_dict=True)

    cm = confusion_matrix_df(true, pred_vals, labels=classes)
    rep_df, acc = classification_report2df(rep)
    roc_df = roc_multiclass(true, predictions)
    prec_recall_df = precision_recall_multiclass(true, predictions)
    return {
        "cm": cm,
        "acc": acc,
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
