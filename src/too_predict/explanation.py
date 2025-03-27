#!/usr/bin/env ipython
from pathlib import Path
from typing import Callable

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse


class FindFeatures:
    """Class for analyzing feature importance/rankings

    The adata objects `train_importances`, `test_importances` are expected to contain
        local feature importances (i.e. per sample) in obsm.


    Parameters
    ----------
    local_getter : a function that takes in a label/class instance and returns
        the key in obsm containing the importances for that specific class
        In multiclass settings, feature importance is typically measured as the
        degree to which the feature affects the probability/score of a given class
        assignment, so there will be as many layers here as classes

    """

    def __init__(
        self,
        train_importances: ad.AnnData,
        test_importances: ad.AnnData,
        label_col: str = "tumor_type",
        true_pred: tuple = ("y_true", "y_pred"),
    ) -> None:
        self.local_getter: Callable[[str], str]
        self.test_vals = test_importances
        self.train_vals = train_importances
        self.true_key, self.pred_key = true_pred
        self.label_col = label_col

    def _split_three(
        self, df: pd.DataFrame, col: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pos = df.loc[df.loc[:, col] < 0, :]
        neg = df.loc[df.loc[:, col] > 0, :]
        zero = df.loc[df.loc[:, col] == 0, :]
        return pos, neg, zero

    def _local_right_wrong(
        self,
        adata: ad.AnnData,
        label: str,
        agg_fn: Callable = lambda x: np.median(x, axis=1),
    ) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
        vals = adata.obsm[self.local_getter(label)]
        wrong = np.transpose(
            vals.loc[adata.obs[self.true_key] != adata.obs[self.pred_key], :]
        )
        right = np.transpose(
            vals.loc[adata.obs[self.true_key] == adata.obs[self.pred_key], :]
        )
        wrong.loc[:, "agg"] = agg_fn(wrong)
        right.loc[:, "agg"] = agg_fn(right)
        return right, wrong

    def _negative_contributions(
        self,
        n: int = -1,
    ) -> tuple[set[str], dict]:
        def i2s(df: pd.DataFrame) -> set[str]:
            return set(df.index)

        def one_label(label) -> tuple[set[str], set[str], set[str]]:
            right, wrong = self._local_right_wrong(self.test_vals, label)
            right_train, _ = self._local_right_wrong(
                self.train_vals[
                    (self.train_vals.obs[self.true_key] == label)
                    & (self.train_vals.obs[self.pred_key] == label)
                ],
                label,
            )
            wrong_p, wrong_n, wrong_0 = self._split_three(wrong, "agg")

            wrong_n = wrong_n.sort_values("agg").iloc[:n, :]

            right_p, right_n, right_0 = self._split_three(right, "agg")
            right_train_p, _, right_train_0 = self._split_three(right_train, "agg")

            neg_contrib = (i2s(wrong_n) - i2s(right_p)) & (
                i2s(right_n) | i2s(right_0) | i2s(right_train_0) | i2s(right_train_p)
            )

            all_zero = i2s(right_train_0) | i2s(right_0) | i2s(wrong_0)
            pos_contrib = i2s(right_p) | i2s(right_train_p) | i2s(wrong_p)

            return pos_contrib, neg_contrib, all_zero

        label_specific = {}

        combined_neg = set()
        combined_0 = set()
        combined_pos = set()
        for label in self.train_vals.obs[self.label_col].unique():
            pos, neg, zero = one_label(label)
            label_specific[label] = neg
            combined_neg |= neg
            combined_pos |= pos
            combined_0 |= zero
        combined_neg = combined_neg & combined_0
        return combined_neg, label_specific

    def shap_neg_contributions(self, n: int = -1) -> tuple[set[str], dict]:
        self.local_getter = lambda x: f"shap_{x}"
        result = self._negative_contributions(n)
        self.local_getter = None
        return result


def get_shap_adata(
    adata: ad.AnnData,
    explainer: shap.Explainer,
    classifier,
    label_col: str = "tumor_type",
    feature_col: str = "GENEID",
    summary_plot: bool = True,
    interaction_matrix: bool = False,
    plot_feature_col: str = "GENENAME",
    plot_directory: Path | None = None,
) -> tuple[ad.AnnData, shap.Explanation]:
    """Get shapley values for the samples in `adata`

    Parameters
    ----------
    classifier : pre-fitted classifier
    explainer : shap.Explainer pre-fitted on `classifier`
    summary_plot : path to directory to save summary plots. A summary plot
        will be made for each available class

    Returns
    -------
    1. Adata object storing the shapley values for each class in adata.obsm[shap_{class}]
        Each of these is a matrix of shape n_samples x n_features
    2. shap explanation object
    """
    classes = classifier.classes_
    y_true: np.ndarray = adata.obs[label_col]
    y_pred: np.ndarray = classifier.predict(adata)
    class2index = dict(zip(classes, range(len(classes))))
    features = adata.var[feature_col]
    counts = pd.DataFrame(
        adata.X if not sparse.isspmatrix(adata.X) else adata.X.toarray(),
        columns=features,
    )
    empty = ad.AnnData(var=adata.var, obs=adata.obs)
    empty.obs.loc[:, "y_true"] = y_true
    empty.obs.loc[:, "y_pred"] = y_pred

    svals: np.ndrray = explainer.shap_values(counts)
    imatrix: np.ndarray | None = None
    explanation = shap.Explanation(svals, base_values=counts, feature_names=features)
    if plot_feature_col:
        p_features = adata.var[plot_feature_col].combine_first(features)
        counts.columns = p_features
    if interaction_matrix and isinstance(explainer, shap.TreeExplainer):
        imatrix = explainer.shap_interaction_values(counts)
    if plot_directory is not None:
        plot_directory.mkdir(parents=True, exist_ok=True)
        for c, i in class2index.items():
            if summary_plot:
                shap.summary_plot(svals[:, :, i], counts)
                plt.savefig(
                    plot_directory.joinpath(f"{c}_summary.png"),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    for clss, i in class2index.items():
        empty.obsm[f"shap_{clss}"] = pd.DataFrame(
            svals[:, :, i], index=empty.obs.index, columns=empty.var.index
        )
    empty.uns["shap_interaction_matrix"] = imatrix
    return empty, explanation
