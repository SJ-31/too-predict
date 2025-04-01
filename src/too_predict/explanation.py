#!/usr/bin/env ipython
from pathlib import Path
from typing import Callable

import alibi.explainers as ae
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as sd
import shap
from scipy import sparse

from too_predict.utils import RANDOM_STATE


class ExpInterpreter:
    """Class for interpreting feature explanations (importance/rankings)

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
        self.tkey, self.pkey = true_pred
        self.label_col = label_col
        self.labels: pd.Series = train_importances.obs[label_col].unique()

    def _importance_consistency(
        self, adata: ad.AnnData, labels, summary: str = "std"
    ) -> dict[str, pd.DataFrame | None] | pd.DataFrame | None:
        """Quantify the consistency of feature importances across samples

        Parameters
        ----------
        summary : function used to measure the spread of importance values
            `counts` : proportion of samples in which the feature is positive, negative, zero
            `std` : standard deviation of importance across samples
            `range` : range of importance values

        Returns
        -----
        One of
            a single df of consistency results if len(label) == 1
            a dictionary mapping label to consistency results, if `labels` is a list
            None if all calculations for each feature failed
        """

        def one_label(label: str):
            vals = adata.obsm[self.local_getter(label)]
            dfs = []
            if summary == "counts":
                directions = ["positive", "negative", "zero"]
                for fn in [lambda x: x > 0, lambda x: x < 0, lambda x: x == 0]:
                    applied = fn(vals)
                    percentage = (applied.sum(axis=0) / vals.shape[0]) * 100
                    dfs.append(percentage)
                df = pd.concat(dfs, axis=1)
                df.loc[:, "label"] = label
                df.columns = directions
            elif summary == "std":
                df = pd.DataFrame({label: np.nanstd(vals, axis=0)}, index=vals.columns)
            elif summary == "range":
                max = np.max(vals, axis=0)
                min = np.min(vals, axis=0)
                df = pd.DataFrame({label: max - min}, index=vals.columns)
            else:
                df = pd.DataFrame({label: np.nanvar(vals, axis=0)}, index=vals.columns)
            if np.all(df.isna()):
                return None
            return df

        if isinstance(labels, str):
            return one_label(labels)
        result = {}
        for label in labels:
            df = one_label(label)
            if df is not None:
                result[label] = df
        return result

    def _split_three(
        self, df: pd.DataFrame, col: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pos = df.loc[df.loc[:, col] > 0, :]
        neg = df.loc[df.loc[:, col] < 0, :]
        zero = df.loc[df.loc[:, col] == 0, :]
        return pos, neg, zero

    def _local_right_wrong(
        self,
        adata: ad.AnnData,
        label: str,
        agg_fn: Callable = lambda x: np.median(x, axis=1),
    ) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
        vals = adata.obsm[self.local_getter(label)]

        wrong_mask = adata.obs[self.tkey].astype(str) != adata.obs[self.pkey].astype(
            str
        )
        right_mask = adata.obs[self.tkey].astype(str) == adata.obs[self.pkey].astype(
            str
        )
        wrong = np.transpose(vals.loc[wrong_mask, :])
        right = np.transpose(vals.loc[right_mask, :])
        wrong.loc[:, "agg"] = agg_fn(wrong)
        right.loc[:, "agg"] = agg_fn(right)

        return right, wrong

    def _negative_contributions(
        self, n: int = -1, strict: bool = False
    ) -> tuple[set[str], dict]:
        def i2s(df: pd.DataFrame) -> set[str]:
            return set(df.index)

        def one_label(label) -> tuple[set[str], set[str], set[str]]:
            right, wrong = self._local_right_wrong(self.test_vals, label)
            right_train, _ = self._local_right_wrong(
                self.train_vals[
                    (self.train_vals.obs[self.tkey] == label)
                    & (self.train_vals.obs[self.pkey] == label)
                ],
                label,
            )
            wrong_p, wrong_n, wrong_0 = self._split_three(wrong, "agg")

            wrong_n = wrong_n.sort_values("agg").iloc[:n, :]

            right_p, right_n, right_0 = self._split_three(right, "agg")
            rtrain_p, rtrain_n, rtrain_0 = self._split_three(right_train, "agg")

            all_zero = i2s(rtrain_0) | i2s(right_0)
            pos_contrib = i2s(right_p) | i2s(rtrain_p) | i2s(wrong_p)
            neg_contrib = i2s(wrong_n) & (i2s(right_n) | i2s(rtrain_n))
            neg_contrib -= pos_contrib

            return pos_contrib, neg_contrib, all_zero

        label_specific = {}

        combined_neg = set()
        combined_0 = set()
        combined_pos = set()
        for label in self.labels:
            pos, neg, zero = one_label(label)
            label_specific[label] = neg
            combined_neg |= neg
            combined_pos |= pos
            combined_0 |= zero
        combined_neg = combined_neg & combined_0
        if strict:
            combined_neg -= combined_pos  # Will discard features that are considered
            # positive in the context of classifying other labels
        return combined_neg, label_specific

    def shap_neg_contributions(self, n: int = -1) -> tuple[set[str], dict]:
        self.local_getter = lambda x: f"shap_{x}"
        result = self._negative_contributions(n)
        self.local_getter = None
        return result

    def shap_consistency(
        self, right_wrong: bool = True, summary: str = "std"
    ) -> tuple[dict, dict]:
        self.local_getter = lambda x: f"shap_{x}"
        results: dict = {}
        stats: dict = {}
        for g, adata in zip(["train", "test"], [self.train_vals, self.test_vals]):
            if right_wrong:
                results[g] = {}
                stats[g] = {self.label_col: [], "right": [], "wrong": []}
                tmp_r, tmp_w = [], []
                for label in self.labels:
                    stats[g][self.label_col].append(label)
                    r, w = self._local_right_wrong(adata, label)

                    right = adata[adata.obs.index.isin(r.columns)]
                    stats[g]["right"].append(right.shape[0])

                    wrong = adata[adata.obs.index.isin(w.columns)]
                    stats[g]["wrong"].append(wrong.shape[0])
                    rdf = self._importance_consistency(right, label, summary)
                    if rdf is not None:
                        tmp_r.append(rdf)
                    wdf = self._importance_consistency(wrong, label, summary)
                    if wdf is not None:
                        tmp_w.append(wdf)
                results[g]["right"] = pd.concat(tmp_r, axis=1) if tmp_r else None
                results[g]["wrong"] = pd.concat(tmp_w, axis=1) if tmp_w else None
                stats[g] = pd.DataFrame(stats[g])
            else:
                results[g] = self._importance_consistency(adata, self.labels)
        self.local_getter = None
        return results, stats

    def _importance_distance(
        self,
        target: str = "test",
        metric: str = "euclidean",
        agg_fn: Callable = lambda x: np.median(x, axis=0),
        square: bool = True,
    ) -> pd.DataFrame | np.ndarray:
        adata = self.test_vals if target == "test" else self.train_vals
        tmp = [
            pd.DataFrame(
                {label: agg_fn(adata.obsm[self.local_getter(label)])},
                index=adata.var.index,
            )
            for label in self.labels
        ]
        df = np.transpose(pd.concat(tmp, axis=1))
        dist: np.ndarray = sd.pdist(df, metric=metric)
        if square:
            return pd.DataFrame(
                sd.squareform(dist), index=self.labels, columns=self.labels
            )
        return dist

    def shap_distance(self, **kwargs):
        self.local_getter = lambda x: f"shap_{x}"
        result = self._importance_distance(**kwargs)
        self.local_getter = None
        return result


# def shap_adata(adata: ad.AnnData):


class Exp:
    """Helper class for unifying feature explanations

    Parameters
    ----------
    model : fitted model
    """

    def __init__(
        self,
        model,
        adata: ad.AnnData,
        label_col: str = "tumor_type",
        feature_col: str = "GENEID",
    ) -> None:
        self.adata: ad.AnnData
        self.features: np.ndarray
        self.y_true: np.ndarray
        self.model = model
        self.label_col = label_col
        self.feature_col = feature_col
        self.class2index = dict(
            zip(self.model.classes_, range(len(self.model.classes_)))
        )
        self.new_adata(adata)

    def new_adata(self, adata: ad.AnnData):
        self.adata = adata
        self.features = adata.var[self.feature_col]
        self.y_true = adata.obs[self.label_col]

    def _count_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.adata.X
            if not sparse.isspmatrix(self.adata.X)
            else self.adata.X.toarray(),
            columns=self.features,
        )

    def _into_obsm(self, adata, prefix, importances: dict[str, np.ndarray]):
        for cls, i_vals in importances.items():
            adata.obsm[f"{prefix}{cls}"] = pd.DataFrame(
                i_vals, index=adata.obs.index, columns=adata.var.index
            )

    @staticmethod
    def _make_empty(adata: ad.AnnData, y_true, y_pred) -> ad.AnnData:
        empty = ad.AnnData(var=adata.var, obs=adata.obs)
        empty.obs.loc[:, "y_true"] = y_true
        empty.obs.loc[:, "y_pred"] = y_pred
        return empty

    def anchor(self, adata: ad.AnnData):
        features = adata.var[self.feature_col]
        explainer = ae.AnchorTabular(
            predictor=lambda x: self.model.get_model().predict_proba(x),
            feature_names=features,
            seed=RANDOM_STATE,
        )

    def shap(
        self,
        explain_fn,
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
        y_pred: np.ndarray = self.model.predict(self.adata)
        explainer: shap.Explainer = explain_fn(self.model.get_model())
        counts = self._count_df()
        empty: ad.AnnData = self._make_empty(self.adata, self.y_true, y_pred)

        svals: np.ndrray = explainer.shap_values(counts)
        imatrix: np.ndarray | None = None
        explanation = shap.Explanation(
            svals, base_values=counts, feature_names=self.features
        )
        if plot_feature_col:
            p_features = self.adata.var[plot_feature_col].combine_first(self.features)
            counts.columns = p_features
        if interaction_matrix and isinstance(explainer, shap.TreeExplainer):
            imatrix = explainer.shap_interaction_values(counts)
        class2shap = {c: svals[:, :, i] for (c, i) in self.class2index.items()}
        if plot_directory is not None:
            plot_directory.mkdir(parents=True, exist_ok=True)
            for c, vals in class2shap.items():
                if summary_plot:
                    shap.summary_plot(vals, counts)
                    plt.savefig(
                        plot_directory.joinpath(f"{c}_summary.png"),
                        dpi=500,
                        bbox_inches="tight",
                    )
                    plt.close()
        self._into_obsm(empty, "shap_", class2shap)
        empty.uns["shap_interaction_matrix"] = imatrix
        return empty, explanation


def get_most_important(
    adata: ad.AnnData,
    n: int = 20,
    agg: str = "median",
    prefix: str = "shap_",
    label_col: str = "tumor_type",
) -> pd.DataFrame:
    """Get the most important features

    Parameters
    ----------
    prefix : prefix of adata.obsm df storing the per-class feature importances

    Returns
    -------
    dataframe of shape n x n_classes, where each column contains the most important
    features for that class assignment
    """
    tmp = []
    for label in adata.obs[label_col].unique():
        vals = adata.obsm[f"{prefix}{label}"]
        if agg == "median":
            vals = vals.median(axis=0)
        elif agg == "mean":
            vals = vals.mean(axis=0)
        else:
            raise ValueError(f"`agg` argument {agg} not recognized!")
        vals = vals.abs().sort_values(ascending=False).iloc[:n]
        tmp.append(pd.DataFrame({label: vals.index}))
    return pd.concat(tmp, axis=1)
