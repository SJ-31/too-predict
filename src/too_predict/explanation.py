#!/usr/bin/env ipython
from pathlib import Path
from typing import Callable

import alibi.api.interfaces as interfaces
import alibi.explainers as ae
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as sd
import shap
from matplotlib.figure import Figure
from scipy import sparse
from sklearn.decomposition import PCA

import too_predict.plotting as plotting
from too_predict.model import PredBase
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
        self.labels_test: pd.Series = test_importances.obs[label_col].unique()

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
        # These dfs have shape n_features x n_samples + 1

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

    def global_importance(
        self, prefix: str, correct_only: bool = False
    ) -> pd.DataFrame:
        """Average the feature importance across every instance,

        Returns
        -------
        A df of shape n_features x n_labels, where each entry is the average
        absolute feature importance of that feature for that label
        """
        self.local_getter = lambda x: f"{prefix}{x}"
        tmp = []
        for label in self.labels:
            tmp_inner = []
            for i, adata in enumerate([self.test_vals, self.train_vals]):
                right, wrong = self._local_right_wrong(
                    adata, label, lambda x: np.nanmean(np.abs(x), axis=1)
                )
                vals = right.loc[:, "agg"]
                if not correct_only:
                    vals = (vals + wrong.loc[:, "agg"]) / 2
                tmp_inner.append(pd.DataFrame({i: vals}, index=vals.index))
            mean = pd.concat(tmp_inner, axis=1).mean(axis=1)
            tmp.append(pd.DataFrame({label: mean}, index=mean.index))
        self.local_getter = None
        df = pd.concat(tmp, axis=1)
        return df

    def neg_contributions(self, prefix: str, n: int = -1) -> tuple[set[str], dict]:
        print("Will retrieve data from the following: ")
        obs_names = [k for k in self.test_vals.obsm.keys() if k.startswith(prefix)]
        print(obs_names)
        self.local_getter = lambda x: f"{prefix}{x}"
        result = self._negative_contributions(n)
        self.local_getter = None
        return result

    def consistency(
        self, prefix: str, right_wrong: bool = True, summary: str = "std"
    ) -> tuple[dict, dict]:
        self.local_getter = lambda x: f"{prefix}{x}"
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

    def instance_pca(
        self,
        prefix: str,
        plot: bool = True,
        subset=None,
        colors=None,
        **kwargs,
    ) -> tuple[np.ndarray, None | Figure]:
        self.local_getter = lambda x: f"{prefix}{x}"
        tmp = []
        for adata in (self.train_vals, self.test_vals):
            for label in self.labels:
                if subset is not None and label not in subset:
                    continue
                mask = adata.obs[self.label_col] == label
                cur = adata[mask, :]
                cur_importance = adata.obsm[self.local_getter(label)]
                if cur_importance.shape[0] != cur.shape[0]:
                    cur_importance = cur_importance.loc[
                        cur_importance.index.isin(cur.obs.index), :
                    ]
                df = pd.DataFrame(
                    cur_importance, index=cur.obs.index, columns=cur.var.index
                )
                tmp.append(df)
        x = pd.concat(tmp, axis=0)

        all_obs = pd.concat(
            [
                self.train_vals.obs.assign(usage="train"),
                self.test_vals.obs.assign(usage="test"),
            ],
            axis=0,
        )
        x, all_obs = x.align(all_obs, join="inner", axis=0)
        pca = PCA(**kwargs)
        result = pca.fit_transform(x)
        if plot:
            if not colors:
                colors = (self.label_col,)
            fig, ax = plt.subplots(ncols=len(colors))
            for i, color in enumerate(colors):
                plotting.plot_pca(result, all_obs[color], ax[i], **kwargs)
            rval = result, fig
        else:
            rval = result, None
        self.local_getter = None
        return rval

    def instance_distances(
        self,
        prefix: str,
        dataset: str = "test",
        metric: str = "euclidean",
        subset=None,
        square: bool = True,
    ) -> dict[str, pd.DataFrame | np.ndarray]:
        """Compute distance between instances on the basis of feature importance
        For each label, collect instances of that label and compute distance between
        them.

        Parameters
        ----------
        dataset : one of test|train|compare. If test or train, compute distances per-label
            within the train or test set.
            If `compare`, compute distance between train and test sets for instances
            of the same label
        subset : restrict to only these labels

        Returns
        -------
        A dict mapping labels to the following
        - If dataset == test|train, a square dataframe containing distances if `square`, else a condensed distance
        matrix
        - if compare, a dataframe with shape train_labels x test_labels
        """
        self.local_getter = lambda x: f"{prefix}{x}"
        result = {}

        def not_compare(cur_adata, label):
            indices = cur_adata.obs[self.label_col] == label
            importances: np.ndarray = cur_adata.obsm[self.local_getter(label)]
            importances = importances[indices]
            dist: np.ndarray = sd.pdist(importances, metric=metric)
            if square:
                result[label] = pd.DataFrame(
                    sd.squareform(dist),
                    index=cur_adata.obs.index[indices],
                    columns=cur_adata.obs.index[indices],
                )
            else:
                result[label] = dist

        def compare(label):
            key = self.local_getter(label)
            if not (self.test_vals.obs[self.label_col] == label).any():
                return
            if key in self.train_vals.obsm and key in self.test_vals.obsm:
                train_indices = self.train_vals.obs[self.label_col] == label
                test_indices = self.test_vals.obs[self.label_col] == label
                train_imp = self.train_vals.obsm[key][train_indices]
                test_imp = self.test_vals.obsm[key][test_indices]
                train_names = self.train_vals.obs.index[train_indices]
                test_names = self.test_vals.obs.index[test_indices]
                dist = sd.cdist(train_imp, test_imp)
                result[label] = pd.DataFrame(
                    dist, index=train_names, columns=test_names
                )

        adata = None
        if dataset != "compare":
            label_set = self.labels_test if dataset == "test" else self.labels
            adata = self.test_vals if dataset == "test" else self.train_vals
        else:
            label_set = self.labels
        for label in label_set:
            if subset is not None and label in subset:
                continue
            if dataset != "compare":
                not_compare(adata, label)
            else:
                compare(label)
        self.local_getter = None
        return result

    def label_distances(
        self,
        prefix: str,
        dataset: str = "test",
        metric: str = "euclidean",
        agg_fn: Callable = lambda x: np.median(x, axis=0),
        square: bool = True,
    ) -> pd.DataFrame | np.ndarray:
        """Compute distances between labels on the basis of their local feature
        importance. For each label, the function aggregates feature importances across
        instances, and computes distance between labels.

        Intended to show the variation in feature importances between labels
            e.g. predictions for class A might rely on different features than class B

        Parameters
        ----------
        agg_fn : function to aggregate feature importance across instances
        target : one of test|train|compare
            When `test` or `train`, the function computes pairwise distances within
            the train or test set
            If `compare`, the function computes pairwise distances between train and test,
               returning an array of train_labels x test_labels

        Returns
        -------
        If dataset == test|train, a square dataframe containing distances if `square`, else a condensed distance
        matrix
        if compare, a dataframe with shape train_labels x test_labels
        """
        self.local_getter = lambda x: f"{prefix}{x}"

        def collect(cur_adata, label_set):
            tmp = [
                pd.DataFrame(
                    {label: agg_fn(cur_adata.obsm[self.local_getter(label)])},
                    index=cur_adata.var.index,
                )
                for label in label_set
            ]
            df = np.transpose(pd.concat(tmp, axis=1))
            return df

        if dataset == "compare":
            v_train = collect(self.train_vals, self.labels)
            v_test = collect(self.test_vals, self.labels_test)
            dist: np.ndarray = sd.cdist(v_train, v_test, metric=metric)
            return pd.DataFrame(dist, index=self.labels, columns=self.labels_test)
        else:
            adata = self.test_vals if dataset == "test" else self.train_vals
            label_set = self.labels_test if dataset == "test" else self.labels
            df = collect(adata, label_set)
            dist: np.ndarray = sd.pdist(df, metric=metric)
            if square:
                return pd.DataFrame(
                    sd.squareform(dist), index=label_set, columns=label_set
                )
        self.local_getter = None
        return dist


# def shap_adata(adata: ad.AnnData):


class Exp:
    """Helper class for unifying feature explanations

    Parameters
    ----------
    model : fitted model
    """

    def __init__(
        self,
        model: PredBase,
        label_col: str = "tumor_type",
        feature_col: str = "GENEID",
    ) -> None:
        self.adata: ad.AnnData
        self.features: np.ndarray
        self.y_true: np.ndarray
        self.model = model
        self.u_model = self.model.get_model()  # Underlying model of PredBase
        self.label_col = label_col
        self.feature_col = feature_col
        self.class2index = dict(
            zip(self.model.classes_, range(len(self.model.classes_)))
        )

    def fit(self, adata: ad.AnnData):
        self.adata = adata
        self.features = adata.var[self.feature_col].astype(str)
        self.y_true = adata.obs[self.label_col]

    def _count_df(self, adata=None) -> pd.DataFrame:
        if adata is None:
            adata = self.adata
        return pd.DataFrame(
            adata.X if not sparse.issparse(adata.X) else adata.X.toarray(),
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

    def anchor(
        self,
        train_data: ad.AnnData,
        adata: ad.AnnData | None = None,
        explain_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
    ) -> tuple[ad.AnnData, pd.DataFrame]:
        if adata is not None:
            self.fit(adata)
        if not explain_kwargs:
            explain_kwargs = {
                "threshold": 0.90,
                "batch_size": 70,
            }
        if not fit_kwargs:
            fit_kwargs = {"disc_perc": (25, 50, 75)}

        def add_anchor_exp(result: interfaces.Explanation, importance_vec: np.ndarray):
            # Define importance of a feature here as the precision increase
            # when adding the feature to the (anchor + coverage) - 1
            # So importance ranges from [-1, 1]
            # [2025-04-01 Tue] might want to change this
            coverage = result.coverage
            precisions = result.raw["precision"]
            feature_indices = result.raw["feature"]
            for i, f_index in enumerate(feature_indices):
                if i == 0:
                    cur_prec = precisions[i]
                else:
                    cur_prec = precisions[i] - precisions[i - 1]
                importance = cur_prec + coverage
                importance_vec[f_index] = importance
            return importance_vec

        explainer: ae.AnchorTabular = ae.AnchorTabular(
            predictor=lambda x: self.u_model.predict_proba(x),
            feature_names=self.features,
            seed=RANDOM_STATE,
        )
        y_pred = self.model.predict(self.adata)
        empty = self._make_empty(self.adata, self.y_true, y_pred)
        train_counts = self._count_df(train_data).values
        counts = self._count_df().values
        feature_vec = np.zeros_like(self.features)
        explainer.fit(train_counts, **fit_kwargs)
        class2importance = {}

        all_tracker = {
            self.label_col: [],
            "anchor": [],
            "precision": [],
            "n_features": [],
            "coverage": [],
        }
        for label in self.class2index.keys():
            indices = np.where(self.y_true == label)[0]
            tmp = []
            for i in indices:
                explanation: interfaces.Explanation = explainer.explain(
                    counts[i, :], **explain_kwargs
                )
                ivec = add_anchor_exp(explanation, feature_vec.copy())
                tmp.append([ivec])
                all_tracker[self.label_col].append(label)
                all_tracker["coverage"].append(explanation.coverage)
                all_tracker["anchor"].append(" AND ".join(explanation.anchor))
                all_tracker["n_features"].append(len(explanation.anchor))
                all_tracker["precision"].append(explanation.precision)
            if tmp:
                tmp_vals = np.zeros_like(counts)
                concatenated = np.concatenate(tmp, axis=1)
                tmp_vals[indices, :] = concatenated
                tmp_vals[~indices, :] = np.nan
                class2importance[label] = tmp_vals
        metric_df = pd.DataFrame(all_tracker)
        self._into_obsm(empty, "anchor_", class2importance)
        return empty, metric_df

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
        explain_fn : function receiving a model as input and returns a shap.Explainer
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
        if len(self.class2index) > 3:
            class2shap = {c: svals[:, :, i] for (c, i) in self.class2index.items()}
        else:
            class2shap = {"True": svals}
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
