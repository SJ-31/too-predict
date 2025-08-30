#!/usr/bin/env ipython

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from typing import Literal, TypeAlias

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scanpy as sc
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd
import seaborn as sns
import sklearn.feature_selection as fs
import sklearn.neighbors as sn
from matplotlib.figure import Figure
from scipy import sparse
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

import too_predict._rust_helpers as rh
import too_predict.explanation as te
import too_predict.plotting as plotting
import too_predict.r_utils as ru
import too_predict.utils as ut
from too_predict.model import PredBase

# * Filter class

FILTER_BEFORE = ["edgeR"]  # edgeR must always filter before

DEFINED_FILTERS: TypeAlias = Literal[
    "edgeR", "mutual_information", "variance_threshold", "Lasso"
]


class Filter:
    """Class for filtering features (genes) in adata objects to a requested subset
    Also re-orders the features in the adata to that of the feature list, filling
    in missing features with 0s.
    """

    def __init__(
        self,
        features: Sequence | None = None,
        method: Sequence[DEFINED_FILTERS] | None | DEFINED_FILTERS = None,
        feature_col="GENENAME",
        inplace=False,
        blacklist=None,
        top: int | Sequence[int] = 500,
        label_col="tumor_type",
        method_kwargs: dict[str, dict] | None = None,
        **kwargs,
    ) -> None:
        """Filter genes from adata object

        Parameters
        ----------
        top : int | Sequence[int]
            Number of top features to keep in feature selection. Can be passed as a sequence
            to specify the number of features wanted by each method
            Usually interpreted as the top features ranked according to the method
        features : Sequence of pre-selected features to use
            Recommended to specify a method and call `fit` to perform automatic
            feature selection with an adata object
        method : Feature selection method, or sequence of methods to apply. If a sequence,
            the methods will be applied in the specified order to select features
        method_kwargs : dictionary of method_name->kwargs
        """
        if not isinstance(features, Sequence) and features is not None:
            raise ValueError(
                """
                Features must be a sequence object!
                Did you mean to pass to `method` instead?"
                """
            )
        if features is None and method is None:
            raise ValueError("One of `features` or `method` must be given!")

        # Requested features to subset data by
        self._features: list | None = None
        self.blacklist: Sequence | None = blacklist
        self.features = features
        self.feature_col: str = feature_col
        self.discarded_features: Sequence | None = (
            None  # Any features discarded during preprocessing e.g.  # due to not being in enough samples
        )
        self.inplace: bool = inplace
        self.missing_features: list = []
        self.top: tuple[int] = (top,) if not isinstance(top, Sequence) else top
        if isinstance(method, str):
            self.methods = [method]
        elif isinstance(method, Sequence):
            self.methods: Sequence[DEFINED_FILTERS] | None = method
        else:
            self.methods = None
        if len(self.top) != len(self.methods):
            raise ValueError(
                "The sequence of n features to get does not match the number of methods given!"
            )
        self.method_kwargs: dict[str, dict] | None = method_kwargs

        self.label_col: str = label_col
        self.kwargs: dict = kwargs
        self.feat_metrics: pd.DataFrame | None = None  # Feature metrics computed during
        # `fit`, if available

    @property
    def features(self) -> list | None:
        return self._features

    @features.setter
    def features(self, value):
        if self.blacklist is not None:
            self._features = [
                f for f in value if list(set(value)) not in self.blacklist
            ]
        elif value is not None:
            self._features = list(set(value))
        else:
            self._features = None

    def copy(self) -> Filter:
        return Filter(
            features=self.features, feature_col=self.feature_col, inplace=self.inplace
        )

    def from_feature_importance(self, model: PredBase) -> None:
        underlying = model.get_model()
        if "feature_importances_" in dir(underlying):
            if len(imp := underlying.feature_importances_) != len(self.features):
                raise ValueError(
                    "The number of features in the fitted model does not match the number in this Filter instance!"
                )
            new_features = []
            self.discarded_features = (
                [] if self.discarded_features is None else self.discarded_features
            )
            for i, f in enumerate(self.features):
                if imp[i] == 0:
                    self.discarded_features.append(f)
                else:
                    new_features.append(f)
            self.features = new_features
        else:
            print("WARNING: the passed model has no feature importances")
            print("ignoring...")

    # TODO: this needs to be multi-step

    def fit(self, adata: ad.AnnData):
        if self.methods is not None:
            features = None
            print("Beginning feature selection...")
            print("Original n features:", adata.shape[1])
            for m, n in zip(self.methods, self.top):
                features = self._get_features(m, n, adata)
                adata = adata[:, adata.var[self.feature_col].isin(features)]
                print(f"n features after {m}:", adata.shape[1])
            self.features = features

    def _get_features(
        self, method: DEFINED_FILTERS, n: int, adata: ad.AnnData
    ) -> Sequence:
        adata = adata.copy()
        kwargs = (
            self.method_kwargs.get(method, {}) if self.method_kwargs is not None else {}
        )
        if method == "edgeR":
            features = self.edgeR(adata, n=n, **self.kwargs, **kwargs)
        elif method == "mutual_information":
            features = self.mutual_information(adata, n=n, **self.kwargs, **kwargs)
        elif method == "variance_threshold":
            features = self.variance_threshold(adata, n=n, **self.kwargs, **kwargs)
        elif method == "Lasso":
            features = self.lasso(adata, n=n, **self.kwargs, **kwargs)
        return tuple(filter(lambda x: x in adata.var[self.feature_col], features))

    def lasso(
        self,
        adata: ad.AnnData,
        n: int,
        remove_zeros: bool = False,
        n_per: int | None = None,
        cv=False,
        **kwargs,
    ) -> Sequence:
        kwargs = copy.deepcopy(kwargs)
        kwargs.update({"penalty": "l1", "solver": "saga"})
        lasso = LogisticRegressionCV(**kwargs) if cv else LogisticRegression(**kwargs)
        lasso.fit(ut.xarray_if_sparse(adata, copy=False), y=adata.obs[self.label_col])
        fmat = pd.DataFrame(
            np.absolute(lasso.coef_),
            index=lasso.classes_,
            columns=adata.var["GENEID"],
        ).transpose()
        features = []
        if remove_zeros:
            for i in fmat:
                cur = fmat.loc[:, i]
                cur = cur[cur != 0]
                features.extend(cur.index)
        elif n_per is not None:  # Take top n features for each class
            for i in fmat:
                sorted = fmat.loc[:, i].sort_values(ascending=False)
                features.extend(sorted.index[:n_per])
        else:
            sorted = fmat.mean(axis=1).sort_values(ascending=False)
            features.extend(sorted.index[:n])
        return features

    def transfer_features(self, other: Filter) -> None:
        """Try to update `other` Filter object to have the feature set of self
        self must be fitted

        Parameters
        ----------
        source : fitted `Filter` object, with defined method and features
        target : unfitted `Filter` object, with defined method
        """
        if self.features:
            if self.methods == other.methods:
                other.features = self.features
                other.method = None  # So nothing happens if you call `target.fit`
        else:
            raise ValueError("`self` object must be fitted before tranfering features!")

    def fit_transform(self, adata: ad.AnnData, _=None) -> ad.AnnData:
        self.fit(adata)
        return self.transform(adata)

    def transform(self, adata: ad.AnnData) -> ad.AnnData:
        adata = adata.copy()
        if self.features is None:
            raise ValueError(
                "No features! Either pass a list of features during init or call fit to find features with the chosen method"
            )
        new_shape = (adata.shape[0], len(self.features))
        new_var = pd.DataFrame(index=self.features).merge(
            adata.var, how="left", left_index=True, right_on=self.feature_col
        )
        new_var.index = self.features
        lookup: pd.Index = pd.Index(adata.var[self.feature_col])
        missing = []
        is_array = isinstance(adata.X, np.ndarray)
        counts: np.ndarray = adata.X.toarray() if not is_array else adata.X
        transformed = ad.AnnData(
            X=np.zeros(new_shape),
            var=new_var,
            obs=adata.obs,
            uns=adata.uns,
            obsm=adata.obsm,
        )
        converted_layers = {}
        for n in adata.layers:
            transformed.layers[n] = np.zeros(new_shape)
            cur = adata.layers[n]
            converted_layers[n] = cur if not sparse.issparse(cur) else cur.toarray()
        for i, f in enumerate(self.features):
            try:
                index = lookup.get_loc(f)
                transformed.X[:, i] = counts[:, index]
                for k, v in converted_layers.items():
                    transformed.layers[k][:, i] = v[:, index]
            except KeyError:
                missing.append(f)
                continue
        if not is_array:
            transformed.X = sparse.csr_array(transformed.X)
        self.missing_features = missing
        if len(missing) > 0:
            print(f"--- WARNING: {len(missing)} missing features!")
        if self.feat_metrics is not None:
            transformed.var = pd.merge(
                transformed.var, self.feat_metrics, how="left", on=self.feature_col
            )
        return transformed

    # ** Selection functions

    def variance_threshold(self, adata, n: int, threshold=0) -> list:
        vt = fs.VarianceThreshold(threshold=threshold)
        counts = ut.xarray_if_sparse(adata, copy=False)  # Copy during fit or transform
        vt.fit(counts)
        support = vt.get_support(indices=True)
        variance = np.nanvar(counts, axis=0)[support]
        kept = list(adata.var[self.feature_col][support])
        if len(kept) <= n:
            return kept
        df = pd.DataFrame({self.feature_col: kept, "variance": variance}).sort_values(
            "variance",
            ascending=False,
        )
        df = df.iloc[:n, :]
        self.feat_metrics = df
        return list(df[self.feature_col])

    def mutual_information(self, adata, n: int, **kwargs) -> list:
        counts = ut.xarray_if_sparse(adata, copy=False)
        y = adata.obs[self.label_col]
        minfo = fs.mutual_info_classif(counts, y, **kwargs)
        info_df = (
            pd.DataFrame(
                {self.feature_col: adata.var[self.feature_col], "mutual_info": minfo}
            )
            .sort_values("mutual_info", ascending=False)
            .iloc[:n, :]
        )
        self.feat_metrics = info_df
        return list(info_df[self.feature_col][:n])

    @ru.r_cleanup
    def edgeR(
        self,
        adata: ad.AnnData,
        n: int,
        n_per: int | None = None,
        fc_cutoff: float = 1.5,
        treat: bool = True,
        intercept: bool = False,
    ) -> list:
        ro.r("library(edgeR)")
        ro.r("library(tidyverse)")
        ru.np_to_r(np.transpose(ut.xarray_if_sparse(adata, copy=False)), "mat")
        ru.df_to_r(adata.obs, "obs")
        ru.df_to_r(adata.var, "var")
        ru.source("utils.R", in_r=True)
        ro.r("dge <- DGEList(mat, samples = obs, genes = var)")
        args = {
            "group": self.label_col,
            "fc_cutoff": fc_cutoff,
            "id_col": self.feature_col,
            "intercept": intercept,
            "treat": treat,
        }
        ro.globalenv["args"] = ro.ListVector(args)
        ro.r("args$dge <- dge")
        ro.r("result <- do.call(edgeR_ovr, args)")
        # print(ro.r("result$num_de"))
        # print(ro.r("class(as.data.frame(result$num_de))"))
        sig_results = ru.df_from_r(ro.r("as.data.frame(result$num_de)"))
        print(sig_results)
        kept = set()
        if n_per is None:
            n_per = n // len(adata.obs[self.label_col].unique())
        dfs = []
        for i in list(ro.r("names(result$top)")):
            current = ru.df_from_r(ro.r(f"result$top${i}"))
            current.loc[:, "absLFC"] = current["logFC"].abs()
            current.sort_values(by="absLFC", ascending=False, inplace=True)
            current = current.iloc[:n_per, :].assign(ovr_comparison=i)
            current[self.feature_col] = current[self.feature_col].astype(str)
            dfs.append(current)
            kept |= set(current[self.feature_col])
        if len(dfs) > 0:
            colnames = dfs[0].columns
            to_agg = {
                c: "first"
                for c in colnames
                if c not in {self.feature_col, "ovr_comparison"}
            }
            to_agg["ovr_comparison"] = lambda x: ";".join(list(x))
            grouped = pd.concat(dfs).groupby(self.feature_col).agg(to_agg).reset_index()
            self.feat_metrics = grouped
        return list(kept)


# * Others


def count_tomek_links(
    adata: ad.AnnData, target_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Count the number of tomek links ocurring in each class of adata.obs['target_col'],
    A tomek link is when two samples of different classes are nearest neighbors

    Returns
    -------
    tuple of [per-class counts of tomek links, pairwise-counts of tomek-links,
        matrix of pairwise counts between classes]

    """
    target = adata.obs[target_col]
    counts = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    neighbors = sn.NearestNeighbors(n_neighbors=2)  # Must get the second closest point
    neighbors.fit(counts)
    _distances, nearest = neighbors.kneighbors(counts)
    nearest_labelled = target.iloc[nearest[:, 1]]

    neighbor_pairs = pd.DataFrame(
        {
            "x": target.index.to_series().reset_index(drop=True),
            "x_label": target.reset_index(drop=True),
            "y": nearest_labelled.index.to_series().reset_index(drop=True),
            "y_label": nearest_labelled.reset_index(drop=True),
        },
    )
    tomek_links = neighbor_pairs.loc[
        neighbor_pairs["x_label"] != neighbor_pairs["y_label"], :
    ]
    suffixes = ["", "_label"]
    for s in suffixes:
        tomek_links.loc[:, f"pair{s}"] = tomek_links[f"x{s}"].combine(
            tomek_links[f"y{s}"], lambda x, y: {x, y}
        )
    tomek_links = tomek_links.loc[~tomek_links["pair"].duplicated(), :]
    # Do not overcount pairs e.g. if point A's nearest neighbor is B, and B's
    # nearest neighbor is A, then they would contribute twice to the class count

    class_counts = tomek_links["x_label"].value_counts()
    # Class counts with samples that are participating in at least one tomek link

    class_totals = target.value_counts()
    formatted_class = pd.DataFrame(
        {
            "class": class_counts.index.to_series(),
            "count": class_counts,
            "class_total": class_totals,
            "percentage": class_counts / class_totals * 100,
        }
    ).reset_index(drop=True)
    pair_counts = tomek_links["pair_label"].value_counts()
    formatted_pairs = pd.DataFrame(
        {
            "pair": pair_counts.index.to_series().apply(lambda x: list(x)),
            "count": pair_counts,
        }
    ).reset_index(drop=True)
    count_matrix = pd.DataFrame(0, index=target.unique(), columns=target.unique())

    for p, c in formatted_pairs.itertuples(index=False):
        x, y = p
        count_matrix.loc[x, y] = c
        count_matrix.loc[y, x] = c

    return formatted_class, formatted_pairs, count_matrix


def get_redundant_features(
    adata,
    height: int | float,
    method: str = "correlation",
    col: str = "cluster",
    n_cell_col: str = "n_cells",
) -> tuple[list, list, pd.DataFrame]:
    """Detect and remove potentially redundant features in `adata` by clustering on
        an association metric and selecting a single feature from each

    Parameters
    ----------
    method : association metric to use, includes `correlation`
        (recommended after a log-transformation), `phi_prop` and `rho_prop`.
        The phi and rho proportionality should be used on raw counts
    height : the cutoff at which to determine clusters

    Returns
    -------
    An tuple of [filtered features, features removed,
            updated adata.var df with cluster assignments]

    Notes
    -----
    The representative of a cluster is chosen as the feature found in the most samples
    of `adata`
    """
    if method == "correlation":
        matrix = spd.pdist(np.transpose(adata.X.toarray()), "correlation")  #
        # transpose because we want correlation between features, not the samples
        height = 1 - height  # because correlation distance is 1 - correlation
    elif method == "phi_prop":
        matrix = rh.phi_matrix(adata.X.toarray(), True)
    elif method == "rho_prop":
        matrix = rh.rho_matrix(adata.X.toarray(), True)
    else:
        raise ValueError(f"Method {method} not supported!")
    link_mat = sch.linkage(matrix, method="average")
    clusters = sch.fcluster(link_mat, t=height, criterion="distance")
    adata.var.loc[:, col] = clusters
    print(f"N clusters: {len(set(clusters))}")
    most_cells = (
        adata.var.loc[:, [col, n_cell_col]]
        .groupby(col)
        .idxmax()
        .loc[:, n_cell_col]
        .to_list()
    )
    removed = list(set(adata.var.index) - set(most_cells))
    return most_cells, removed, adata.var.copy()


class CompareSplits:
    """Compare the feature distribution of train vs test instances
        To help identify which features are responsible for misclassifications
    Parameters
    ----------
    y : column of adata.obs that we are trying to predict

    Notes
    -----
    For LFC methods, the idea is that instances that are difficult to classify will
    have high absolute lfc between train and test sets WITHIN a given label of `y`

    We can compare this lfc to the lfc of the `y` label against all other labels
    """

    def __init__(
        self, train: ad.AnnData, test: ad.AnnData, y: str = "tumor_type"
    ) -> None:
        self.adata = ad.concat([train, test], merge="first")
        self.adata.obs["usage"] = ["train"] * train.shape[0] + ["test"] * test.shape[0]
        self.lfcs: dict[str, pd.DataFrame] | None = None
        self.y = y  # Attribute of obs we want to predict
        self.train_y: Iterable = train.obs[y].unique()
        self.prototypes: dict[str, dict] = {}

    @ru.r_cleanup
    def edgeR_lfc(self) -> pd.DataFrame:
        i_name = self.adata.var.index.name
        ru.source("utils.R", in_r=True)
        ru.df_to_r(self.adata.obs, r_symbol="obs")
        ru.df_to_r(self.adata.var.reset_index(), r_symbol="var")
        counts = (
            self.adata.X.toarray() if sparse.issparse(self.adata.X) else self.adata.X
        )
        ru.np_to_r(np.transpose(counts), r_symbol="counts")
        ro.globalenv["label"] = self.y
        ro.r("result <- edgeR_lfc_train_test(counts, obs, var, label)")
        df = ru.df_from_r(ro.globalenv["result"])
        df.index = df[i_name]
        return df.drop(i_name, axis=1)

    def scanpy_lfc(self, threshold: float = 0.05) -> None:
        """Estimate log fold changes with scanpy's rank_genes_groups
        For each label, produces a df of three columns
        - test_vs_train : the lfc of the label in the train vs test set
        - vs_all : the lfc of the label against all other labels
        - abs_ratio : ratio of abs(vs_all) / abs(test_vs_train)
        """
        key = "usage"
        lfcs = {}
        train = self.adata[self.adata.obs["usage"] == "train", :]
        sc.tl.rank_genes_groups(train, groupby=self.y)
        ri = ut.RankInterpreter(train)
        all_lfc = ri.feature_stat("logfoldchanges")

        for label in self.adata.obs[self.y].unique():
            cur = self.adata[self.adata.obs[self.y] == label]
            batch_counts = cur.obs[key].value_counts()
            if len(batch_counts) == 1 or (batch_counts == 1).any():
                continue
            cur = self.adata[self.adata.obs[self.y] == label, :]
            sc.tl.rank_genes_groups(
                cur, groupby=key, method="wilcoxon", pts=True, reference="train"
            )
            ri = ut.RankInterpreter(cur)
            names = {"logfoldchanges": "test_vs_train", label: "vs_all"}
            lfc = (
                ri.feature_stat("logfoldchanges", threshold=threshold)
                .join(all_lfc.filter(items=[label], axis=1), on="names", how="inner")
                .rename(names, axis=1)
            )
            lfc.loc[:, "abs_ratio"] = np.abs(lfc["vs_all"]) - np.abs(
                lfc["test_vs_train"]
            )
            lfcs[label] = lfc
        self.lfcs = lfcs

    def get_prototypes(
        self,
        all_types_together: bool = False,
        **kwargs,
    ) -> None:
        self.prototypes["train_test_dist"] = {}

        def add_prototype_to_obs(sink: ad.AnnData, source: ad.AnnData, expl) -> None:
            indices = expl.prototype_indices
            index_vals = source.obs.index[indices]
            new_mask = sink.obs.index.isin(index_vals)
            if "is_prototype" not in sink.obs.columns:
                sink.obs["is_prototype"] = new_mask
            else:
                previous = sink.obs["is_prototype"]
                sink.obs["is_prototype"] = previous | new_mask

        def train_test_protos_dist(adata: ad.AnnData, label: str):
            tr_mask = adata.obs["usage"] == "train"
            lmask = adata.obs[self.y] == label
            is_proto = adata.obs["is_prototype"]
            # [2025-04-09 Wed] Prototypes don't appear in the train set
            train_p = adata[(tr_mask & lmask & is_proto).values, :]
            test_p = adata[((~tr_mask) & lmask & is_proto).values, :]
            if (train_p.shape[0] > 0) and (test_p.shape[0] > 0):
                dist = spd.cdist(train_p, test_p, metric="euclidean").mean()
                self.prototypes["train_test_dist"][label] = dist

        if all_types_together:
            protos = te.prototype_helper(self.adata, y=self.y, **kwargs)
            self.prototypes["all"] = protos
            add_prototype_to_obs(self.adata, self.adata, protos)
        else:
            self.prototypes["by_label"] = {}
            for label in self.train_y:
                current = self.adata[self.adata.obs[self.y] == label, :]
                cur_protos = te.prototype_helper(current, y=self.y, **kwargs)
                self.prototypes["by_label"][label] = cur_protos
                add_prototype_to_obs(self.adata, current, cur_protos)

        [train_test_protos_dist(self.adata, label) for label in self.train_y]

    def plot_prototypes(self, **kwargs) -> Figure:
        if not self.prototypes:
            raise ValueError("Prototypes haven't been calculated yet!")
        return self.plot_pca(style="is_prototype", **kwargs)

    def plot_pca(
        self,
        subset: Iterable | None = None,
        style: str | None | list[str] = None,
        plot_together: bool = False,
        **kwargs,
    ) -> Figure:
        return plotting.plot_adata(
            self.adata,
            self.y,
            subset=subset,
            style=style,
            plot_together=plot_together,
            **kwargs,
        )

    def get_plots(self, subset=None, **kwargs) -> Figure:
        if self.lfcs is None:
            raise ValueError("An lfc method has not been run yet!")
        keys = self.lfcs.keys() if subset is None else subset
        fig, axes = plt.subplots(ncols=len(keys), sharey=True, sharex=True)
        multiple = len(keys) > 1
        for i, k in enumerate(keys):
            df = self.lfcs[k].loc[:, ["test_vs_train", "vs_all"]]
            ax = axes if not multiple else axes[i]
            sns.scatterplot(x=df.iloc[:, 1], y=df.iloc[:, 0], ax=ax, **kwargs)
            ax.set_xlabel(f"{k} vs. all")
            ax.set_ylabel(None)
            if i == 0:
                ax.set_ylabel("lfc test vs. train")
        if multiple:
            fig.align_labels()
        return fig

    def edgeR_get_noisy(self, threshold: float = 0.05) -> list[str]:
        df = self.edgeR_lfc()
        return df.loc[df["PValue"] < threshold, :].index.to_list()

    def sc_get_noisy(
        self, quantile: float = 0.10, subset=None, agg_method: str = "any"
    ) -> list[str]:
        """Identify noisy features from lfc ratios

        Parameters
        ----------
        quantile : if a feature has a vs_all / test_vs_train lfc
            ratio less than this quantile (calculated within each label
            if agg_method is not mean or median), it is considered noisy

        agg_method : one of any|all|median|mean
            for median|mean, aggregate the ratio with the chosen method and filter with
            the summarized value.
        subset : only use these labels when determining noisy features

        Returns
        -------
        List of noisy features

        """
        if self.lfcs is None:
            raise ValueError("An lfc method has not been run yet!")
        combined = pd.concat(
            [
                v.assign(label=k).reset_index()
                for k, v in self.lfcs.items()
                if subset is None or k in subset
            ]
        ).reset_index(drop=True)
        if agg_method in {"mean", "median"}:
            if agg_method == "mean":
                agg = combined.groupby("names")["abs_ratio"].mean()
            else:
                agg = combined.groupby("names")["abs_ratio"].median()
            cutoff = np.nanquantile(agg, quantile)
            return agg.index[agg < cutoff].to_list()
        grouped = (
            combined.groupby("label")
            .apply(
                lambda x: x.assign(
                    passed=x["abs_ratio"] < np.nanquantile(x["abs_ratio"], quantile)
                ),
                include_groups=False,
            )
            .groupby("names")
        )
        if agg_method == "all":
            mask = grouped["passed"].all()
        else:
            mask = grouped["passed"].any()
        return mask.index[mask].to_list()
