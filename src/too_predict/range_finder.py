#!/usr/bin/env ipython
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, override

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rustworkx as rx
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from sortedcontainers import SortedSet

import too_predict.plotting as tp
import too_predict.utils as ut
from too_predict.filter import Filter
from too_predict.model import PredBase
from too_predict.transformer import Transformer


@dataclass
class RangeData:
    rge: list[tuple]  # start, end of range
    gini: list[float]  # gini impurity of range
    contents: pd.DataFrame  # Counts of labels within the range
    labels: set  # Set of labels that the range is deemed informative to


class RangeFinder:
    """Class to find discriminatory gene expression ranges, while masking
    noisy ranges.

    A `noisy` range is one containing the expression of multiple tumor types,
        quantitatively determined by Gini impurity
    An `informative feature` for a given label is defined as one that has a range that
        distinctly separates it from all other labels

    Supports using multiple labels to separate samples (multitask)
        e.g. by tumor type and by sample type

    multitask_method : how to handle impurity in the multitask setting.
        If "mean", the average is impurity is taken while considering each label
        separately
        If "combine", new labels are created from the product of the labels and the
            impurity is calculated as normal
    """

    def __init__(
        self,
        label_col: str | Sequence = "tumor_type",
        batch_col: str = "is_organoid",
        id_col: str = "GENEID",
        features_per_label: int = 50,
        use_unique: bool = False,
        n_bins: int = 30,
        purity_cutoff: float = 0.5,
        premature_stop: bool = False,
        min_label_within_p: float | None | dict = None,
        report_n: int = 3,
        max_features: int | None = None,
        mask_method: Literal["binary", "mean", "median"] = "mean",
        multitask_method: Literal["mean", "combine", None] = "combine",
    ) -> None:
        self.labels: pd.Series | pd.DataFrame | pd.MultiIndex
        self.adata: ad.AnnData
        self.label_col: str | Sequence = label_col
        self.id_col: str = id_col
        self.batch_col: str = batch_col

        # Lookups
        self.imap: dict[str, RangeData]  # Dict of id-> range data:
        self.cmap: dict[str, str]

        self.label_metrics: pd.DataFrame
        self.id_metrics: pd.DataFrame
        self.label_tracker: dict
        self.failed_ids: set[str]
        self.label_totals: pd.Series

        # Range-finding parameters
        self.n_bins: int = n_bins
        self.multitask_method: Literal["combine", "mean", None] = multitask_method
        self.do_multitask: bool = (
            not isinstance(self.label_col, str) or multitask_method is None
        )
        self.use_unique: bool = use_unique
        self.features_per: int = features_per_label
        self.premature_stop: bool = premature_stop  # If true, stop the range-finding
        # process when every label has at least n = `features_per` features with
        # informative ranges
        self.min_lwp: float | None | dict = (
            min_label_within_p  # Minimum percent of labels that must be in a range to be considered informative
        )
        self.max_features: int | None = max_features
        self.report_n: int = 3
        self.impurity_cutoff: float = purity_cutoff  # Accept ranges with Gini impurity
        # below this value
        self.mask_method: str = mask_method  # How to transform data within the range

    @staticmethod
    def gini_impurity(counts: pd.Series, size: int) -> float:
        "Calculate the gini index for an expression range"
        counts = counts[counts != 0]
        if len(counts) == 0:
            return np.inf
        val = counts.apply(lambda x: x / size * (1 - x / size)).sum()
        return val

    def _check_n_features(self) -> bool:
        return all(np.array(list(self.label_tracker.values())) >= self.features_per)

    def fit(self, x: ad.AnnData) -> None:
        self.imap = {}
        self.label_tracker = {}
        self.failed_ids = set()
        self.adata = x.copy()
        self.adata.X = ut.xarray_if_sparse(x)
        if self.do_multitask:
            self.labels = pd.MultiIndex.from_frame(
                self.adata.obs.loc[:, self.label_col]
            )
        else:
            self.labels = self.adata.obs[self.label_col]
        self.label_totals = self.labels.value_counts()
        self.cmap = tp.rand_cmap_d(self.labels)
        ids = self.adata.var[self.id_col].dropna()
        for i, id in enumerate(ids):
            if (
                self.premature_stop
                and self._check_n_features()
                or (self.max_features is not None and i == self.max_features)
            ):
                break
            self.get_range(id)
        if not self._check_n_features():
            print(
                f"WARING: At least one label doesn't have {self.features_per} informative features"
            )
        print("Counts of informative features for each label")
        print(self.label_tracker)
        self._get_metrics()

    def id_label_counts(self, id: str) -> pd.DataFrame:
        self._has_data(id)
        return self.imap[id].contents

    def _get_metrics(self) -> None:
        labels = []
        tmp = {self.id_col: [], "avg_gini": [], "ginis": [], "n": [], "ranges": []}
        data: RangeData
        for id, data in self.imap.items():
            tmp[self.id_col].append(id)
            tmp["ginis"].append(data.gini)
            tmp["avg_gini"].append(np.mean(data.gini))
            tmp["n"].append(len(data.rge))
            tmp["ranges"].append(data.rge)
            labels.append(data.labels)
        if not self.do_multitask:
            groupby_explode: str = self.label_col
        else:
            groupby_explode = "combined"
        self.label_metrics = (
            pd.DataFrame(
                {
                    self.id_col: tmp[self.id_col],
                    groupby_explode: labels,
                }
            )
            .assign(count=1)
            .explode(groupby_explode)
            .groupby(groupby_explode)
            .agg({self.id_col: lambda x: list(x), "count": sum})
            .sort_values("count", ascending=False)
        )
        self.id_metrics = pd.DataFrame(tmp).sort_values("avg_gini")

    def transform(self, x: ad.AnnData) -> ad.AnnData:
        ids_to_use: set = set(self.imap.keys()) - self.failed_ids
        not_present: set = ids_to_use - set(x.var[self.id_col])
        filter: Filter = Filter(
            features=list(ids_to_use), feature_col=self.id_col, inplace=False
        )
        x = filter.fit_transform(x)
        old_expr: np.ndarray = ut.xarray_if_sparse(x).copy()
        new_expr = np.zeros_like(old_expr)
        for i, var in enumerate(x.var[self.id_col]):
            if var in not_present:
                continue
            ranges = self.imap[var].rge
            for rge in ranges:
                mask = (rge[0] <= old_expr[:, i]) & (old_expr[:, i] <= rge[1])
                if self.mask_method == "binary":
                    new_expr[:, i][mask] = 1
                elif self.mask_method == "mean":
                    new_expr[:, i][mask] = old_expr[:, i][mask].mean()
                elif self.mask_method == "median":
                    new_expr[:, i][mask] = old_expr[:, i][mask].median()
        new_adata: ad.AnnData = ad.AnnData(X=new_expr, var=x.var, obs=x.obs)
        return new_adata

    def fit_transform(self, x: ad.AnnData) -> ad.AnnData:
        self.fit(x)
        return self.transform(x)

    def get_range(self, id: str) -> None:
        if self.adata is None:
            raise ValueError("Not fitted yet!")
        expr = self._get_id_expr(id)
        ranges, contents, ginis, labels = self._get_ranges_rx(id, expr, self.labels)
        merged_contents = pd.concat(contents, axis=1)
        merged_contents.columns = contents.keys()
        self.imap[id] = RangeData(
            rge=ranges, gini=ginis, labels=labels, contents=merged_contents
        )

    def _get_id_expr(self, id: str, adata: ad.AnnData | None = None) -> np.ndarray:
        if adata is None:
            adata = self.adata
        return adata.X[:, adata.var[self.id_col].values == id].flatten()

    def _check_label_p(self, label: str, label_count: int) -> bool:
        total: int = self.label_totals.get(label, 0)
        if self.min_lwp is None:
            return True
        elif isinstance(self.min_lwp, dict):
            return (label_count / total) >= self.min_lwp[label]
        else:
            return (label_count / total) >= self.min_lwp

    def _has_data(self, id: str) -> None:
        data = self.imap.get(id)
        if data is None:
            raise ValueError(f"Ranges haven't been found for {id=} yet!")
        elif id in self.failed_ids:
            raise ValueError(f"No informative ranges were found for {id=}!")

    # ** Plotting

    def range_stripplot(
        self, id: str, adata: ad.AnnData | None = None, hue: str | None = None
    ) -> Figure:
        fig, ax = plt.subplots()
        self._has_data(id)
        data: RangeData = self.imap[id]
        ranges = data.rge
        expr = self._get_id_expr(id, adata)
        expr[expr == 0] = np.nan
        target_labels = self.imap[id].labels
        order = list(target_labels) + ["NOISE"]
        xs = [lab if lab in target_labels else "NOISE" for lab in self.labels]
        if adata is not None and hue is not None:
            hue_lst = adata.obs[hue]
        elif adata is None and hue is not None:
            hue_lst = self.adata.obs[hue]
        else:
            hue_lst = xs
        sns.stripplot(y=expr, x=xs, hue=hue_lst, ax=ax, order=order)
        xlim = ax.get_xlim()
        for rge in ranges:
            ax.add_patch(
                Rectangle(
                    (xlim[0], rge[0]),
                    width=xlim[1] - xlim[0],
                    height=rge[1] - rge[0],
                    alpha=0.1,
                    facecolor="green",
                )
            )
        return fig

    # ** Range getter backends

    def _ranges_from_sg_rx(
        self,
        sg: rx.PyGraph,
        seen: set,
        ranges: list,
        range2contents: dict,
        ginis: list,
    ):
        """
        Process a set of connected ranges in the subgraph, extracting the largest range
        The purity of the range set is taken to be the purity of the largest range,
        as are the label counts
        """
        cur_counts: pd.Series = None
        cur_gini: float = -np.inf
        s_nodes = SortedSet()
        for e_begin, e_end, data in sg.edge_index_map().values():
            s_nodes.update((sg[e_begin], sg[e_end]))
            counts = data["counts"]
            gini = data["gini"]
            if cur_counts is None or (counts.values >= cur_counts.values).all():
                cur_counts = counts
                cur_gini = gini
        rge = (s_nodes[0], s_nodes[-1])
        cur_counts = cur_counts.sort_values(ascending=False)
        top_count, top_label = cur_counts[0], cur_counts.index[0]

        if self._check_label_p(top_label, top_count):
            # Check that the current range meets the minimum parameters
            # for the proportion of labels within, if provided
            ranges.append(rge)
            range2contents[rge] = cur_counts
            ginis.append(cur_gini)
            for lab in cur_counts.index[: self.report_n]:
                if lab not in seen:
                    seen.add(lab)
                    self.label_tracker[lab] = self.label_tracker.get(lab, 0) + 1

    def _get_ranges_rx(
        self,
        id: str,
        vals: np.ndarray,
        labels: pd.Series | pd.DataFrame | pd.MultiIndex,
    ) -> tuple:
        if self.use_unique:
            nodes = np.unique(vals)
        else:
            nodes = np.linspace(start=min(vals), stop=max(vals), num=self.n_bins)
        expr = pd.Series(vals, index=labels)
        nodes = sorted(nodes)
        i2n: dict = {}
        G: rx.PyGraph = rx.PyGraph()
        for pair in itertools.combinations(nodes, 2):
            begin = min(pair)
            end = pair[0] if begin == pair[1] else pair[1]
            narrowed = expr[(begin <= expr) & (expr <= end)]
            counts = narrowed.index.value_counts()
            if self.multitask_method != "mean" and self.do_multitask:
                gini = self.gini_impurity(counts=counts, size=len(narrowed))
            else:
                gini_tracker = []
                df = narrowed.to_frame().reset_index()
                for lab in self.label_col:
                    cur_counts = df[lab].value_counts()
                    gini_tracker.append(
                        self.gini_impurity(counts=cur_counts, size=narrowed.shape[0])
                    )
                gini = np.mean(gini_tracker)
            if gini < self.impurity_cutoff:
                if begin not in i2n:
                    i2n[begin] = G.add_node(begin)
                if end not in i2n:
                    i2n[end] = G.add_node(end)
                G.add_edge(i2n[begin], i2n[end], {"counts": counts, "gini": gini})
        if G.num_nodes() == 0:
            self.failed_ids.add(id)
            return [], {}
        ranges = []
        range2contents = {}
        ginis = []
        seen: set = set()
        for cmp in rx.connected_components(G):
            sg = G.subgraph(list(cmp))
            self._ranges_from_sg_rx(sg, seen, ranges, range2contents, ginis)
        return ranges, range2contents, ginis, seen


# * Wrapper for predictor


class RangeFinderPred(PredBase):
    def __init__(
        self,
        model: PredBase,
        transformer: Transformer,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.transformer: Transformer = transformer
        self.genewise_params: np.ndarray
        self.kwargs: dict = kwargs
        self.rf: RangeFinder = RangeFinder(**kwargs)

    @override
    def fit(self, X: ad.AnnData, y="tumor_type") -> None:
        learned: ad.AnnData = self.rf.fit_transform(
            X
        )  # Learn ranges and apply to training data
        transformed = self.transformer.fit_transform(learned)
        self.model.fit(transformed, y)

    @override
    def predict(self, X: ad.AnnData) -> np.ndarray:
        x = self.rf.transform(X)
        x = self.transformer.fit_transform(x)
        return self.model.predict(x)

    @override
    def predict_proba(self, X: ad.AnnData) -> np.ndarray:
        x = self.rf.transform(X)
        x = self.transformer.fit_transform(x)
        return self.model.predict_proba(x)
