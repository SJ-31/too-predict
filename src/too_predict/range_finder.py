#!/usr/bin/env ipython
import itertools
from collections.abc import Sequence
from functools import reduce
from typing import Literal

import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import rustworkx as rx
import scipy.sparse as sparse
import seaborn as sns
from intervaltree import Interval, IntervalTree
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import too_predict.plotting as tp


class RangeFinder:
    """Class to find discriminatory gene expression ranges, while masking
    noisy ranges.

    A `noisy` range is one containing the expression of multiple tumor types,
        quantitatively determined by Gini impurity
    An `informative feature` for a given label is defined as one that has a range that
        distinctly separates it from all other labels
    """

    def __init__(
        self,
        label_col: str = "tumor_type",
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
    ) -> None:
        self.labels: Sequence
        self.adata: ad.AnnData
        self.label_col: str = label_col
        self.id_col: str = id_col
        self.batch_col: str = batch_col

        # Lookups
        self.cmap: dict[str, str] = {}
        self.id2range: dict = {}
        self.id2labels: dict[str, set] = {}
        self.id2contents: dict = {}
        self.label_tracker: dict = {}
        self.failed_ids: set[str] = set()
        self.label_totals: pd.Series

        # Range-finding parameters
        self.n_bins: int = n_bins
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
        self.mask_method: str = mask_method

    @staticmethod
    def gini_impurity(counts: pd.Series, size: int, report_n: int = 3) -> float:
        "Calculate the gini index for an expression range"
        counts = counts[counts != 0]
        if len(counts) == 0:
            return np.inf
        val = counts.apply(lambda x: x / size * (1 - x / size)).sum()
        return val

    def _check_n_features(self) -> bool:
        return all(np.array(list(self.label_tracker.values())) >= self.features_per)

    def fit(self, x: ad.AnnData) -> None:
        self.adata = x.copy()
        self.adata.X = (
            self.adata.X.toarray() if sparse.issparse(self.adata.X) else self.adata.X
        )
        self.labels = self.adata.obs[self.label_col]
        self.label_totals = self.labels.value_counts()
        self.cmap = tp.rand_cmap_d(self.labels)
        ids = self.adata.var[self.id_col].dropna()
        for i, id in enumerate(ids):
            if (
                self.premature_stop
                and self._check_n_features()
                or (self.max_features is not None and i == self.max_features - 1)
            ):
                break
            self.get_range(id)
        if not self._check_n_features():
            print(
                f"WARING: At least one label doesn't have {self.features_per} informative features"
            )
        print("Counts of informative features for each label")
        print(self.label_tracker)

    def transform(self, x: ad.AnnData) -> ad.AnnData:
        ids = self.adata.var[self.id_col].dropna()
        # to_keep =
        all_expr = np.zeros((x.shape[0], len(ids)))
        # for i in ids:

        # for i

    def get_range(
        self,
        id: str,
        backend: Literal["networkx", "rustworkx", "intervaltree"] = "rustworkx",
    ) -> None:
        if self.adata is None:
            raise ValueError("Not fitted yet!")
        expr = self._get_id_expr(id)
        if backend == "rustworkx":
            ranges, contents = self._get_ranges_rx(
                id,
                expr,
                self.labels,
                use_unique=self.use_unique,
                n_bins=self.n_bins,
                report_n=self.report_n,
                cutoff=self.impurity_cutoff,
            )
        elif backend == "networkx":
            ranges, contents = self._get_ranges_nx(
                id,
                expr,
                self.labels,
                use_unique=self.use_unique,
                n_bins=self.n_bins,
                report_n=self.report_n,
                cutoff=self.impurity_cutoff,
            )
        else:
            ranges, contents = self._get_ranges_it(
                id,
                expr,
                self.labels,
                use_unique=self.use_unique,
                n_bins=self.n_bins,
                report_n=self.report_n,
                cutoff=self.impurity_cutoff,
            )
        self.id2range[id] = ranges
        self.id2contents[id] = contents

    def _get_id_expr(self, id: str) -> np.ndarray:
        return self.adata.X[:, self.adata.var[self.id_col].values == id].flatten()

    def _check_label_p(self, label: str, label_count: int) -> bool:
        total: int = self.label_totals[label]
        if self.min_lwp is None:
            return True
        elif isinstance(self.min_lwp, dict):
            return (label_count / total) >= self.min_lwp[label]
        else:
            return (label_count / total) >= self.min_lwp

    # * Plotting

    def range_stripplot(self, id: str) -> Figure:
        fig, ax = plt.subplots()
        ranges = self.id2range.get(id)
        if ranges is None:
            raise ValueError(f"Ranges haven't been found for {id=} yet!")
        elif id in self.failed_ids:
            raise ValueError(f"No informative ranges were found for {id=}!")
        expr = self._get_id_expr(id)
        target_labels = self.id2labels.get(id)
        order = list(target_labels) + ["NOISE"]
        hue = [lab if lab in target_labels else "NOISE" for lab in self.labels]
        sns.stripplot(y=expr, x=hue, hue=hue, ax=ax, order=order)
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

    # * Range getter backends

    def _get_ranges_rx(
        self,
        id: str,
        vals: np.ndarray,
        labels: pd.Series,
        use_unique: bool = True,
        n_bins: int = 30,
        report_n: int = 3,
        cutoff=0.5,
    ) -> tuple:
        if use_unique:
            nodes = np.unique(vals)
        else:
            nodes = np.linspace(start=min(vals), stop=max(vals), num=n_bins)
        expr = pd.Series(vals, index=labels)
        nodes = sorted(nodes)
        i2n: dict = {}
        G: rx.PyGraph = rx.PyGraph()
        for pair in itertools.combinations(nodes, 2):
            begin = min(pair)
            end = pair[0] if begin == pair[1] else pair[1]
            narrowed = expr[(begin <= expr) & (expr <= end)]
            counts = narrowed.index.value_counts()
            gini = self.gini_impurity(
                counts=counts, size=len(narrowed), report_n=report_n
            )
            if gini < cutoff:
                if begin not in i2n:
                    i2n[begin] = G.add_node(begin)
                if end not in i2n:
                    i2n[end] = G.add_node(end)
                G.add_edge(i2n[begin], i2n[end], counts)
        if G.num_nodes() == 0:
            self.failed_ids.add(id)
            return [], {}
        ranges = []
        range2contents = {}
        seen: set = set()
        self.id2labels[id] = set()
        for cmp in rx.connected_components(G):
            sg = G.subgraph(list(cmp))
            s_nodes = sg.nodes()
            rge = (min(s_nodes), max(s_nodes))
            cur_counts = reduce(
                lambda x, y: x if all(x.values >= y.values) else y,
                (sg.get_edge_data_by_index(e) for e in sg.edge_indices()),
            ).sort_values(ascending=False)
            top_count, top_label = cur_counts[0], cur_counts.index[0]
            if self._check_label_p(top_label, top_count):
                ranges.append(rge)
                range2contents[rge] = cur_counts
                for lab in cur_counts.index[:report_n]:
                    if lab not in seen:
                        seen.add(lab)
                        self.id2labels[id].add(lab)
                        self.label_tracker[lab] = self.label_tracker.get(lab, 0) + 1
        return ranges, range2contents

    # TODO: haven't implemented id2labels for the others
    def _get_ranges_nx(
        self,
        id: str,
        vals: np.ndarray,
        labels: pd.Series,
        use_unique: bool = True,
        n_bins: int = 30,
        report_n: int = 3,
        cutoff=0.5,
    ) -> tuple:
        if use_unique:
            nodes = np.unique(vals)
        else:
            nodes = np.linspace(start=min(vals), stop=max(vals), num=n_bins)
        expr = pd.Series(vals, index=labels)
        nodes = sorted(nodes)
        G: nx.Graph = nx.Graph()
        for pair in itertools.combinations(nodes, 2):
            begin = min(pair)
            end = pair[0] if begin == pair[1] else pair[1]
            narrowed = expr[(begin <= expr) & (expr <= end)]
            counts = narrowed.index.value_counts()
            gini = self.gini_impurity(
                counts=counts, size=len(narrowed), report_n=report_n
            )
            if gini < cutoff:
                G.add_edge(begin, end, within=counts)
        ranges = []
        range2contents = {}
        for cmp in nx.connected_components(G):
            s = G.subgraph(cmp)
            rge = (min(s.nodes), max(s.nodes))
            cur_counts = reduce(
                lambda x, y: x if all(x.values >= y.values) else y,
                nx.get_edge_attributes(s, "within").values(),
            ).sort_values(ascending=False)
            top_count, top_label = cur_counts[0], cur_counts.index[0]
            if self._check_label_p(top_label, top_count):
                ranges.append(rge)
                range2contents[rge] = cur_counts
                for lab in cur_counts.index[:report_n]:
                    self.label_tracker[lab] = self.label_tracker.get(lab, 0) + 1
        return ranges, range2contents

    def _get_ranges_it(
        self,
        id: str,
        vals: np.ndarray,
        labels: pd.Series,
        use_unique: bool = True,
        n_bins: int = 30,
        report_n: int = 3,
        cutoff=0.5,
    ) -> tuple:
        if use_unique:
            nodes = np.unique(vals)
        else:
            nodes = np.linspace(start=min(vals), stop=max(vals), num=n_bins)
        expr = pd.Series(vals, index=labels)
        It: IntervalTree = IntervalTree()
        for pair in itertools.combinations(nodes, 2):
            begin = min(pair)
            end = pair[0] if begin == pair[1] else pair[1]
            narrowed = expr[(begin <= expr) & (expr <= end)]
            counts = narrowed.index.value_counts()
            gini = self.gini_impurity(
                counts=counts, size=len(narrowed), report_n=report_n
            )
            if gini < cutoff:
                It.add(Interval(begin, end, data=counts))
        ranges = []
        range2contents = {}
        It.merge_overlaps(
            data_reducer=lambda x, y: x if all(x.values >= y.values) else y
        )
        for it in It.items():
            rge = (it.begin, it.end)
            sorted = it.data.sort_values(ascending=False)
            top_count, top_label = sorted[0], sorted.index[0]
            if self._check_label_p(top_label, top_count):
                ranges.append(rge)
                range2contents[rge] = it.data
                for lab in it.data.sort_values().index[:report_n]:
                    self.label_tracker[lab] = self.label_tracker.get(lab, 0) + 1
        return ranges, range2contents
