#!/usr/bin/env ipython
import itertools
from collections.abc import Sequence
from functools import reduce
from typing import Literal

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import rustworkx as rx
from intervaltree import Interval, IntervalTree
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


class RangeFinder:
    def __init__(
        self,
        label_col: str = "tumor_type",
        batch_col: str = "is_organoid",
        id_col: str = "GENEID",
        min_features_per_label: int = 50,
        use_unique: bool = False,
        n_bins: int = 30,
        purity_cutoff: float = 0.5,
        min_labels_within: int | None = None,
        report_n: int = 3,
        mask_method: Literal["binary", "mean", "median"] = "mean",
    ) -> None:
        self.label_col: str = label_col
        self.id_col: str = id_col
        self.n_bins: int = n_bins
        self.use_unique: bool = use_unique
        self.batch_col: str = batch_col
        self.min_fpl: int = min_features_per_label
        self.min_lw: int | None = min_labels_within
        self.labels: Sequence = None
        self.adata: ad.AnnData | None = None
        self.report_n: int = 3
        self.purity_cutoff: float = purity_cutoff
        self.mask_method: str = mask_method
        self.cmap: dict[str, str] = {}
        self.id2range: dict = {}
        self.id2labels: dict[str, set] = {}
        self.id2contents: dict = {}
        self.label_tracker: dict = {}

    @staticmethod
    def gini_index(counts: pd.Series, size: int, report_n: int = 3) -> float:
        "Calculate the gini index for an expression range"
        counts = counts[counts != 0]
        if len(counts) == 0:
            return np.inf
        val = counts.apply(lambda x: x / size * (1 - x / size)).sum()
        return val

    def fit(self, x: ad.AnnData) -> None:
        self.adata = x.copy()
        self.adata.X = (
            self.adata.X.toarray() if sparse.issparse(self.adata.X) else self.adata.X
        )
        self.labels = self.adata.obs[self.label_col]
        self.cmap: dict = tp.rand_cmap_d(self.labels)

    def transform() -> ad.AnnData: ...

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
                expr,
                self.labels,
                use_unique=self.use_unique,
                n_bins=self.n_bins,
                report_n=self.report_n,
                cutoff=self.purity_cutoff,
            )
        elif backend == "networkx":
            ranges, contents = self._get_ranges_nx(
                expr,
                self.labels,
                use_unique=self.use_unique,
                n_bins=self.n_bins,
                report_n=self.report_n,
                cutoff=self.purity_cutoff,
            )
        else:
            ranges, contents = self._get_ranges_it(
                expr,
                self.labels,
                use_unique=self.use_unique,
                n_bins=self.n_bins,
                report_n=self.report_n,
                cutoff=self.purity_cutoff,
            )
        self.id2range[id] = ranges
        self.id2contents[id] = contents

    def _get_id_expr(self, id: str) -> np.ndarray:
        return self.adata.X[:, self.adata.var[self.id_col].values == id].flatten()

    def range_stripplot(self, id: str) -> Figure:
        fig, ax = plt.subplots()
        ranges = self.id2range.get(id)
        if ranges is None:
            raise ValueError(f"Ranges haven't been found for {id=} yet!")
        expr = self._get_id_expr(id)
        target_labels = self.id2labels.get(id)
        order = list(target_labels) + ["NOISE"]
        hue = [lab if lab in target_labels else "NOISE" for lab in labels]
        sns.stripplot(y=expr, x=hue, hue=hue, ax=ax, order=order)
        xlim = ax.get_xlim()
        for rge in ranges:
            ax.axhline(y=rge[0], xmin=xlim[0], xmax=xlim[1])
            ax.axhline(y=rge[1], xmin=xlim[0], xmax=xlim[1])
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

    def _get_ranges_rx(
        self,
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
            gini = self.gini_index(counts=counts, size=len(narrowed), report_n=report_n)
            if gini < cutoff:
                if begin not in i2n:
                    i2n[begin] = G.add_node(begin)
                if end not in i2n:
                    i2n[end] = G.add_node(end)
                G.add_edge(i2n[begin], i2n[end], counts)
        ranges = []
        range2contents = {}
        for cmp in rx.connected_components(G):
            sg = G.subgraph(list(cmp))
            s_nodes = sg.nodes()
            rge = (min(s_nodes), max(s_nodes))
            cur_counts = reduce(
                lambda x, y: x if all(x.values >= y.values) else y,
                (sg.get_edge_data_by_index(e) for e in sg.edge_indices()),
            ).sort_values(ascending=False)
            if self.min_lw is None or cur_counts[0] > self.min_lw:
                ranges.append(rge)
                range2contents[rge] = cur_counts
                self.id2labels[id] = set()
                for lab in cur_counts.index[:report_n]:
                    self.id2labels[id].add(lab)
                    self.label_tracker[lab] = self.label_tracker.get(lab, 0) + 1
        return ranges, range2contents

    def _get_ranges_nx(
        self,
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
            gini = self.gini_index(counts=counts, size=len(narrowed), report_n=report_n)
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
            )
            if self.min_lw is None or cur_counts[0] > self.min_lw:
                ranges.append(rge)
                range2contents[rge] = cur_counts
                for lab in cur_counts.index[:report_n]:
                    self.label_tracker[lab] = self.label_tracker.get(lab, 0) + 1
        return ranges, range2contents

    def _get_ranges_it(
        self,
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
            gini = self.gini_index(counts=counts, size=len(narrowed), report_n=report_n)
            if gini < cutoff:
                It.add(Interval(begin, end, data=counts))
        ranges = []
        range2contents = {}
        It.merge_overlaps(
            data_reducer=lambda x, y: x if all(x.values >= y.values) else y
        )
        for it in It.items():
            rge = (it.begin, it.end)
            if self.min_lw is None or it.data.max() > self.min_lw:
                ranges.append(rge)
                range2contents[rge] = it.data
                for lab in it.data.sort_values().index[:report_n]:
                    self.label_tracker[lab] = self.label_tracker.get(lab, 0) + 1
        return ranges, range2contents
