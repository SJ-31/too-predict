#!/usr/bin/env ipython

from collections.abc import Sequence
from typing import Literal

import anndata as ad
import matplotlib.pyplot as plt
import numba as nb
import numba.types as nt
import numpy as np
import pandas as pd
import scipy.stats as stats
from adjustText import adjust_text
from matplotlib.figure import Figure
from pypalettes import load_cmap
from sklearn.preprocessing import LabelEncoder

import too_predict.plotting as tp

# Python implementation of MetaMarkers [1]
# References
#


# Pareto frontier is the set of all points that are not "dominated" by another
# point. Basically the maxima (or minima) of a point set. Here, we consider AUROC and lfc
# original authors do it by sorting on the two values, which seems dubious
def maximal_points(df: pd.DataFrame, x: str, y: str, ids: str) -> pd.DataFrame:
    pfront = []
    to_sort = df.loc[:, [x, y]].set_index(df[ids]).sort_values(x, ascending=False)
    maximum_y = -np.inf
    for point in to_sort.iterrows():
        if point[1][y] > maximum_y:
            maximum_y = point[1][y]
            pfront.append(point[0])
    return to_sort.loc[to_sort.index.isin(pfront), :]


def marker_auroc_score(
    adata: ad.AnnData,
    target: str,
    markers: Sequence,
    label_col: str,
    marker_col: str | None = None,
) -> pd.DataFrame:
    """Compute auROC scores for markers

    Parameters
    ----------
    target : the class instance which the classifier is attempting to predict with
        markers' expression values
    marker_col : column in adata.var corresponding to markers
    label_col : column in adata.obs containing the classes

    Returns
    -------
    Dataframe with auROC scores for each markers, pvalues and adjusted pvalues

    Notes
    -----

    """
    encoder: LabelEncoder = LabelEncoder()
    labels: np.ndarray = encoder.fit_transform(adata.obs[label_col])
    target: int = encoder.transform([target])[0]
    counts: pd.Series = pd.Series(labels).value_counts()
    mmask = (
        adata.var[marker_col].isin(markers)
        if marker_col is not None
        else adata.var.index.isin(markers)
    )
    expr_all: np.ndarray = adata.X[:, mmask.values].toarray()

    pmask: np.ndarray = labels == target

    n_pos: int = pmask.sum()
    n_neg: int = counts.sum() - n_pos

    rhs: float = (n_pos * (n_pos + 1)) / 2
    rs, tc = _get_ranks(
        np.array(range(len(markers))), expr_all, labels, target, n_pos, n_neg
    )
    df = pd.DataFrame({"rank_sum": rs, "tie_correction": tc}, index=markers)

    df.loc[:, "sigma"] = np.sqrt(
        (n_pos * n_neg) / 12 * (n_pos + 1 - df["tie_correction"])
    )
    frac: float = 1 / (n_pos * n_neg)
    df.loc[:, "AUROC"] = frac * (df["rank_sum"] - rhs)
    df.loc[:, "z"] = (df["AUROC"] - 0.5) / df["sigma"]
    df.loc[:, "pval"] = stats.norm.sf(np.abs(df["z"])) * 2

    nan_indices = df["pval"].isna()
    df.loc[:, "padj"] = np.full(df.shape[0], np.nan)
    df.loc[~nan_indices, "padj"] = stats.false_discovery_control(
        df["pval"][~nan_indices], method="bh"
    )
    return df.loc[:, ["AUROC", "z", "pval", "padj"]]


@nb.jit(
    nt.Tuple((nb.float64[:], nb.float64[:]))(
        nb.int64[:],
        nb.float64[:, :],
        nt.int64[:],
        nt.int64,
        nb.int64,
        nb.int64,
    ),
    nopython=True,
)
def _get_ranks(
    markers: np.ndarray,
    all_expression: np.ndarray,
    labels: np.ndarray,
    target: int,
    n_pos: int,
    n_neg: int,
):
    n_markers = len(markers)
    ranks_sum = np.zeros(n_markers)
    tie_correction = np.zeros(n_markers)
    for i in markers:
        expr: np.ndarray = all_expression[:, i]
        sorted_indices = np.argsort(expr)
        # Rank by expression in ascending order
        ranks = np.argwhere(labels[sorted_indices] == target).flatten()
        uniques = np.unique(expr)
        unique_counts = np.array([(expr == u).sum() for u in uniques])
        tie_correct = (
            np.array([v**3 - v for v in unique_counts])
            / ((n_pos + n_neg) * (n_pos + n_neg + 1))
        ).sum()
        ranks_sum[i] = ranks.sum()
        tie_correction[i] = tie_correct
    return ranks_sum, tie_correction


def plot_aurocs(
    dfs: dict[str, pd.DataFrame] | list[pd.DataFrame],
    palette: str | None = None,
    id_col: str = "GENEID",
    fc_col: str = "logFC",
    auroc_col: str = "AUROC",
) -> tuple[Figure, dict]:
    """Plot the Pareto frontier of markers from several datasets

    Parameters
    ----------
    dfs : dataframe of marker auROC scores from a single classification task e.g.
        from "marker_auroc_score"

    Returns
    -------
    Matplotlib figure

    Notes
    -----

    """
    if isinstance(dfs, list):
        dfs = dict(zip(range(len(dfs)), dfs))
    if palette is None:
        cmap: dict = tp.rand_cmap_d(dfs.keys())
    else:
        colors = load_cmap(palette).colors
        if len(colors) < len(dfs):
            raise ValueError("The provided palette has too few colors!")
        cmap = {k: colors[i] for i, k in enumerate(dfs.keys())}
    fig, ax = plt.subplots()
    paretos = {}
    ax.set_ylim(bottom=0, top=1)
    all_texts = []
    for dataset, df in dfs.items():
        color = cmap[dataset]
        pfront: pd.DataFrame = maximal_points(df, auroc_col, fc_col, id_col)
        paretos[dataset] = pfront.index
        lighter = tp.adjust_lightness(color, 0.7)
        darker = tp.adjust_lightness(color, 0.4)
        cvec = []
        for id in df[id_col]:
            if id in pfront.index:
                cvec.append(color)
            else:
                cvec.append(darker)
        for name, point in pfront.iterrows():
            xy = (point[fc_col], point[auroc_col])
            anno = ax.annotate(
                name,
                xy=xy,
                bbox={"edgecolor": color, "facecolor": "white"},
            )
            all_texts.append(anno)
        ax.plot(pfront[fc_col], pfront[auroc_col], c=lighter, ls="--")
        ax.scatter(x=df[fc_col], y=df[auroc_col], c=cvec, label=dataset)
    adjust_text(
        all_texts,
        min_arrow_len=2,
        only_move={"text": "xy", "static": "xy", "explode": "x", "pull": "xy"},
        arrowprops=dict(arrowstyle="-", color="k", lw=0.5),
    )
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(
        title="Dataset", loc="lower right", title_fontproperties={"weight": "bold"}
    )
    ax.set_ylabel(auroc_col)
    ax.set_xlabel(fc_col)
    return fig, paretos


class MetaMarkers:
    def __init__(
        self,
        datasets: list[ad.AnnData] | dict[str, ad.AnnData],
        label_col: str,
        subset: Sequence | None = None,  # Only calculate markers for these
        known_markers: dict | None | pd.DataFrame = None,
    ) -> None:
        self.marker_df: pd.DataFrame
        self.datasets: dict[str, ad.AnnData]
        if known_markers is not None and isinstance(known_markers, dict):
            self.marker_df = (
                pd.DataFrame(
                    {"target": known_markers.keys(), "gene_id": known_markers.values()}
                )
                .explode("gene_id")
                .set_index("gene_id")
            )
        elif known_markers is not None and isinstance(known_markers, pd.DataFrame):
            self.marker_df = known_markers
        else:
            self.marker_df = pd.DataFrame(
                columns="target", index=pd.Index(name="gene_id")
            )
        if isinstance(datasets, list):
            self.datasets = {f"dataset_{i}": datasets[i] for i in range(len(datasets))}
        else:
            self.datasets = datasets

    def find_markers(self, method: Literal["edgeR", "scanpy"], **kwargs) -> None: ...

    # def calc_auroc(self, target: str | None = None):
    #     if
    # for d in self.

    def plot_auroc(
        self,
        targets: Sequence | None = None,
    ) -> None | Figure: ...
