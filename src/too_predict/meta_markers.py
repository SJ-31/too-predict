#!/usr/bin/env ipython

from typing import Literal, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.figure import Figure


def marker_auroc_score(
    adata: ad.AnnData,
    target: str,
    markers: Sequence,
    label_col: str,
    marker_col: str | None = None,
    style: Literal["ovr", "ovo"] = "ovr",
) -> pd.DataFrame:
    labels: pd.Series = adata.obs[label_col]
    counts: pd.Series = labels.value_counts()
    mmask = (
        adata.var[marker_col].isin(markers)
        if marker_col is not None
        else adata.var.index.isin(markers)
    )
    expr_all: np.ndarray = adata.X[:, mmask.values].toarray()
    pmask: np.ndarray = adata.obs[label_col] == target

    n_pos: int = pmask.sum()
    n_neg: int = counts.sum() - n_pos

    rhs: float = (n_pos * (n_pos + 1)) / 2
    result_tmp: dict = {"rank_sum": [], "tie_correction": []}
    for i, _ in enumerate(markers):
        expr: pd.Series = pd.Series(expr_all[:, i], index=labels)
        # Rank by expression in ascending order
        ranks = np.argwhere(expr.sort_values().index == target).flatten()
        result_tmp["rank_sum"].append(ranks.sum())

        unique_expr: pd.Series = expr.value_counts()
        tie_correct: float = (
            np.array([(v**3) - v for v in unique_expr])
            / ((n_pos + n_neg) * (n_pos + n_neg + 1))
        ).sum()
        result_tmp["tie_correction"].append(tie_correct)
    df = pd.DataFrame(result_tmp, index=markers)

    df.loc[:, "sigma"] = np.sqrt(
        (n_pos * n_neg) / 12 * (n_pos + 1 - df["tie_correction"])
    )
    frac: float = 1 / (n_pos * n_neg)
    df.loc[:, "AUROC"] = frac * (df["rank_sum"] - rhs)
    df.loc[:, "z"] = (df["AUROC"] - 0.5) / df["sigma"]
    df.loc[:, "pval"] = stats.norm.sf(np.abs(df["z"])) * 2
    df.loc[:, "padj"] = stats.false_discovery_control(df["pval"], method="bh")
    return df.loc[:, ["AUROC", "z", "pval", "padj"]]


class MetaMarkers:
    def __init__(
        self,
        datasets: list[ad.AnnData] | dict[str, ad.AnnData],
        known_markers: dict | None | pd.DataFrame = None,
    ) -> None:
        self.marker_df: pd.DataFrame
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

    def find_markers(self, method: Literal["edgeR", "scanpy"], **kwargs) -> None: ...

    def plot_auroc(
        self,
        targets: Sequence | None = None,
    ) -> None | Figure: ...
