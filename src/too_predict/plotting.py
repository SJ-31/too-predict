#!/usr/bin/env ipython
from collections.abc import Iterable

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scanpy as sc
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_diagonal_matrix(matrix: pd.DataFrame, ax: Axes, **kwargs) -> None:
    """matrix : a square m x m matrix"""
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, mask=mask, square=True, ax=ax, **kwargs)


def plot_instance_dist(dist: pd.DataFrame, ax: Axes, **kwargs) -> None:
    sns.heatmap(dist, ax=ax, **kwargs)
    ax.set(xlabel="Test instances", ylabel="Train instances")


def plot_pca(
    pca: np.ndarray, labels: pd.Series, ax: Axes, pcs=(0, 1), **kwargs
) -> None:
    x = f"PC{pcs[0] + 1}"
    y = f"PC{pcs[1] + 1}"
    tmp = pd.DataFrame({x: pca[:, pcs[0]], y: pca[:, pcs[1]], labels.name: labels})
    sns.scatterplot(tmp, x=x, y=y, hue=labels.name, ax=ax, **kwargs)


def plot_local_consistency(cons: dict, label):
    """Create a dumbbell plot comparing the consistency of feature importance
    between correctly classified (right) and incorrectly classified (wrong)
    samples

    Parameters
    ----------
    cons : dict output of Explain.shap_consistency()

    TODO: this isn't great
    """
    plots = []
    for status in ["right", "wrong"]:
        vals = cons[set][status][label].copy()
        scatter = go.Scatter(x=vals, y=vals.index, name=status, mode="markers")
        plots.append(scatter)

    # diff = go.Scatter(
    #     x=np.abs(cons[set]["wrong"].loc[:, current] - cons[set]["right"].loc[:, current]),
    #     y=cons[set]["wrong"][current].index,
    #     mode="lines",
    # )
    # plots.append(diff)
    go.Figure(plots)


def plot_pca_adata(
    adata: ad.AnnData,
    y: str,
    subset: Iterable | None = None,
    style: str | None | list[str] = None,
    plot_together: bool = False,
    **kwargs,
) -> Figure:
    if "pca" not in adata.uns:
        sc.pp.pca(adata)
    keys = adata.obs[y] if subset is None else subset
    ncols = 1 if plot_together else len(keys)

    if style is not None and plot_together:
        ncols = len(style)
    elif style is None and plot_together:
        style = [None]
    elif plot_together and isinstance(style, str):
        style = [style]
    elif style is not None and not plot_together and not isinstance(style, str):
        raise ValueError("Multiple styles not supported when !`plot_together`")

    fig, axes = plt.subplots(ncols=ncols, sharey=True, sharex=True)
    multiple = ncols > 1
    pcs: np.ndarray = adata.obsm["X_pca"]
    data = adata.obs
    if subset is not None and plot_together:
        mask = adata.obs[y].isin(subset)
        type_map = {y: str}
        if style is not None:
            for s in style:
                type_map[s] = str
        data = data.loc[mask, :].astype(type_map)
        pcs = pcs[mask, :]
    var_ratio = adata.uns["pca"]["variance_ratio"]
    pc1_var, pc2_var = round(var_ratio[0], 2), round(var_ratio[1], 2)
    if not plot_together:
        for i, label in enumerate(keys):
            ax: Axes = axes if not multiple else axes[i]
            mask = adata.obs[y] == label
            pc1 = pcs[mask, 0]
            pc2 = pcs[mask, 1]
            sns.scatterplot(
                data=data.loc[mask, :],
                x=pc1,
                y=pc2,
                ax=ax,
                hue="usage",
                style=style,
                **kwargs,
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(label)
            if i != len(keys) - 1:
                ax.get_legend().remove()
    else:
        for i, s in enumerate(style):
            ax: Axes = axes if not multiple else axes[i]
            pc1 = pcs[:, 0]
            pc2 = pcs[:, 1]
            sns.scatterplot(
                data=data,
                x=pc1,
                y=pc2,
                ax=ax,
                hue=y,
                style=s,
                **kwargs,
            )
    fig.suptitle(f"PC1, PC2 variance explained: {pc1_var}, {pc2_var}")
    return fig
