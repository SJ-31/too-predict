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


def plot_adata(
    adata: ad.AnnData,
    y: str,
    subset: Iterable | None = None,
    style: str | None | list[str] = None,
    plot_together: bool = False,
    plot_mode: str = "pca",
    colors: list[str] | None = None,
    **kwargs,
) -> Figure:
    if "pca" not in adata.uns and plot_mode == "pca":
        sc.pp.pca(adata)
    elif "distances" not in adata.obsp and plot_mode == "umap":
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    show_axes: bool = True
    match plot_mode:
        case "pca":
            obsm_key = "X_pca"
            xlab, ylab = "PC1", "PC2"
            var_ratio = adata.uns["pca"]["variance_ratio"]
            pc1_var, pc2_var = round(var_ratio[0], 2), round(var_ratio[1], 2)
            subtitle = f"PC1, PC2 variance explained: {pc1_var}, {pc2_var}"
        case "umap":
            obsm_key = "X_umap"
            xlab, ylab = "UMAP1", "UMAP2"
            subtitle = "UMAP"
            show_axes = False
        case _:
            raise ValueError(f"Plotting mode {plot_mode} not supported!")

    keys = adata.obs[y] if subset is None else subset
    ncols = 1 if plot_together else len(keys)
    nrows = len(colors) if colors is not None else 1

    if style is not None and plot_together:
        ncols = len(style)
    elif style is None and plot_together:
        style = [None]
    elif plot_together and isinstance(style, str):
        style = [style]
    elif style is not None and not plot_together and not isinstance(style, str):
        raise ValueError("Multiple styles not supported when !`plot_together`")

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, sharex=True)

    multiple = ncols > 1
    pts: np.ndarray = adata.obsm[obsm_key]
    data = adata.obs
    if colors is None:
        colors = [y]
    if subset is not None and plot_together:
        mask = adata.obs[y].isin(subset)
        type_map = {y: str}
        if style is not None:
            for s in style:
                type_map[s] = str
        data = data.loc[mask, :].astype(type_map)
        pts = pts[mask, :]

    def get_ax(j, i):
        if len(colors) == 1 and not multiple:
            return axes
        elif len(colors) == 1 and multiple:
            return axes[i]
        elif len(colors) > 1 and multiple:
            return axes[j, i]
        elif len(colors) > 1 and not multiple:
            return axes[j]

    def set_labels(ax: Axes):
        if show_axes:
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
        else:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    if not plot_together:
        for j, color in enumerate(colors):
            for i, label in enumerate(keys):
                ax: Axes = get_ax(j, i)
                mask = adata.obs[y] == label
                pt1 = pts[mask, 0]
                pt2 = pts[mask, 1]
                sns.scatterplot(
                    data=data.loc[mask, :],
                    x=pt1,
                    y=pt2,
                    ax=ax,
                    hue=color,
                    style=style,
                    **kwargs,
                )
                set_labels(ax)
                ax.set_title(label)
                if i != len(keys) - 1:
                    ax.get_legend().remove()
    else:
        for j, color in enumerate(colors):
            for i, s in enumerate(style):
                ax: Axes = get_ax(j, i)
                pt1 = pts[:, 0]
                pt2 = pts[:, 1]
                sns.scatterplot(
                    data=data,
                    x=pt1,
                    y=pt2,
                    ax=ax,
                    hue=color,
                    style=s,
                    **kwargs,
                )
                set_labels(ax)
    fig.suptitle(subtitle)
    return fig
