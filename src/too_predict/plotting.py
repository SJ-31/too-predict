#!/usr/bin/env ipython
import colorsys
from collections.abc import Iterable
from functools import reduce
from typing import Literal, Sequence, overload

import anndata as ad
import marsilea as ma
import marsilea.plotter as mp
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotnine as gg
import scanpy as sc
import scipy.sparse as sparse
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pypalettes import load_cmap


def adjust_lightness(color, amount=0.5):
    # Credit: @Ian Hincks & @FLekschas
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


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
    colors: list[str] | str,
    plot_mode: Literal["pca", "umap"] = "pca",
) -> gg.ggplot:
    if "pca" not in adata.uns and plot_mode == "pca":
        sc.pp.pca(adata)
    elif "distances" not in adata.obsp and plot_mode == "umap":
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    colors = colors if isinstance(colors, list) else [colors]

    if plot_mode == "pca":
        obsm_key = "X_pca"
        xlab, ylab = "PC1", "PC2"
        var_ratio = adata.uns["pca"]["variance_ratio"]
        pc1_var, pc2_var = round(var_ratio[0], 2), round(var_ratio[1], 2)
        subtitle = f"PC1, PC2 variance explained: {pc1_var}, {pc2_var}"
    elif plot_mode == "umap":
        obsm_key = "X_umap"
        xlab, ylab = "UMAP1", "UMAP2"
        subtitle = "UMAP"
    df: pd.DataFrame = pd.concat(
        [
            pd.DataFrame(
                adata.obsm[obsm_key][:, [0, 1]], columns=[xlab, ylab]
            ).reset_index(),
            adata.obs.loc[:, colors].reset_index(),
        ],
        axis="columns",
    )
    plots = []
    for i, color_key in enumerate(colors):
        plot = (
            gg.ggplot(df, gg.aes(x=xlab, y=ylab, color=color_key))
            + gg.geom_point()
            + gg.theme(figure_size=(15, 10))
        )
        if plot_mode == "umap":
            plot = plot + gg.theme(
                axis_text_y=gg.element_blank(),
                axis_text_x=gg.element_blank(),
                axis_title_x=gg.element_blank(),
                axis_title_y=gg.element_blank(),
            )
        if i == 0:
            plot = plot + gg.ggtitle(subtitle=subtitle)
        plots.append(plot)
    if len(plots) == 1:
        return plots[0]
    return reduce(lambda x, y: x / y, plots)


def scanpy_plot_gs(
    adata: ad.AnnData,
    genes: Sequence,
    grouping_var: str | None = None,
    gene_sets: dict[str, set] | None = None,
    gene_set_limit: int = 5,
    method: Literal["heatmap", "tracksplot"] = "tracksplot",
    gene_symbols: str | None = None,
    gene_symbols_to_show: str | None = None,
    **kwargs,
):
    """Convenience function for creating mappings for scanpy plots

    Parameters
    ----------
    genes : list of genes to plot
    gene_sets : dict defining gene set groupings to plot. Genes in this dict must
        use the same identifiers as in `genes`
    grouping_var : a column in adata.var to be used to group genes
    gene_symbols : column in adata.var corresponding to gene ids. Defaults to adata.var.index
    gene_symbols_to_show : column in adata.var to show in the plot instead of
        gene_symbols. The mapping is performed internally
    -------
    """
    var: pd.DataFrame = adata.var
    mapping_to_plot: dict = {}
    id_mapping = {}

    if gene_symbols_to_show is not None:
        for gene in genes:
            try:
                if gene_symbols is None:
                    id_mapping[gene] = var[gene_symbols_to_show][var.index == gene][0]
                else:
                    id_mapping[gene] = var[gene_symbols_to_show][
                        var[gene_symbols] == gene
                    ][0]
            except IndexError:
                id_mapping[gene] = None

    def map_if(genes, filter: set | None):
        # Only allow genes present in `filter` if it is provided
        if id_mapping and filter is not None:
            return [id_mapping[g] for g in genes if g in filter]
        elif id_mapping:
            return [id_mapping[g] for g in genes]
        elif filter is not None:
            return [g for g in genes if g in filter]
        return genes

    as_set: set = set(genes)
    if gene_sets is not None:
        for k, v in gene_sets.items():
            if len(v & as_set) == 0:
                continue
            mapping_to_plot[k] = map_if(v, as_set)
            if len(mapping_to_plot) >= gene_set_limit:
                break
    elif grouping_var is not None:
        for group in adata.var[grouping_var].unique():
            subset = adata.var.loc[adata.var[grouping_var] == group, :]
            if gene_symbols is None:
                subset_g = subset.index[subset.index.isin(as_set)]
            else:
                subset_g = subset[gene_symbols][subset[gene_symbols].isin(as_set)]
            mapping_to_plot[group] = map_if(subset_g, None)

    if method == "heatmap":
        sc.pl.heatmap(
            adata, mapping_to_plot, gene_symbols=gene_symbols_to_show, **kwargs
        )
    elif method == "tracksplot":
        sc.pl.tracksplot(
            adata, mapping_to_plot, gene_symbols=gene_symbols_to_show, **kwargs
        )


@overload
def rand_cmap_d(val: int | None, assign: bool = True) -> list[str]: ...


@overload
def rand_cmap_d(val: Sequence, assign: bool = False) -> dict: ...


def rand_cmap_d(
    val: Sequence | None | int = None, assign: bool = False
) -> list[str] | dict:
    def get_n(length: int) -> list[str]:
        tmp = set()
        while len(tmp) < length:
            tmp |= set(load_cmap().colors)
        return list(tmp)[:length]

    if val is None:
        return load_cmap().colors
    if isinstance(val, int):
        return get_n(val)
    else:
        uniques = set(val)
        colors = get_n(len(uniques))
        mapping = dict(zip(uniques, colors))
        if assign:
            return [mapping[v] for v in val]
        return mapping


def mp_plot(
    adata: ad.AnnData,
    genes: Sequence,
    var_groupings: str | Sequence = None,
    method: Literal["heatmap", "tracksplot"] = "tracksplot",
    gene_symbols: str | None = None,
    gene_symbols_to_show: str | None = None,
    sample_groupings: str | None | Sequence = None,
    cmaps: dict | None = None,
    var_spacing: float = 0.001,
    obs_spacing: float = 0.001,
    **kwargs,
):
    expr: np.ndaarray = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    cmaps = {} if cmaps is None else cmaps
    var: pd.DataFrame = adata.var
    vmask: np.ndarray = (
        var.index.isin(genes) if gene_symbols is None else var[gene_symbols].isin(genes)
    )
    expr = expr[:, vmask]
    if gene_symbols_to_show is not None:
        labels = mp.Labels(var[gene_symbols_to_show][vmask])
    else:
        labels = mp.Labels(genes)
    if method == "heatmap":
        m = ma.Heatmap(expr, **kwargs)
        m.add_top(labels, pad=0.1)
    else:
        raise ValueError("Not implemented yet!")

    # Annotate genes
    if isinstance(var_groupings, str):
        var_groupings = [var_groupings]
    if var_groupings is not None:
        vals_first = list(var[var_groupings[0]][vmask])
        v_order = set(vals_first)
        v_cmap: dict = cmaps.get("grouping_var", rand_cmap_d(v_order))
        var_chunk = mp.Chunk(
            list(v_order),
            fill_colors=[v_cmap[g] for g in v_order],
            rotation=90,
            align="center",
        )
        m.add_top(var_chunk, pad=0.1)
        m.group_cols(vals_first, order=v_order, spacing=var_spacing)
    if var_groupings is not None and len(var_groupings) > 1:
        for vals in var_groupings[1:]:
            values = list(adata.var[vals])
            cmap = cmaps.get(vals, rand_cmap_d(values))
            colors = mp.Colors(values, palette=cmap, label=vals, label_loc=None)
            m.add_top(colors, size=0.2, pad=0.1)

    # Annotate samples
    if sample_groupings is not None and isinstance(sample_groupings, str):
        sample_groupings = [sample_groupings]
    if sample_groupings is not None:
        svals_first = list(adata.obs[sample_groupings[0]])
        order = set(svals_first)
        chnk = mp.Chunk(list(order), rotation=0, align="right")
        m.add_left(chnk, pad=0.1)
        m.group_rows(svals_first, order=order, spacing=obs_spacing)
    if sample_groupings is not None and len(sample_groupings) > 1:
        for svals in sample_groupings[1:]:
            values = list(adata.obs[svals])
            cmap = cmaps.get(svals, rand_cmap_d(values))
            colors = mp.Colors(values, palette=cmap, label=svals, label_loc=None)
            m.add_left(colors, size=0.2, pad=0.1)

    m.add_legends()
    m.render()
    return m
