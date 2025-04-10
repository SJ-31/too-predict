#!/usr/bin/env ipython

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.axes import Axes


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
