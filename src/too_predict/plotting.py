#!/usr/bin/env ipython

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_diagonal_matrix(matrix: pd.DataFrame, ax: Axes, **kwargs) -> None:
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, mask=mask, square=True, ax=ax, **kwargs)
