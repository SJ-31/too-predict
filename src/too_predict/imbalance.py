#!/usr/bin/env ipython
import math
from typing import Literal

import anndata as ad
import imblearn.combine as icc
import imblearn.over_sampling as ios
import imblearn.under_sampling as ius
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scipy.stats as stats
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.under_sampling.base import BaseUnderSampler
from imblearn.utils import check_sampling_strategy
from numpy.random import Generator
from scanpy import AnnData

import too_predict.r_utils as ru
import too_predict.utils as ut

# Utilities for handling imbalanced data
OTHERS: set = {"nb_edgeR"}
IMBLEARN_METHODS: set = {
    "SMOTE",
    "KMeansSMOTE",
    "TomekLinks",
    "ADASYN",
    "SVMSMOTE",
    "BorderLineSMOTE",
    "InstanceHardnessThreshold",
    "RandomOverSampler",
    "RandomUnderSampler",
    "NearMiss",
    "TomekLinks",
    "SMOTEENN",
    "SMOTETomek",
    "EditedNearestNeighbours",
}
IMPLEMENTED_BALANCE: set = IMBLEARN_METHODS | OTHERS


class Balancer:
    def __init__(self, method: str, **kwargs) -> None:
        if method not in IMPLEMENTED_BALANCE:
            raise ValueError(f"Method {method} not implemented!")
        self.method: str = method
        self.label_col: str | None = None
        self.is_imblearn: bool = False
        if self.method in IMBLEARN_METHODS:
            self.is_imblearn = True
            self.model: BaseOverSampler | BaseUnderSampler | None = (
                self._imblearn_model(method, **kwargs)
            )
        else:
            self.model = None
        self.kwargs: dict = kwargs

    def _imblearn_model(self, model, **kwargs):
        if model == "SMOTE":
            return ios.SMOTE(**kwargs)
        if model == "KMeansSMOTE":
            return ios.KMeansSMOTE(**kwargs)
        if model == "NearMiss":
            return ius.NearMiss(**kwargs)
        if model == "SVMSMOTE":
            return ios.SVMSMOTE(**kwargs)
        if model == "ADASYN":
            return ios.ADASYN(**kwargs)
        if model == "RandomOverSampler":
            return ios.RandomOverSampler(**kwargs)
        if model == "SMOTEENN":
            return icc.SMOTEENN(**kwargs)
        if model == "SMOTETomek":
            return icc.SMOTETomek(**kwargs)
        if model == "EditedNearestNeighbours":
            return ius.EditedNearestNeighbours(**kwargs)
        if model == "TomekLinks":
            return ius.TomekLinks(**kwargs)
        if model == "BorderLineSMOTE":
            return ios.BorderlineSMOTE(**kwargs)
        if model == "InstanceHardnessThreshold":
            return ius.InstanceHardnessThreshold(**kwargs)
        if model == "RandomUnderSampler":
            return ius.RandomUnderSampler(**kwargs)

    def check_sampling_strategy(
        self,
        y,
        type: Literal["over-sampling", "under-sampling", "clean-sampling"],
        strategy: Literal["minority", "not minority", "not majority", "all"]
        | None = None,
        n: int | None = None,
    ) -> dict:
        if strategy is None and n is None:
            raise ValueError("Either `n` or `strategy` must be provided!")
        if strategy:
            counts = check_sampling_strategy(
                sampling_strategy=strategy, y=y, sampling_type=type
            )
            return counts
        return {u: n for u in y.unique()}

    def nb_edgeR_wrapper(self, adata: ad.AnnData, **kwargs) -> ad.AnnData:
        if n := kwargs.pop("n", None):
            return nb_edgeR(adata=adata, n=n, **kwargs)
        counts = self.check_sampling_strategy(
            y=adata.obs[self.label_col],
            strategy=kwargs.pop("sampling_strategy", "auto"),
            type="over-sampling",
        )
        n = np.sum(list(counts.values()))
        prop = {k: (v / n).item() for k, v in counts.items()}
        return nb_edgeR(adata=adata, n=n.item(), group_prop=dict(prop), **kwargs)

    def fit(self, adata: ad.AnnData, y="tumor_type", _=None) -> None:
        self.label_col = y
        if self.is_imblearn:
            self.model.fit(adata.X, adata.obs[y])
        elif self.method == "nb_edgeR":
            # TODO: you could rewrite this so that the params are saved as python
            # objects. but then would need to do the simulation in python and only
            # use R for parameter estimation
            self.kwargs["y"] = y

    def fit_transform(self, adata: ad.AnnData, y: str = "tumor_type") -> ad.AnnData:
        self.fit(adata, y)
        return self.transform(adata)

    def transform(self, adata: ad.AnnData) -> ad.AnnData:
        if self.is_imblearn:
            resampled_x, y = self.model.fit_resample(
                adata.X, y=adata.obs[self.label_col]
            )
            new = AnnData(
                X=resampled_x, var=self.adata.var, obs=pd.DataFrame({self.label_col: y})
            )
        elif self.method == "nb_edgeR":
            new = self.nb_edgeR_wrapper(adata, **self.kwargs)
        return new


@ru.r_cleanup
def nb_edgeR(
    adata: ad.AnnData,
    y: str,
    n: int | None = None,
    group_prop: dict[str, int] | None = None,
) -> ad.AnnData:
    ro.r("library(edgeR)")
    ru.source("simulation.R", in_r=True)
    ru.adata_to_r(adata, "dge", object="dge")
    ru.r_null_if_none(y, "group_col")
    ro.globalenv["n"] = n
    ru.r_null_if_none(group_prop, "prop", conversion=lambda x: ro.ListVector(x))
    ro.r("sim <- nb_simulate(dge, n, group_col = group_col, group_prop = prop)")
    new = ad.AnnData(
        X=np.transpose(ru.np_from_r(ro.r("sim$counts"))),
        obs=ru.df_from_r(ro.r("sim$samples")),
        var=adata.var,
    )
    return new


def spaced_resample(
    labels: np.ndarray | pd.Series,
    targets: dict[str, int | None] | None = None,
    undersample: bool = False,
    bin_step: int = 1,
    n_bins: int = 10,
    space: str = "geom",
):
    """Determine counts for each label by incrementing along a sequence

      This function creates a regularly-spaced sequence from the counts of the
    given labels with a specified numpy function (geom, histogram or linspace).
    The new value for each label is obtained by first indexing it on the sequence,
    then adding/subtracting the offset `bin_step` in an undersampling or oversampling
    context respectively

    Parameters
    ----------
    labels : array of class labels i.e. `y` passed to model.fit(X, y)
    targets : dictionary of label->bin_step or None to use the default bin step
        if the value is a float, it is interpreted as a scaling factor
    bin_step : the offset of the bins with which to promote/demote the current label
        to
    n_bins : the length of the space. The higher the less aggressive the sampling

    Returns
    -------
    dictionary mapping labels to their new counts
    """
    counts: pd.Series = (
        labels.value_counts()
        if isinstance(labels, pd.Series)
        else pd.Series(labels).value_counts()
    )
    count_dict: dict = counts.to_dict()
    match space:
        case "geom":
            vals = np.geomspace(counts.min(), counts.max(), num=n_bins)
        case "linspace":
            vals = np.linspace(counts.min(), counts.max(), num=n_bins)
        case "hist" | "histogram":
            _, vals = np.histogram(counts, bins=n_bins)
        case _:
            raise ValueError(f"Space {space} not supported!")
    for label, c in count_dict.items():
        if targets is not None and label not in targets:
            continue
        elif targets is not None and (s := targets.get(label)):
            if isinstance(s, float):
                count_dict[label] = round(c * s)
                continue
            step = s
        else:  # `targets` not given
            step = bin_step
        locs = np.where(vals <= c)[0]
        new_loc: int = locs.max() + step if not undersample else locs.max() - step
        new_loc = min(new_loc, len(vals) - 1) if not undersample else max(new_loc, 0)
        count_dict[label] = round(vals[new_loc])
    return count_dict


def dirichlet_sim(
    adata: ad.AnnData,
    y: str,
    targets: dict,
    shuffle: bool = True,
    replace: bool = False,
    n_sim: int = 3,
    rng: Generator = ut.RNG,
    prior: float = 0.5,
) -> ad.AnnData:
    """Simulate samples from a Dirichlet distribution, inspired by ALDEx2

    Parameters
    ----------
    targets : mapping of group names -> desired counts
    y : group column in adata.obs
    shuffle : bool
        whether to shuffle samples before sampling
    replace : bool
        sample from simulations with replacement, recommended if meeting the desired number
        of targets takes too many simulations
    n_sim : int
        number of simulations to generate, only used if `replace` is true
    rng : Generator
        generator object for sampling

    Returns
    -------
    AnnData object with simulated samples according to ``targets``
    """
    # Determine minimum number of simulations to be able to get counts specified
    # in ``targets`` without needing replacement
    old_counts = adata.obs[y].value_counts()
    diffs = {
        k: max(math.ceil(max(old_counts[k], v) / min(old_counts[k], v)), 1)
        for k, v in targets.items()
    }
    n_sims = max(diffs.values()) if not replace else n_sim

    counts = ut.xarray_if_sparse(adata) + prior
    dirichlet = np.apply_along_axis(lambda x: stats.dirichlet.rvs(x, n_sims), 1, counts)
    group_sims = {}
    for group, count in targets.items():
        mask = adata.obs[y] == group
        current = dirichlet[mask, :, :].reshape((-1, dirichlet.shape[2]))
        group_sims[group] = rng.choice(
            current, size=count, replace=replace, shuffle=shuffle
        )
    obs = pd.DataFrame({y: [val for k, v in targets.items() for val in [k] * v]})
    return ad.AnnData(
        X=np.concatenate(list(group_sims.values())), obs=obs, var=adata.var
    )
