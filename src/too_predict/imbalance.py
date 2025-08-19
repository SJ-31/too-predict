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
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter

import too_predict.r_utils as ru
import too_predict.utils as ut

# Utilities for handling imbalanced data
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


class Balancer:
    def __init__(self, method: str, **kwargs) -> None:
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
        self.inv_ecdfs: dict[str, stats.interp1d] | None = None

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

    def empirical_wrapper(self, adata: ad.AnnData, **kwargs) -> ad.AnnData:
        group_counts = self.check_sampling_strategy(
            y=adata.obs[self.label_col],
            strategy=kwargs.pop("sampling_strategy", "auto"),
            type="over-sampling",
        )
        new, self.inv_ecdfs = empirical(
            adata=adata,
            y=self.label_col,
            rng=kwargs.get("rng", ut.RNG),
            targets=group_counts,
        )
        return new

    def nb_edgeR_wrapper(self, adata: ad.AnnData, **kwargs) -> ad.AnnData:
        if n := kwargs.pop("n", None):
            return nb_edgeR(adata=adata, n=n, **kwargs)
        group_counts = self.check_sampling_strategy(
            y=adata.obs[self.label_col],
            strategy=kwargs.pop("sampling_strategy", "auto"),
            type="over-sampling",
        )
        if not kwargs.get("blocking", False):
            n = np.sum(list(group_counts.values()))
            prop = {k: (v / n).item() for k, v in group_counts.items()}
            return nb_edgeR(adata=adata, n=n.item(), targets=dict(prop), **kwargs)
        else:
            return nb_edgeR(adata=adata, targets=group_counts, **kwargs)

    def splatter_wrapper(self, adata: ad.AnnData, **kwargs) -> ad.AnnData:
        counts = self.check_sampling_strategy(
            y=adata.obs[self.label_col],
            strategy=kwargs.pop("sampling_strategy", "auto"),
            type="over-sampling",
        )
        return splatter_bulk(adata, y=self.label_col, targets=counts, **kwargs)

    def dirichlet_wrapper(self, adata: ad.AnnData, **kwargs) -> ad.AnnData:
        counts = self.check_sampling_strategy(
            y=adata.obs[self.label_col],
            strategy=kwargs.pop("sampling_strategy", "auto"),
            type="over-sampling",
        )
        _ = kwargs.pop("as_transform", None)
        return dirichlet_sim(adata, y=self.label_col, targets=counts, **kwargs)

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
                X=resampled_x, var=adata.var, obs=pd.DataFrame({self.label_col: y})
            )
        elif self.method == "nb_edgeR":
            new = self.nb_edgeR_wrapper(adata, **self.kwargs)
        elif self.method == "dirichlet":
            new = self.dirichlet_wrapper(adata, **self.kwargs)
        elif self.method == "empirical":
            new = self.empirical_wrapper(adata, **self.kwargs)
        elif self.method == "splatter":
            new = self.splatter_wrapper(adata, **self.kwargs)
        else:
            raise ValueError(f"method {self.method} not implemented!")
        return new


@ru.r_cleanup
def nb_edgeR(
    adata: ad.AnnData,
    y: str,
    n: int | None = None,
    targets: dict[str, int] | None = None,
    blocking: bool = True,
    sample_mus: bool = False,
) -> ad.AnnData:
    ro.r("library(edgeR)")
    ru.source("simulation.R", in_r=True)
    ro.globalenv["sample_mus"] = sample_mus
    if not blocking:
        ru.adata_to_r(adata, "dge", object="dge")
        ru.r_null_if_none(y, "group_col")
        ru.r_null_if_none(n, "n")
        ru.r_null_if_none(targets, "prop", conversion=lambda x: ro.ListVector(x))
        ro.r("""sim <- nb_simulate(dge, n, group_col = group_col,
                        group_prop = prop, sample_mus = sample_mus)""")
        new = ad.AnnData(
            X=np.transpose(ru.np_from_r(ro.r("sim$counts"))),
            obs=ru.df_from_r(ro.r("sim$samples")),
            var=adata.var,
        )
    elif targets is not None:
        groups = []
        mats = []
        for group, count in targets.items():
            current = adata[adata.obs[y] == group, :]
            ru.adata_to_r(current, "dge", object="dge")
            ro.globalenv["n"] = count.item()
            groups.extend([group] * count)
            ro.r("sim <- nb_simulate(dge, n = n, sample_mus = sample_mus)")
            mats.append(np.transpose(ru.np_from_r(ro.r("sim$counts"))))
        new = ad.AnnData(
            X=np.vstack(mats), obs=pd.DataFrame({y: groups}), var=adata.var
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


# * Simulation functions


def dirichlet_sim(
    adata: ad.AnnData,
    y: str | None = None,
    targets: dict | None = None,
    shuffle: bool = True,
    replace: bool = False,
    n_sim: int = 3,
    rng: Generator | None = ut.RNG,
    prior: float = 0.5,
    as_transform: bool = False,
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
    counts = ut.xarray_if_sparse(adata) + prior
    if not as_transform and targets is None:
        raise ValueError("`targets` must be provided if using to simulate samples!")
    elif not as_transform and y is None:
        raise ValueError("`y` must be provided if using to simulate samples!")
    if not as_transform:
        old_counts = adata.obs[y].value_counts()
        diffs = {
            k: max(math.ceil(max(old_counts[k], v) / min(old_counts[k], v)), 1)
            for k, v in targets.items()
        }
        n_sims = max(diffs.values()) if not replace else n_sim

        dirichlet = np.apply_along_axis(
            lambda x: stats.dirichlet.rvs(x, n_sims), 1, counts
        )
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
    dirichlet = np.apply_along_axis(lambda x: stats.dirichlet.rvs(x, 1), 1, counts)
    transformed = dirichlet[:, 0, :]
    return ad.AnnData(X=transformed, obs=adata.obs, var=adata.var)


def empirical(adata: ad.AnnData, y, targets: dict, rng) -> tuple[ad.AnnData, dict]:
    """Simulate samples using inverse of empirical distribution for each gene,
    blocking by group
    """

    def get_inv_ecdf(arr: np.ndarray):
        fn = monotone_fn_inverter(ECDF(arr), arr)
        return fn

    def make_samples(inv_ecdfs, n):
        return np.vstack(
            [
                np.hstack(
                    list(map(lambda fn: fn(max(rng.random(), fn.x[0])), inv_ecdfs))
                )
                for _ in range(n)
            ]
        )

    samples = []
    groups = []
    inv_ecdfs: dict = {}
    counts = ut.xarray_if_sparse(adata)
    for group, count in targets.items():
        masked = counts[adata.obs[y] == group, :]
        cur_fns = np.apply_along_axis(get_inv_ecdf, 0, masked)
        samples.append(make_samples(cur_fns, count))
        inv_ecdfs[group] = cur_fns
        groups.extend([group] * count)
    return ad.AnnData(
        X=np.vstack(samples), obs=pd.DataFrame({y: groups}), var=adata.var
    ), inv_ecdfs


@ru.r_cleanup
def splatter_bulk(adata: ad.AnnData, y: str, targets: dict) -> ad.AnnData:
    groups = []
    mats = []
    for group, count in targets.items():
        cur = adata[adata.obs[y] == group, :]
        ru.adata_to_r(cur, "sce", "sce")
        ro.globalenv["count"] = count.item()
        ro.r("params <- splatter::splatEstimate(sce)")
        ro.r("sim <- splatter::splatSimulate(params, batchCells = count)")
        mat = ru.np_from_r(ro.r("assays(sim)$counts")).transpose()
        mats.append(mat)
        groups.extend([group] * count)
    return ad.AnnData(X=np.vstack(mats), var=adata.var, obs=pd.DataFrame({y: groups}))
