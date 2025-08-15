#!/usr/bin/env ipython
import anndata as ad
import imblearn.combine as icc
import imblearn.over_sampling as ios
import imblearn.under_sampling as ius
import numpy as np
import pandas as pd
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.under_sampling.base import BaseUnderSampler
from scanpy import AnnData

# Utilities for handling imbalanced data
OTHERS: set = {"noisy"}
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

    def fit(self, adata: ad.AnnData, y="tumor_type", _=None) -> None:
        self.label_col = y
        if self.is_imblearn:
            self.model.fit(adata.X, adata.obs[y])

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
