#!/usr/bin/env ipython
import anndata as ad
import imblearn.over_sampling as ios
import imblearn.under_sampling as ius
import pandas as pd
from scanpy import AnnData

# Utilities for handling imbalanced data
OTHERS: set = {"tmp"}
IMBLEARN_METHODS: set = {
    "SMOTE",
    "KMeansSMOTE",
    "TomekLinks",
    "ADASYN",
    "SVMSMOTE",
    "BorderLineSMOTE",
    "RandomOverSampler",
    "RandomUnderSampler",
    "TomekLinks",
}
IMPLEMENTED_BALANCE: set = IMBLEARN_METHODS | OTHERS


class Balancer:
    def __init__(self, method: str, label_col: str | None = None, **kwargs) -> None:
        if method not in IMPLEMENTED_BALANCE:
            raise ValueError(f"Method {method} not implemented!")
        self.model = None
        self.method = method
        self.label_col: str | None = label_col
        self.is_imblearn: bool = False
        if self.method in IMBLEARN_METHODS:
            self.is_imblearn = True
            self.model = self._imblearn_model(method, **kwargs)
        self.kwargs = kwargs

    def _imblearn_model(self, model, **kwargs):
        match model:
            case "SMOTE":
                return ios.SMOTE(**kwargs)
            case "KMeansSMOTE":
                return ios.KMeansSMOTE(**kwargs)
            case "SVMSMOTE":
                return ios.SVMSMOTE(**kwargs)
            case "ADASYN":
                return ios.ADASYN(**kwargs)
            case "RandomOverSampler":
                return ios.RandomOverSampler(**kwargs)
            case "TomekLinks":
                return ius.TomekLinks(**kwargs)
            case "BorderLineSMOTE":
                return ios.BorderlineSMOTE(**kwargs)
            case "RandomUnderSampler":
                return ius.RandomUnderSampler(**kwargs)

    def fit(self, adata: ad.AnnData, y="tumor_type", _=None) -> None:
        self.adata = adata.copy()
        self.label_col = y
        if self.is_imblearn:
            self.model.fit(adata.X, adata.obs[y])

    def fit_transform(self, adata: ad.AnnData, y, _=None) -> ad.AnnData:
        self.fit(adata, y)
        return self.transform()

    def transform(self, _=None) -> ad.AnnData:
        if self.is_imblearn:
            resampled_x, y = self.model.fit_resample(
                self.adata.X, y=self.adata.obs[self.label_col]
            )
            new = AnnData(
                X=resampled_x, var=self.adata.var, obs=pd.DataFrame({self.label_col: y})
            )
        return new
