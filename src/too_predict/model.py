#!/usr/bin/env ipython
import gc
import pickle
from abc import abstractmethod
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import sklearn.model_selection as ms
from sklearn.ensemble import RandomForestClassifier

from too_predict.evaluation import get_all_metrics
from too_predict.imputer import Imputer
from too_predict.normalizer import Normalizer

# def train_test_split_adata():
# <2025-02-13 Thu> TODO: make a batch-aware and stratified test_train_split fn
# for adata objects


class Base:
    """
    A wrapper class around an sklearn-style classifier to streamline
    interactions between the anndata object, as well as carrying out imputation
    and normalization
    """

    def __init__(
        self,
        normalization: str,
        imputation: str,
        model,
        features=None,
    ) -> None:
        self.model = model
        if normalization:
            self.n_method = normalization.lower()
        else:
            self.n_method = None
        self.impute: Callable = Imputer(imputation).run
        self.i_method = imputation.lower()

    def fit(self, X: ad.AnnData, label_col="tumor_type") -> None:
        N = self.normalize(X)
        self.model.fit(N.X, N.obs[label_col])

    def load(self, path: str) -> None:
        """Load the fitted estimator from the saved path"""
        self.model = pickle.load()
        pass

    def predict(self, X: ad.AnnData) -> np.ndarray:
        N = self.normalize(X)
        return self.model.predict(N)

    def normalize(self, adata: ad.AnnData) -> ad.AnnData:
        if self.n_method is not None:
            normalized: ad.AnnData = Normalizer(
                adata, self.n_method, impute_fn=self.impute, inplace=False
            ).run()
            gc.collect()
            return normalized
        print("WARNING: Using raw counts")
        return adata.copy()

    def cross_validate(self, adata, cv=None, label_col="tumor_type") -> dict:
        """Determine model performance with cross-validation"""
        if not cv:
            cv = ms.StratifiedKFold(n_splits=5)
        N = self.normalize(adata)
        cm: dict = {}
        roc, prec_recall, report = [], [], []
        accs: dict = {"fold": [], "acc": []}
        for fold, (train_i, test_i) in enumerate(cv.split(N.X, N.obs[label_col])):
            x_train = N[train_i].X
            y_train = N.obs[label_col].iloc[train_i]
            self.model.fit(x_train, y=y_train)

            x_test = N[test_i].X
            y_true = N.obs[label_col].iloc[test_i]  # True values

            proba = self.model.predict_proba(x_test)
            res: dict = get_all_metrics(y_true, proba, self.model.classes_)
            accs["fold"].append(fold)
            accs["acc"].append(res["acc"])
            cm[fold] = res["cm"]
            for df, lst in zip(
                [res["report"], res["prec_recall"], res["roc"]],
                [report, prec_recall, roc],
            ):
                df["fold"] = fold
                lst.append(df)
        return {
            "cm": cm,
            "acc": pd.DataFrame(accs),
            "report": pd.concat(report),
            "prec_recall": pd.concat(prec_recall),
            "roc": pd.concat(roc),
        }


# TODO: going to need a meta model or something to implement the ALR normalization properly
