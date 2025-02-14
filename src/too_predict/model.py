#!/usr/bin/env ipython
import gc
import pickle
from abc import abstractmethod
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
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
        self.features = features
        self.impute: Callable = Imputer(imputation).run
        self.i_method = imputation.lower()

    def fit(self, X: ad.AnnData, label_col="tumor_type", preprocess=True) -> None:
        if preprocess:
            X = self.preprocess(X)
        self.model.fit(X.X, X.obs[label_col])

    def predict_proba(self, X: ad.AnnData, preprocess=True) -> np.ndarray:
        if preprocess:
            X = self.preprocess(X)
        return self.model.predict_proba(X.X)

    def load(self, path: str) -> None:
        """Load the fitted estimator from the saved path"""
        self.model = pickle.load()

    def predict(self, X: ad.AnnData, preprocess: bool = True) -> np.ndarray:
        if preprocess:
            N = self.preprocess(X)
        return self.model.predict(N)

    def preprocess(self, adata: ad.AnnData) -> ad.AnnData:
        if self.features is not None:
            adata = adata[:, self.features]
        if self.n_method is not None:
            normalized: ad.AnnData = Normalizer(
                adata, self.n_method, impute_fn=self.impute, inplace=False
            ).run()
            gc.collect()
            return normalized
        return adata.copy()

    def _classes():
        return self.model.classes_

    def cross_validate(self, adata, cv=None, label_col="tumor_type") -> dict:
        """Determine model performance with cross-validation"""
        if not cv:
            cv = ms.StratifiedKFold(n_splits=5)
        N = self.preprocess(adata)
        cm: dict = {}
        roc, prec_recall, report = [], [], []
        accs: dict = {"fold": [], "acc": []}
        for fold, (train_i, test_i) in enumerate(cv.split(N.X, N.obs[label_col])):
            x_train = N[train_i]
            self.fit(x_train, preprocess=False)

            x_test = N[test_i]
            y_true = N.obs[label_col].iloc[test_i]  # True values

            proba = self.predict_proba(x_test, preprocess=False)
            res: dict = get_all_metrics(y_true, proba, self._classes())
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


class TooRandomForest(Base):
    def __init__(self, normalization: str, imputation: str) -> None:
        super().__init__(
            normalization=normalization,
            imputation=imputation,
            model=RandomForestClassifier(),
            features=None,
        )


# TODO: going to need a meta model or something to implement the ALR normalization properly
