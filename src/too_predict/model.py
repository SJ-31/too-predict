#!/usr/bin/env ipython
import pickle
from typing import override

import anndata as ad
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

from too_predict.evaluation import get_all_metrics
from too_predict.simulation import Simulator
from too_predict.transformer import Transformer
from too_predict.utils import RANDOM_STATE, RNG, adata_to_df

# def train_test_split_adata():
# <2025-02-13 Thu> TODO: make a batch-aware and stratified test_train_split fn
# for adata objects


class PredBase:
    """
    A wrapper class around an sklearn-style classifier to streamline
    interactions between the anndata object, as well as carrying out imputation
    and normalization
    """

    def __init__(
        self,
        model,
    ) -> None:
        self.model = model
        self._estimator_type = "classifier"
        self._is_fitted = False
        self.missing_features = (
            None  # Requested features to subset by that weren't found
        )
        self.var = None

    def fit(
        self,
        X: ad.AnnData,
        label_col="tumor_type",
    ) -> None:
        """Fit model to the given adata object

        Should ignore any previous calls to fit
        Parameters
        ----------
        X : data to fit to
        """
        if label_col not in X.obs.columns:
            raise ValueError(f"The column '{label_col}' is not present in X.obs")
        self.var = X.var
        self._is_fitted = True
        self.model.fit(X.X, X.obs[label_col])

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    def __sklearn_is_fitted__(self):
        return self._is_fitted

    def predict_proba(self, X: ad.AnnData) -> np.ndarray:
        return self.model.predict_proba(X.X)

    def load(self, path: str) -> None:
        """Load the fitted estimator from the saved path"""
        self.model = pickle.load(path)

    def predict(self, X: ad.AnnData) -> np.ndarray:
        return self.model.predict(X.X)

    @property
    def classes_(self):
        return self.model.classes_

    def cross_validate(
        self,
        adata,
        label_col="tumor_type",
        group_col="",
        n_splits=5,
        random_state=RANDOM_STATE,
    ) -> dict:
        """Evaluate model performance with cross-validation"""
        N = adata.copy()
        labels = N.obs[label_col]
        if not group_col:
            cv = ms.StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            splits = cv.split(N.X, labels)
        else:
            cv = ms.StratifiedGroupKFold(
                n_splits=n_splits, random_state=random_state, shuffle=True
            )
            splits = cv.split(N.X, labels, groups=N.obs[group_col])
        cm: dict = {}
        roc, prec_recall, report = [], [], []
        accs: dict = {"fold": [], "acc": []}
        for fold, (train_i, test_i) in enumerate(splits):
            x_train = N[train_i]
            self.fit(x_train, label_col=label_col)

            x_test = N[test_i]
            y_true = labels.iloc[test_i]  # True values

            proba = self.predict_proba(x_test)
            res: dict = get_all_metrics(y_true, proba, self.classes_)
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

    def rfecv(
        self,
        X: ad.AnnData,
        label_col="tumor_type",
        rfecv_params: dict = None,
    ) -> RFECV:
        """Perform recursive feature elimination with cross validation

        Parameters
        ----------
        X : anndata object to pass to RFECV.fit
        rfecv_params : parameters to pass to rfecv

        Returns
        -------
        Fitted RFECV object
        """
        params = rfecv_params if rfecv_params else {}
        rfecv = RFECV(self.model, **params)
        rfecv.fit(X.X, X.obs[label_col])
        return rfecv


class RandomForestPred(PredBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(model=RandomForestClassifier(random_state=RNG, **kwargs))

        # <2025-02-21 Fri> be sure to try out extra trees


class AlrBase(PredBase):
    def __init__(
        self,
        model,
        references: dict | list,
        weights=None,
    ) -> None:
        super().__init__(
            model=AlrEstimator(model=model, references=references, weights=weights)
        )

    @override
    def fit(self, X: ad.AnnData, label_col="tumor_type", var_col="GENENAME") -> None:
        if not isinstance(X.X, np.ndarray):
            vals = X.X.toarray()
        else:
            vals = X.X
        counts = pd.DataFrame(vals, columns=X.var[var_col], index=None)
        self.var = X.var
        self._is_fitted = True
        self.model.fit(counts, X.obs[label_col])

    @override
    def predict(self, X: ad.AnnData, var_col="GENENAME") -> np.ndarray:
        df = adata_to_df(X, var_col=var_col)
        return self.model.predict(df)

    @override
    def predict_proba(self, X: ad.AnnData, var_col="GENENAME") -> np.ndarray:
        df = adata_to_df(X, var_col=var_col)
        return self.model.predict_proba(df)


# * Estimators
# Lower-level estimators that more directly follow the sklearn API
# intended to be passed to the 'model' argument of classes inheriting
# PredBase


class AlrEstimator:
    """Base class for aggregating the results of classifier models trained on
    ALR-transformed with multiple references e.g. different genes

    Predicted labels are obtained with soft voting (weighted average probabilities)
    TODO: try to parallelize training here

    Don't pre-transform data with this model
    """

    def __init__(
        self,
        model,
        references: dict | list,
        weights=None,
    ) -> None:
        self.refs = (
            list(references.keys()) if isinstance(references, dict) else references
        )
        self.weights = references.values() if isinstance(references, dict) else None
        if weights:
            self.weights = weights
        self.models = {r: clone(model) for r in self.refs}
        self.n_fit = 0
        self.n_pred = 0
        self.missing_references = []

    def _alr(self, X: pd.DataFrame, by: str, **kwargs) -> np.ndarray:
        result: np.ndarray = Transformer(
            "alr", impute_fn=None, inplace=False, by=by, **kwargs
        ).fit_transform(X)
        return result

    def fit(
        self,
        X: pd.DataFrame,
        y,
        **kwargs,
    ) -> None:
        self.missing_references = []
        self.n_fit = 0
        self._is_fitted = True
        for r in self.refs:
            if r in X.columns:
                transformed = self._alr(X, r, **kwargs)
                self.models[r].fit(transformed, y)
                self.n_fit += 1
            else:
                self.missing_references.append(r)
                print(f"WARNING: reference {r} missing")
        print(
            f"Fit with {self.n_fit} ({self.n_fit // len(self.refs) * 100}) references"
        )

    def predict(self, X) -> np.ndarray:
        proba_df = pd.DataFrame(self.predict_proba(X), columns=self.classes_)
        return np.array(proba_df.idxmax(1))

    def predict_proba(self, X) -> np.ndarray:
        """Get predictions using all trained estimators for each reference

        Parameters
        ----------
        X : a dataframe where columns are features
        """
        proba = []
        self.n_pred = 0
        self.missing_references = []
        for r, m in self.models.items():
            if r in X.columns:
                transformed = self._alr(X, r)
                proba.append(m.predict_proba(transformed))
                self.n_pred += 1
            else:
                # Don't try to normalize by it if it isn't present
                print(f"WARNING: reference {r} missing")
                self.missing_references.append(r)
                proba.append(np.zeros((len(self.classes_), len(X.shape[0]))))

        message = f"Predicted with {self.n_pred} ({self.n_pred // len(self.refs) * 100}) references"
        print(message)
        proba = np.array(proba)

        if self.weights is not None and len(self.weights) == proba.shape[0]:
            proba = np.reshape(self.weights, [proba.shape[0], 1, 1]) * proba
        return proba.mean(axis=0)

    @property
    def classes_(self):
        return self.models[self.refs[0]].classes_


# <2025-02-23 Sun> TODO: still haven't tested this yet
class SimEstimator:
    def __init__(
        self,
        simulation: str,
        model,
        prefix: str = "mc_",
        n: int = 10,
        predict_from_sim: bool = False,
    ) -> None:
        """A class to fit models where the data preprocessing involves some
        form of simulation
        e.g. generating Monte Carlo instances

        Parameters
        ----------
        predict_from_sim : whether the model should make predictions from data after
            running the simulation process
        """
        self.model = model
        self.s_method = simulation
        self.predict_from_sim = predict_from_sim
        self.prefix = prefix
        self.cross_validating = False
        self.n = n

    def _get_instances(self, X, labels=None):
        instances = Simulator(
            X,
            self.s_method,
            self.n,
            prefix=self.prefix,
            inplace=False,
            make_sparse=False,  # Required for concatenation
        ).run()
        counts = np.concatenate(instances)
        if labels is not None:
            labels = np.concatenate([np.copy(labels) for _ in range(self.n)])
        return counts, labels

    def fit(
        self,
        X,
        y,
    ) -> None:
        """
        Fit the underlying model on combined data from the all the Monte Carlo instances

        @param mc_kwargs
            - n: number of Monte Carlo instances to generate
            - prefix: number
        @param instance_prefix: prefix denoting layers in the the adata object containing
        the instances
        """
        X, y = self._get_instances(X, y)
        self.model.fit(X, y)

    def predict_proba(self, X) -> np.ndarray:
        if self.predict_from_sim:
            X, _ = self._get_instances(X)
        return self.model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        if self.predict_from_sim:
            X = self._get_instances(X)
        return self.model.predict(X)

    @property
    def classes_(self):
        return self.model.classes_
