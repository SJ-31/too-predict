#!/usr/bin/env ipython
import pickle
from typing import Callable, override

import anndata as ad
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from too_predict.evaluation import get_all_metrics
from too_predict.transformer import Transformer
from too_predict.utils import RANDOM_STATE, RNG, adata_to_df, find_confounded, str_mode

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

    def holdout(
        self,
        adata: ad.AnnData,
        split_fn: Callable[[ad.AnnData], tuple[ad.AnnData, ad.AnnData]],
        label_col="tumor_type",
    ) -> dict:
        """Wrapper function for doing the classic holdout method (train-test-split)

        Parameters
        ---------
        split_fn: A function that splits adata into a tuple of train, test

        Return
        ------
        A dictionary containing model evaluation results for each unique instance of
            `group_col`

        Notes
        -----
        - Only use in place of cross_validate with StratifiedGroupKFold
            when the group category to be evaluated is
            confounded with the target labels
        """
        adata = adata.copy()
        n = len(adata)
        x_train, x_test = split_fn(adata)
        split_prop = np.array([len(x_train), len(x_test)]) / n
        results: dict = {"split_prop": split_prop}
        self.fit(x_train, label_col=label_col)
        proba = self.predict_proba(x_test)
        y_true = x_test.obs[label_col]
        y_uniques = y_true.unique()
        res: dict = get_all_metrics(y_true, proba, self.classes_)
        for k, v in res.items():
            if k == "cm":
                res[k] = v.loc[v.index.isin(y_uniques), v.columns.isin(y_uniques)]
            elif isinstance(v, pd.DataFrame) and v.shape[0] > 0:
                res[k] = v.loc[v["class"].isin(y_uniques), :]
        results["results"] = res
        return results

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
        others: dict = {
            "fold": [],
            "acc": [],
            "jaccard": [],
            "kappa": [],
            "fowlkes_mallows": [],
            "mcc": [],
        }
        for fold, (train_i, test_i) in enumerate(splits):
            x_train = N[train_i]
            self.fit(x_train, label_col=label_col)

            x_test = N[test_i]
            y_true = labels.iloc[test_i]  # True values

            proba = self.predict_proba(x_test)
            res: dict = get_all_metrics(y_true, proba, self.classes_)
            others["fold"].append(fold)
            for o in others.keys():
                if o != "fold":
                    others[o].append(res[o])
            cm[fold] = res["cm"]
            for df, lst in zip(
                [res["report"], res["prec_recall"], res["roc"]],
                [report, prec_recall, roc],
            ):
                df["fold"] = fold
                lst.append(df)
        return {
            "cm": cm,
            "misc": pd.DataFrame(others),
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


class XgboostPred(PredBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(model=XGBClassifier(), **kwargs)

    def fit(self, X: ad.AnnData, label_col="tumor_type") -> None:
        self.encoder = LabelEncoder()
        X.obs[label_col] = self.encoder.fit_transform(X.obs[label_col])
        return super().fit(X, label_col)

    def predict(self, X: ad.AnnData) -> np.ndarray:
        vals = super().predict(X)
        return self.encoder.inverse_transform(vals)

    @property
    def classes_(self):
        return self.encoder.inverse_transform(super().classes_)


class RandomForestPred(PredBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(model=RandomForestClassifier(random_state=RNG, **kwargs))


class AlrBase(PredBase):
    def __init__(
        self,
        model,
        references: dict | list,
        var_col: str = "GENEID",
        weights=None,
    ) -> None:
        super().__init__(
            model=AlrEstimator(model=model, references=references, weights=weights)
        )
        if var_col:
            self.var_col = var_col

    @override
    def fit(self, X: ad.AnnData, label_col="tumor_type") -> None:
        if sparse.isspmatrix(X.X):
            vals = X.X.toarray()
        else:
            vals = X.X
        counts = pd.DataFrame(vals, columns=X.var[self.var_col], index=None)
        self.var = X.var
        self._is_fitted = True
        self.model.fit(counts, X.obs[label_col])

    @override
    def predict(self, X: ad.AnnData) -> np.ndarray:
        df = adata_to_df(X, var_col=self.var_col)
        return self.model.predict(df)

    @override
    def predict_proba(self, X: ad.AnnData) -> np.ndarray:
        df = adata_to_df(X, var_col=self.var_col)
        return self.model.predict_proba(df)


class SimPred(PredBase):
    def __init__(self, model, method) -> None:
        super().__init__(model=SimEstimator(method, model=model))


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


class SimEstimator:
    def __init__(self, method: str, model, **kwargs) -> None:
        """A class to fit models where the data preprocessing involves some
        form of simulation
        e.g. generating Monte Carlo instances

        Parameters
        ----------
        predict_from_sim : whether the model should make predictions from data after
            running the simulation process
        """
        self.method = method
        self.model = model
        self.kwargs = kwargs

    @staticmethod
    def _validate_x(X: np.ndarray) -> None:
        shape = X.shape
        if len(shape) != 3:
            raise ValueError(
                "X must be an array of shape (n_simulations, n_obs, n_features)!"
            )
        else:
            print(f"Fitting to data with {shape[0]} instances...")

    def _simulate(self, X: np.ndarray) -> np.ndarray:
        array: np.ndarray = Transformer(
            self.method, inplace=False, make_sparse=False, **self.kwargs
        ).fit_transform(X)
        return array

    def _format_instances(self, X: np.ndarray, labels=None):
        self._validate_x(X)
        counts = np.concatenate(X)
        if labels is not None:
            labels = np.concatenate([np.copy(labels) for _ in range(X.shape[0])])
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
        X = self._simulate(X)
        X, y = self._format_instances(X, y)
        self.model.fit(X, y)

    def predict_proba(self, X) -> np.ndarray:
        X = self._simulate(X)
        self._validate_x(X)
        all_proba = []
        for i in range(X.shape[0]):
            cur = self.model.predict_proba(X[i, :, :])
            all_proba.append(cur)
        return np.array(all_proba).mean(axis=0)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        voted = [self.model.classes_[c] for c in np.argmax(proba, axis=1)]
        return np.array(voted)

    @property
    def classes_(self):
        return self.model.classes_
