#!/usr/bin/env ipython
import pickle
from typing import Callable, override

import anndata as ad
import numpy as np
import pandas as pd
import sklearn.metrics as sm
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from too_predict.evaluation import cross_validate, get_all_metrics, holdout
from too_predict.imputer import Imputer
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

    def __init__(self, model, make_dense: bool = False) -> None:
        self.model = model
        self._estimator_type = "classifier"
        self._is_fitted = False
        self.missing_features = (
            None  # Requested features to subset by that weren't found
        )
        self.make_dense = make_dense
        self.var = None

    def fit(
        self,
        X: ad.AnnData,
        y="tumor_type",
    ) -> None:
        """Fit model to the given adata object

        Should ignore any previous calls to fit
        Parameters
        ----------
        X : data to fit to
        """
        if y not in X.obs.columns:
            raise ValueError(f"The column '{y}' is not present in X.obs")
        self.var = X.var
        self._is_fitted = True
        self.model.fit(self._check_dense(X.X), X.obs[y])

    def _check_dense(self, X):
        if not self.make_dense or not sparse.isspmatrix(X):
            return X
        elif self.make_dense and sparse.isspmatrix(X):
            return X.toarray()

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    def __sklearn_is_fitted__(self):
        return self._is_fitted

    def predict_proba(self, X: ad.AnnData) -> np.ndarray:
        return self.model.predict_proba(self._check_dense(X.X))

    def decision_function(self, X: ad.AnnData) -> np.ndarray:
        return self.model.decision_function(self._check_dense(X.X))

    def load(self, path: str) -> None:
        """Load the fitted estimator from the saved path"""
        self.model = pickle.load(path)

    def predict(self, X: ad.AnnData) -> np.ndarray:
        return self.model.predict(self._check_dense(X.X))

    @property
    def classes_(self):
        return self.model.classes_

    def holdout(
        self,
        adata: ad.AnnData,
        split_fns: dict[str, Callable[[ad.AnnData], tuple[ad.AnnData, ad.AnnData]]],
        label_col="tumor_type",
    ) -> dict:
        return holdout(self, adata, split_fns, label_col)

    def cross_validate(
        self,
        adata,
        label_col="tumor_type",
        group_col="",
        n_splits=5,
        random_state=RANDOM_STATE,
    ) -> dict:
        return cross_validate(
            self,
            adata,
            label_col=label_col,
            group_col=group_col,
            n_splits=n_splits,
            random_state=random_state,
        )

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


class AlrBase(PredBase):
    def __init__(
        self,
        model,
        references: dict | list,
        var_col: str = "GENEID",
        imputation: str = "none",
        weights=None,
        n_refs=-1,
        make_dense: bool = False,
    ) -> None:
        super().__init__(
            model=AlrEstimator(
                model=model,
                references=references,
                weights=weights,
                imputation=imputation,
                n_refs=n_refs,
            ),
            make_dense=make_dense,
        )
        if var_col:
            self.var_col = var_col

    @override
    def fit(self, X: ad.AnnData, y="tumor_type") -> None:
        if sparse.isspmatrix(X.X):
            vals = X.X.toarray()
        else:
            vals = X.X
        counts = pd.DataFrame(vals, columns=X.var[self.var_col], index=None)
        self.var = X.var
        self._is_fitted = True
        self.model.fit(counts, X.obs[y])

    @override
    def predict(self, X: ad.AnnData) -> np.ndarray:
        df = adata_to_df(X, var_col=self.var_col)
        return self.model.predict(df)

    @override
    def predict_proba(self, X: ad.AnnData) -> np.ndarray:
        df = adata_to_df(X, var_col=self.var_col)
        return self.model.predict_proba(df)


class SimPred(PredBase):
    def __init__(self, model, method, **kwargs) -> None:
        if not kwargs:
            kwargs = {"n_instances": 15}
        super().__init__(model=SimEstimator(method, model=model, **kwargs))


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
        imputation: str = "none",
        n_refs: int = -1,
    ) -> None:
        self.all_refs: np.ndarray = np.array(
            list(references.keys()) if isinstance(references, dict) else references
        )
        self.model = model
        self.all_weights: np.ndarray = np.array(
            references.values() if isinstance(references, dict) else None
        )
        self.all_weights = weights if weights is not None else np.array([])
        self.cur_refs: list = []
        self.cur_weights: list = []
        self.impute_fn = Imputer(imputation)
        self.n_refs = -1 if (n_refs >= len(references) or n_refs < 0) else n_refs
        self.n_fit = 0
        self.n_pred = 0
        self.missing_references = []

    def _alr(self, X: pd.DataFrame, by: str, **kwargs) -> np.ndarray:
        result: np.ndarray = Transformer(
            "alr", impute_fn=self.impute_fn, inplace=False, by=by, **kwargs
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
        if self.n_refs != -1:
            mask = [x in X.columns for x in self.all_refs]
            ref_options: np.ndarray = self.all_refs[mask]
            chosen = RNG.choice(
                range(len(ref_options)), size=self.n_refs, replace=False
            )
            self.cur_refs = [ref_options[i] for i in chosen]
            if self.all_weights.size > 0:
                weight_options: np.ndarray = self.all_weights[mask]
                self.cur_weights = [weight_options[i] for i in chosen]
        else:
            self.cur_refs = np.copy(self.all_refs)
            self.cur_weights = np.copy(self.all_weights)
        self.models = {r: clone(self.model) for r in self.cur_refs}

        for r in self.cur_refs:
            if r in X.columns:
                transformed = self._alr(X, r, **kwargs)
                self.models[r].fit(transformed, y)
                self.n_fit += 1
            else:
                self.missing_references.append(r)
                print(f"WARNING: reference {r} missing")
        print(
            f"Fit with {self.n_fit} references ({self.n_fit // len(self.cur_refs) * 100}% success)"
        )
        if self.n_fit == 0:
            raise ValueError("No model could be fitted!")

    def predict(self, X) -> np.ndarray:
        proba_df = pd.DataFrame(self.predict_proba(X), columns=self.classes_)
        return np.array(proba_df.idxmax(1))

    def _predict_score(self, X, score_method: str) -> np.ndarray:
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
                if score_method == "predict_proba":
                    proba.append(m.predict_proba(transformed))
                elif score_method == "decision_function":
                    proba.append(m.decision_function(transformed))
                else:
                    raise ValueError(f"Score method {score_method} not recognized!")
                self.n_pred += 1
            else:
                # Don't try to normalize by it if it isn't present
                print(f"WARNING: reference {r} missing")
                self.missing_references.append(r)
                proba.append(np.zeros((len(self.classes_), len(X.shape[0]))))

        message = f"Predicted with {self.n_pred} ({self.n_pred // len(self.cur_refs) * 100}) references"
        print(message)
        proba = np.array(proba)

        if len(self.cur_weights) == proba.shape[0]:
            proba = np.reshape(self.cur_weights, [proba.shape[0], 1, 1]) * proba
        return proba.mean(axis=0)

    def predict_proba(self, X) -> np.ndarray:
        return self._predict_score(X, "predict_proba")

    def decision_function(self, X) -> np.ndarray:
        return self._predict_score(X, "decision_function")

    @property
    def classes_(self):
        return self.models[next(iter(self.models))].classes_


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


class XGBEstimator:
    """Convenience wraper for XGBClassifier that supports string labels
    Early stopping parameters are set to be consistent with sklearn's defaults
    """

    def __init__(self, early_stop=False, **kwargs) -> None:
        if early_stop:
            self.model: XGBClassifier = XGBClassifier(
                early_stopping_rounds=10, eval_metric=sm.log_loss, **kwargs
            )
        else:
            self.model: XGBClassifier = XGBClassifier(**kwargs)

    def fit(self, X, y) -> None:
        self.encoder = LabelEncoder()
        recoded = self.encoder.fit_transform(y)
        self.model.fit(X, recoded)

    def predict(self, X) -> np.ndarray:
        vals = self.model.predict(X)
        return self.encoder.inverse_transform(vals)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_params(self, deep=True) -> dict:
        return self.model.get_params(deep)

    def set_params(self, **params) -> dict:
        return self.model.set_params(**params)

    def get_metadata_routing(self):
        return self.model.get_metadata_routing()

    def score(self, X, y, sample_weight=None):
        return self.model.score(X, y, sample_weight)

    @property
    def classes_(self):
        return self.encoder.inverse_transform(self.model.classes_)


class SVMEstimator:
    """Wrapper class for Sklearn estimators
    Only need this so as to optinally do scaling as part of the model
    """

    def __init__(self) -> None:
        pass
