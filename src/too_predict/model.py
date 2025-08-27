#!/usr/bin/env ipython
import pickle
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Literal, override

import anndata as ad
import numpy as np
import pandas as pd
import sklearn.metrics as sm
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

from too_predict.corrector import Corrector
from too_predict.imbalance import Balancer
from too_predict.imputer import Imputer
from too_predict.transformer import Transformer
from too_predict.utils import RNG, adata_to_df


class PredBase:
    """
    A wrapper class around an sklearn-style classifier to streamline
    interactions between the anndata object, as well as carrying out imputation
    and normalization
    """

    def __init__(
        self, model, make_dense: bool = False, balancer: Balancer = None
    ) -> None:
        self.model = model
        self._estimator_type: str = "classifier"
        self._is_fitted: bool = False
        self.missing_features: Sequence = (
            None  # Requested features to subset by that weren't found
        )
        self.make_dense: bool = make_dense
        self.had_inf: bool = False
        self.var = None
        self.balancer: None | Balancer = balancer  # Address class imbalance ONLY during
        # fitting
        if "predict_proba" in dir(model):
            self.score_fn: str = "predict_proba"
        elif "decision_function" in dir(model):
            self.score_fn = "decision_function"
        else:
            self.score_fn = None

    def __sklearn_clone__(self):
        return PredBase(
            model=clone(self.model),
            make_dense=self.make_dense,
            balancer=self.balancer,
        )

    def get_model(self):
        if isinstance(self.model, XGBEstimator):
            return self.model.model
        elif isinstance(self.model, AlrEstimator):
            return self.models
        elif isinstance(self, BatchBase):
            return self.i_model
        return self.model

    def _check_inf(self, X: np.ndarray) -> np.ndarray | sparse.csr_matrix:
        was_sparse = sparse.issparse(X)
        X = X.toarray() if was_sparse else X
        is_inf = np.isinf(X)
        count = is_inf.sum()
        if is_inf.any():
            print(f"Warning: X contains {count} inf values, converting to nan...")
            X[is_inf] = np.nan
        self.had_inf = True
        return sparse.csr_matrix(X) if was_sparse else X

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
        if self.balancer is not None:
            X = self.balancer.fit_transform(X, y=y)
        if y not in X.obs.columns:
            raise ValueError(f"The column '{y}' is not present in X.obs")
        self.var = X.var
        self._is_fitted = True
        y_vals = X.obs[y]
        if y_vals.dtype == "category":
            y_vals = y_vals.astype(str)
        self.model.fit(self._validate(X.X), y_vals)

    def _check_dense(self, X):
        if not self.make_dense or not sparse.issparse(X):
            return X
        elif self.make_dense and sparse.issparse(X):
            return X.toarray()

    def _validate(self, X):
        X = self._check_dense(X)
        X = self._check_inf(X)
        return X

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    def __sklearn_is_fitted__(self):
        return self._is_fitted

    def predict_proba(self, X: ad.AnnData) -> np.ndarray:
        return self.model.predict_proba(self._validate(X.X))

    def decision_function(self, X: ad.AnnData) -> np.ndarray:
        return self.model.decision_function(self._validate(X.X))

    def load(self, path: str) -> None:
        """Load the fitted estimator from the saved path"""
        self.model = pickle.load(path)

    def predict(self, X: ad.AnnData) -> np.ndarray:
        return self.model.predict(self._validate(X.X))

    @property
    def classes_(self):
        return self.model.classes_

    def rfecv(self, X: ad.AnnData, y="tumor_type", rfecv_params: dict = None) -> RFECV:
        """Perform recursive feature elimination with cross validation

        Parameters
        ----------
        X : anndata object to pass to RFECV.fit
        rfecv_params : parameters to pass to rfecv

        Returns
        -------
        Fitted RFECV object
        """
        encoder: LabelEncoder = LabelEncoder()
        labels = encoder.fit_transform(X.obs[y])
        params = rfecv_params if rfecv_params else {}
        rfecv = RFECV(self.get_model(), **params)
        rfecv.fit(X.X, labels)
        rfecv.ranking_ = X.var.index[rfecv.ranking_]
        return rfecv


class RandomForestPred(PredBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(model=RandomForestClassifier(random_state=RNG, **kwargs))


# class BatchBase(PredBase):
#     def __init__(self, model, outer=None) -> None:
#         super().__init__(model=BatchEstimator(model, outer), make_dense=True)

#     @override
#     def fit(self, X: ad.AnnData, y=["Sample_Type", "tumor_type"]) -> None:
#         vals = pd.DataFrame(self._validate(X.X))
#         y_vals = X.obs.loc[:, y].values
#         self.model.fit(vals, y_vals)


class AlrBase(PredBase):
    def __init__(
        self,
        model,
        references: dict | list,
        var_col: str = "GENEID",
        imputation: str = "none",
        weights=None,
        n_refs=-1,
    ) -> None:
        super().__init__(
            model=AlrEstimator(
                model=model,
                references=references,
                weights=weights,
                imputation=imputation,
                n_refs=n_refs,
            ),
            make_dense=True,
        )
        if var_col:
            self.var_col = var_col

    @override
    def fit(self, X: ad.AnnData, y="tumor_type") -> None:
        vals = self._validate(X.X)
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

    @override
    def decision_function(self, X: ad.AnnData) -> np.ndarray:
        df = adata_to_df(X, var_col=self.var_col)
        return self.model.decision_function(df)


class SimPred(PredBase):
    def __init__(self, model, method, **kwargs) -> None:
        if not kwargs:
            kwargs = {"n_instances": 15}
        super().__init__(model=SimEstimator(method, model=model, **kwargs))


class BatchBase(PredBase):
    """Ensemble estimator that uses an `outer` classifier to
    predict unknown categorical features of X. The `outer` predictions are added into
    X's features before being classified with the final `inner` estimator for the class
    labels

    WARNING: the inner model must have support for categorical features
    """

    def __init__(
        self,
        inner,
        outer_y: str,
        outer=None | PredBase,
        categorical_support: bool = False,
        make_dense: bool = False,
        balancer: Balancer = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=inner,
            make_dense=make_dense | categorical_support,
            balancer=balancer,
            **kwargs,
        )
        self.outer_y = outer_y
        self.o_model: PredBase = outer if outer is not None else PredBase(inner)
        self.encoder: OneHotEncoder | None = (
            OneHotEncoder(sparse_output=False) if not categorical_support else None
        )
        # When no native categorical support is available

    def _add_pred(self, X: ad.AnnData, fit: bool = True) -> np.ndarray | pd.DataFrame:
        if fit:
            self.o_model.fit(X, self.outer_y)
        predictions = self.o_model.predict(X)
        reshaped = pd.DataFrame(np.reshape(predictions, (-1, 1)))
        # Get outer model predictions

        counts = self._validate(X.X)
        if self.encoder is None:
            if isinstance(counts, np.ndarray):
                counts = pd.DataFrame(counts)
            combined = pd.concat([counts, reshaped], axis=1)
            final = list(combined.columns)[-1]
            combined = combined.astype({final: "category"})
        if fit:
            self.encoder.fit(reshaped)
        encoded = self.encoder.transform(reshaped)
        combined = np.concatenate([counts, encoded], axis=1)
        return combined

    @override
    def fit(self, X: ad.AnnData, y: str = "tumor_type") -> None:
        counts = self._add_pred(X, fit=True)
        y_vals = X.obs[y]
        if y_vals.dtype == "category":
            y_vals = y_vals.astype(str)
        self.model.fit(counts, y_vals)
        self._is_fitted = True

    @property
    def classes_(self):
        return self.i_model.classes_

    @override
    def predict(self, X: ad.AnnData, _=None) -> np.ndarray:
        counts = self._add_pred(X, fit=False)
        return self.model.predict(counts)

    @override
    def predict_proba(self, X, _=None) -> np.ndarray:
        counts = self._add_pred(X, fit=False)
        return self.i_model.predict_proba(counts)


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
        if "predict_proba" in dir(self.model):
            self._score_method = "predict_proba"
            self.predict_proba = partial(
                self._predict_score, score_method=self._score_method
            )
        elif "decision_function" in dir(self.model):
            self._score_method = "decision_function"
            self.decision_function = partial(
                self._predict_score, score_method=self._score_method
            )

    def _alr(self, X: pd.DataFrame, by: str, **kwargs) -> np.ndarray:
        X = self.impute_fn(X)
        result: np.ndarray = Transformer(
            "alr", inplace=False, by=by, **kwargs
        ).fit_transform(X)
        total = np.prod(result.shape)
        is_inf = np.isinf(result)
        is_nan = np.isnan(result)
        if is_inf.any():
            count = is_inf.sum()
            percent = round(count / total * 100)
            print(
                f"Warning: X contains {count} inf values ({percent}%), converting to nan..."
            )
            result[is_inf] = np.nan
        if is_nan.any():
            count = is_nan.sum()
            percent = round(count / total * 100)
            print(f"Warning: X contains {count} nan values ({percent}%)")
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
            zeros, X, is_here = self._check_ref(X, r)
            X = X.loc[~X[r].isna(), :]
            if is_here:
                if len(zeros) > 0:
                    print(f"WARNING: {len(zeros)} samples have reference {r} as zero!")
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

    def _check_ref(
        self, X: pd.DataFrame, ref: str
    ) -> tuple[np.ndarray | None, pd.DataFrame | None, bool]:
        """Ensure that all samples in X (sample x feature dataframe)
            have valid values for ALR reference feature `ref`

        Returns
        -------
        A tuple of
            the indices of the rows for which `ref` is zero
            X where values of `ref` with zero are replaced with nan
            Whether or not `ref` is in X at all

        Notes
        -----
        Ideally, imputation for `ref` should be handled before passing X to this model
        """
        if ref not in X.columns:
            return None, None, False
        bmask = X[ref] == 0
        X.loc[bmask, ref] = np.nan
        zeros: np.ndarray = np.where(bmask)[0]
        return (zeros, X, True)

    def predict(self, X) -> np.ndarray:
        score_df = pd.DataFrame(
            self._predict_score(X, self._score_method), columns=self.classes_
        )
        return np.array(score_df.idxmax(1))

    def _predict_score(self, X: pd.DataFrame, score_method: str) -> np.ndarray:
        """Get predictions using all trained estimators for each reference

        Parameters
        ----------
        X : a dataframe where columns are features
        """
        scores = []
        self.n_pred = 0
        self.missing_references = []
        for r, m in self.models.items():
            zeros, X, is_here = self._check_ref(X, r)
            if is_here:
                if len(zeros) > 0:
                    print(f"WARNING: {len(zeros)} samples have reference {r} as zero!")
                transformed = self._alr(X, r)
                if score_method == "predict_proba":
                    cur_score = m.predict_proba(transformed)
                elif score_method == "decision_function":
                    cur_score = m.decision_function(transformed)
                else:
                    raise ValueError(f"Score method {score_method} not recognized!")
                cur_score[zeros] = np.nan
                scores.append(cur_score)
                self.n_pred += 1
            else:
                # Don't try to normalize by it if it isn't present
                print(f"WARNING: reference {r} missing")
                self.missing_references.append(r)
                scores.append(np.zeros((len(self.classes_), len(X.shape[0]))))

        message = f"Predicted with {self.n_pred} ({self.n_pred // len(self.cur_refs) * 100}) references"
        print(message)
        scores = np.array(scores)

        if len(self.cur_weights) == scores.shape[0]:
            scores = np.reshape(self.cur_weights, [scores.shape[0], 1, 1]) * scores
        return scores.mean(axis=0)

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
        if "predict_proba" in dir(self.model):
            self._score_method = "predict_proba"
            self.predict_proba = partial(
                self._predict_score, score_method=self._score_method
            )
        elif "decision_function" in dir(self.model):
            self._score_method = "decision_function"
            self.decision_function = partial(
                self._predict_score, score_method=self._score_method
            )

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

    def _predict_score(self, X, score_method: str) -> np.ndarray:
        if score_method == "predict_proba":
            score_fn = self.model.predict_proba
        elif score_method == "decision_function":
            score_fn = self.model.decision_function
        else:
            raise ValueError(f"Score method {score_method} not recognized!")
        X = self._simulate(X)
        self._validate_x(X)
        all_proba = []
        for i in range(X.shape[0]):
            cur = score_fn(X[i, :, :])
            all_proba.append(cur)
        return np.array(all_proba).mean(axis=0)

    def predict(self, X) -> np.ndarray:
        scores = self._predict_score(X, self._score_method)
        voted = [self.model.classes_[c] for c in np.argmax(scores, axis=1)]
        return np.array(voted)

    @property
    def classes_(self):
        return self.model.classes_


class XGBEstimator:
    """Convenience wraper for XGBClassifier that supports string labels
    Early stopping parameters are set to be consistent with sklearn's defaults
    """

    def __init__(self, early_stop=False, **kwargs) -> None:
        self.encoder: LabelEncoder | None = None
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

    def __sklearn_clone__(self):
        est = XGBEstimator(model=clone(self.model))
        return est

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

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


class PredWithCorrection(PredBase):
    def __init__(
        self,
        model: PredBase,
        corrector: Corrector,
        transformer: Transformer,
        how: Literal["fc_mean", "none"],
        give_direct: bool = True,  # Give the underlying model the corrected count
        # data directly, instead of the approximation (i.e. the `how` parameter)
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.give_direct: bool = give_direct
        self.corrector: Corrector = corrector
        self.transformer: Transformer = transformer
        self.genewise_params: np.ndarray
        self.how: str = how

    # NOTE: this doesn't work at all
    def _fc_mean_adjust(self, original: ad.AnnData) -> ad.AnnData:
        new = original.copy()
        adj = new.X / self.genewise_params
        new.X = adj.toarray() if sparse.issparse(adj) else adj
        return new

    def _transform(self, original: ad.AnnData) -> ad.AnnData:
        if self.how == "fc_mean":
            adj = self._fc_mean_adjust(original)
        elif self.how == "none":
            adj = original
        else:
            raise ValueError("Not implemented!")
        adj: ad.AnnData = self.transformer.fit_transform(adj)
        return adj

    @override
    def fit(self, X: ad.AnnData, y="tumor_type") -> None:
        corrected = self.corrector.fit_transform(X)
        if self.how == "fc_mean":
            self.genewise_params = np.mean(X.X / corrected.X, axis=0)
            self.genewise_params[np.isnan(self.genewise_params)] = 0
        if self.give_direct:
            corrected = self.transformer.fit_transform(corrected)
            self.model.fit(corrected, y)
        else:
            x = self._transform(X)
            self.model.fit(x, y)

    @override
    def predict(self, X: ad.AnnData) -> np.ndarray:
        x = self._transform(X)
        return self.model.predict(x)

    @override
    def predict_proba(self, X: ad.AnnData) -> np.ndarray:
        x = self._transform(X)
        return self.model.predict_proba(x)


# * Pipeline


class Pipeline:
    def __init__(self, steps: Sequence, predictor: PredBase | None = None) -> None:
        self.preprocessing: Sequence = [s for s in steps if s is not None]
        self.predictor: PredBase | None = predictor
        self.dct: dict = {"Predictor": self.predictor}
        repeat_counter = {}
        self.dct = {}
        for step in self.preprocessing:
            cls_name = type(step).__name__
            count = repeat_counter.get(cls_name, -1) + 1
            repeat_counter[cls_name] = count
            if count > 0:
                self.dct[f"{cls_name}.{count}"] = step
            else:
                self.dct[cls_name] = step

    def fit(self, x: ad.AnnData, y: str = "tumor_type") -> None:
        for step in self.preprocessing:
            x = step.fit_transform(x)
        if self.predictor is not None:
            self.predictor.fit(x, y)

    @override
    def __repr__(self) -> str:
        return repr(self.dct)

    @property
    def score_fn(self) -> str:
        return self.predictor.score_fn

    @property
    def had_inf(self) -> bool:
        return self.predictor.had_inf

    @property
    def classes_(self):
        return self.predictor.classes_

    def transform(self, x: ad.AnnData) -> ad.AnnData:
        for step in self.preprocessing:
            x = step.transform(x)
        return x

    def fit_transform(self, x: ad.AnnData) -> ad.AnnData:
        for step in self.preprocessing:
            x = step.fit_transform(x)
        return x

    def fit_predict(self, x: ad.AnnData, y: str = "tumor_type") -> np.ndarray:
        self.fit(x, y)
        return self.predict(x)

    def predict(self, x) -> np.ndarray:
        if self.predictor is None:
            raise ValueError("predictor object not passed during init!")
        for step in self.preprocessing:
            x = step.transform(x)
        return self.predictor.predict(x)

    def predict_proba(self, x) -> np.ndarray:
        for step in self.preprocessing:
            x = step.transform(x)
        return self.predictor.predict_proba(x)
