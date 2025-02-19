#!/usr/bin/env ipython
import gc
import pickle
from typing import Callable, override

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.model_selection as ms
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

from too_predict.evaluation import get_all_metrics
from too_predict.imputer import Imputer
from too_predict.normalizer import Normalizer
from too_predict.simulation import Simulator

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
        normalization: str,
        imputation: str,
        model,
        features=None,
        feature_col="GENENAME",
        n_kwargs: dict | None = None,
    ) -> None:
        self.model = model
        if normalization:
            self.n_method = normalization.lower()
        else:
            self.n_method = None
        self.normalize_kwargs: dict = n_kwargs if n_kwargs else {}
        self.features = features
        self.feature_col = feature_col
        self.impute: Callable = Imputer(imputation).run
        self.i_method: str | None = (
            imputation.lower() if isinstance(imputation, str) else None
        )
        self.normalize_kwargs = {}

    def fit(self, X: ad.AnnData, label_col="tumor_type", preprocess=True) -> None:
        if preprocess:
            X = self._preprocess(X)
        self.model.fit(X.X, X.obs[label_col])

    def predict_proba(self, X: ad.AnnData, preprocess=True) -> np.ndarray:
        if preprocess:
            X = self._preprocess(X)
        return self.model.predict_proba(X.X)

    def load(self, path: str) -> None:
        """Load the fitted estimator from the saved path"""
        self.model = pickle.load(path)

    def predict(self, X: ad.AnnData, preprocess: bool = True) -> np.ndarray:
        if preprocess:
            X = self._preprocess(X)
        return self.model.predict(X)

    def _filter_features(self, adata: ad.AnnData) -> ad.AnnData:
        passed_filter = sc.pp.filter_genes(
            adata, min_cells=2, inplace=False
        )  # Genes must be nonzero in at least two samples
        adata = adata[:, passed_filter[0]].copy()
        if self.features is not None:
            adata = adata[:, adata.obs[self.feature_col] == self.features]
            missing = set(self.features) - set(adata.obs[self.feature_col])
            if len(missing) > 0:
                print("--- WARNING: Missing features!")
                print(missing)
                print("---")
        return adata

    def _preprocess(self, adata: ad.AnnData) -> ad.AnnData:
        adata = self._filter_features(adata)
        if self.n_method is not None:
            normalized: ad.AnnData = Normalizer(
                adata, self.n_method, impute_fn=self.impute, inplace=False
            ).run(**self.normalize_kwargs)
            gc.collect()
            return normalized
        return adata

    def _classes(self):
        return self.model.classes_

    def cross_validate(
        self, adata, cv=None, label_col="tumor_type", preprocess=True
    ) -> dict:
        """Evaluate model performance with cross-validation"""
        if not cv:
            cv = ms.StratifiedKFold(n_splits=5)
        if preprocess:
            N = self._preprocess(adata)
        else:
            N = adata.copy()
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


class RandomForestPred(PredBase):
    def __init__(
        self, normalization: str, imputation: str, features=None, feature_col="GENENAME"
    ) -> None:
        super().__init__(
            normalization=normalization,
            imputation=imputation,
            model=RandomForestClassifier(),
            features=features,
            feature_col=feature_col,
        )


class AlrBase(PredBase):
    """Base class for aggregating the results of classifier models trained on
    ALR-transformed with multiple references e.g. different genes

    Predicted labels are obtained with soft voting (weighted average probabilities)
    """

    def __init__(
        self,
        imputation: str,
        model,
        references: dict | list,
        features=None,
        feature_col="GENENAME",
        weights=None,
    ) -> None:
        super().__init__(
            normalization=None,
            imputation=imputation,
            model=None,
            features=features,
            feature_col=feature_col,
        )
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

    def _alr(self, X: ad.AnnData, by: str, vc) -> ad.AnnData:
        res: ad.AnnData = Normalizer(X, "alr", self.impute, inplace=False).run(
            by=by, var_col=vc
        )
        return res

    @override
    def fit(
        self, X: ad.AnnData, label_col="tumor_type", preprocess=True, var_col="GENENAME"
    ) -> None:
        self.missing_references = []
        self.n_fit = 0
        if preprocess:
            X = self._preprocess(X)
        for r in self.refs:
            if r in X.var[var_col]:
                transformed = self._alr(X, r, var_col)
                self.models[r].fit(transformed.X, transformed.obs[label_col])
                self.n_fit += 1
            else:
                self.missing_references.append(r)
                print(f"WARNING: reference {r} missing")
        print(
            f"Fit with {self.n_fit} ({self.n_fit // len(self.refs) * 100}) references"
        )

    @override
    def predict(self, X: ad.AnnData, preprocess: bool = True) -> np.ndarray:
        proba_df = pd.DataFrame(
            self.predict_proba(X, preprocess), columns=self._classes()
        )
        return np.array(proba_df.idxmax(1))

    @override
    def predict_proba(
        self, X: ad.AnnData, preprocess=True, var_col="GENENAME"
    ) -> np.ndarray:
        if preprocess:
            X = self._preprocess(X)
        proba = []
        self.n_pred = 0
        self.missing_references = []
        for r, m in self.models.items():
            if r in X.var[var_col]:
                transformed = self._alr(X, r, var_col)
                proba.append(m.predict_proba(transformed.X))
                self.n_pred += 1
            else:
                # Don't try to normalize by it if it isn't present
                print(f"WARNING: reference {r} missing")
                self.missing_references.append(r)
                proba.append(np.zeros((len(self._classes()), len(X.X.shape[0]))))

        message = f"Predicted with {self.n_pred} ({self.n_pred // len(self.refs) * 100}) references"
        print(message)
        proba = np.array(proba)

        if self.weights is not None and len(self.weights) == proba.shape[0]:
            proba = np.reshape(self.weights, [proba.shape[0], 1, 1]) * proba
        return proba.mean(axis=0)

    def _classes(self):
        return self.models[self.refs[0]].classes_


class SimBase(PredBase):
    def __init__(
        self,
        normalization: str,
        imputation: str,
        simulation: str,
        model,
        features=None,
        prefix: str = "mc_",
        n: int = 10,
        feature_col="GENENAME",
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
        super().__init__(normalization, imputation, model, features, feature_col)
        # TODO <2025-02-17 Mon>: too few instances, need a better way
        self.s_method = simulation
        self.predict_from_sim = predict_from_sim
        self.prefix = prefix
        self.cross_validating = False
        self.n = n

    def _get_instances(self, X: ad.AnnData, label_col: str = None):
        X = self._filter_features(X)
        X = Simulator(
            X,
            self.s_method,
            self.n,
            prefix=self.prefix,
            inplace=False,
            make_sparse=False,  # Required for concatenation
        ).run()
        instances = list(filter(lambda x: x.startswith(self.prefix), X.layers.keys()))
        counts = np.concatenate([X.layers[inst] for inst in instances])
        labels = np.concatenate(
            [np.copy(X.obs[label_col]) for _ in range(len(instances))]
        )
        with_sim = ad.AnnData(X=counts, var=X.var)
        if label_col:
            with_sim.obs = pd.DataFrame({label_col: labels})
        return with_sim

    @override
    def fit(
        self,
        X: ad.AnnData,
        label_col="tumor_type",
        preprocess=True,
    ) -> None:
        """
        Fit the underlying model on combined data from the all the Monte Carlo instances

        @param mc_kwargs
            - n: number of Monte Carlo instances to generate
            - prefix: number
        @param instance_prefix: prefix denoting layers in the the adata object containing
        the instances
        """
        if not self.cross_validating:
            X = self._get_instances(X)
        return super().fit(X, label_col=label_col, preprocess=preprocess)

    @override
    def predict_proba(self, X: ad.AnnData, preprocess=True) -> np.ndarray:
        if self.predict_from_sim and not self.cross_validating:
            X = self._get_instances(X)
        return super().predict_proba(X, preprocess)

    @override
    def predict(self, X: ad.AnnData, preprocess=True) -> np.ndarray:
        if self.predict_from_sim and not self.cross_validating:
            X = self._get_instances(X)
        return super().predict(X, preprocess)

    @override
    def cross_validate(
        self, adata, cv=None, label_col="tumor_type", preprocess=True
    ) -> dict:
        self.cross_validating = True
        adata = self._get_instances(adata, label_col=label_col)
        result = super().cross_validate(adata, cv, label_col, preprocess)
        self.cross_validating = False
        return result
