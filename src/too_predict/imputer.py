#!/usr/bin/env ipython
import numpy as np
import pandas as pd
from skbio.stats.composition import multi_replace
from sklearn.ensemble import RandomForestRegressor

"""
References
[1] https://bojarlab.github.io/glycowork/
"""

IMPLEMENTED_IMPUTATION = {
    "plus_one",
    "replace_one",
    "multi_replace",
    "none",
    # "missforest", # <2025-02-25 Tue> Was way too slow
    None,
    "labelled_median",
}


class Imputer:
    def __init__(self, method: str, **kwargs) -> None:
        if method and method.lower() not in IMPLEMENTED_IMPUTATION:
            raise ValueError(f"Imputation method {method} not implemented!")
        self.method: str | None = method
        self.kwargs: dict = kwargs

    @staticmethod
    def replace_one(mat: np.ndarray) -> np.ndarray:
        copy = mat.copy()
        copy[copy == 0] = 1
        return copy

    @staticmethod
    def labelled_median(mat: np.ndarray, labels) -> np.ndarray:
        """Imputation for training datasets

        Imputes zero variables in a given sample as the median of that variable computed
        from samples with the same label
        """
        if mat.shape[0] != len(labels):
            raise ValueError(
                "The number of labels is not the same as the number of samples!"
            )
        replaced = mat.copy()
        has_zeros = (replaced == 0).any(axis=0).astype(np.int16)
        for label in set(labels):
            locs = labels == label
            current = mat[locs, :]
            median = np.nanmedian(current, axis=0)
            replaced[locs, :] = current + median * has_zeros
        return replaced

    def __call__(self, counts: np.ndarray) -> np.ndarray:
        match self.method:
            case "plus_one":
                return counts + 1
            case "replace_one":
                return self.replace_one(counts)
            case "multi_replace":
                return multi_replace(counts)
            case "missforest":
                return MissForest().fit_transform(counts)
            case "labelled_median":
                return self.labelled_median(counts, **self.kwargs)
            case None | "none":
                return counts


class MissForest:
    "Code taken from [1]"

    def __init__(
        self,
        regressor: RandomForestRegressor = RandomForestRegressor(
            n_jobs=-1
        ),  # estimator object for each imputation
        max_iter: int = 5,  # number of iterations for imputation process
        tol: float = 1e-5,  # convergence tolerance
    ) -> None:
        "A class to perform MissForest imputation adapted from https://github.com/yuenshingyan/MissForest"
        self.regressor = regressor
        self.max_iter = max_iter
        self.tol = tol

    def fit_transform(
        self,
        X: np.ndarray,  # input dataframe with missing values
    ) -> np.ndarray:  # imputed dataframe
        "Replace missing values using the MissForest algorithm"
        # Step 1: Initialization
        # Keep track of where NaNs are in the original dataset
        X = pd.DataFrame(X).replace(0, None)
        X_nan = X.isnull()
        # Replace NaNs with median of the column in a new dataset that will be transformed
        X_transform = X.fillna(X.median())
        # Sort columns by the number of NaNs (ascending)
        sorted_columns = X_nan.sum().sort_values().index
        for _ in range(self.max_iter):
            total_change = 0
            # Step 2: Imputation
            for column in sorted_columns:
                missing_idx: pd.Series = X_nan[column]
                if missing_idx.any():
                    # if column has missing values in original dataset
                    # Split data into observed and missing for the current column
                    observed = X_transform.loc[~missing_idx]
                    missing = X_transform.loc[missing_idx]
                    features = observed.drop(columns=column)
                    if features.notna().any().any():
                        # Use other columns to predict the current column
                        self.regressor.fit(
                            observed.drop(columns=column), observed[column]
                        )
                        y_missing_pred = self.regressor.predict(
                            missing.drop(columns=column)
                        )
                        # Replace missing values in the current column with predictions
                        total_change += np.sum(
                            np.abs(
                                X_transform.loc[missing_idx, column] - y_missing_pred
                            )
                        )
                        X_transform.loc[missing_idx, column] = y_missing_pred
            # Check for convergence
            if total_change < self.tol:
                break  # Break out of the loop if converged
        # Avoiding zeros
        X_transform += 1e-6
        return X_transform.values
