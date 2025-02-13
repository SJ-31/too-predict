#!/usr/bin/env ipython
from abc import abstractmethod

import anndata as ad
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import skbio.stats.composition as comp
from rpy2.robjects import default_converter, numpy2ri
from scipy import sparse

from too_predict.normalizer import IMPLEMENTED_NORMALIZATION, Normalizer
from too_predict.utils import library, r_cleanup

IMPLEMENTATION_IMPUTATION = {"plus_one", "replace_one"}


class Model:
    def __init__(
        self, adata: ad.AnnData, normalization: str, imputation: str, features=None
    ) -> None:
        self.ad = adata.copy()  # Training data
        if imputation.lower() not in IMPLEMENTATION_IMPUTATION:
            raise ValueError(f"Imputation method {imputation} not implemented!")

        self.n_method = normalization.lower()
        self.i_method = imputation.lower()
        pass

    @abstractmethod
    def predict(self, data=None) -> pd.Series:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    def impute(self, mat):
        match self.i_method:
            case "plus_one":
                return mat + 1
            case "replace_one":
                copy = mat.copy()
                copy[copy == 0] = 1
                return copy

    def normalize(self) -> None:
        _ = Normalizer(self.ad, self.n_method, self.impute, inplace=True).run()
        self.ad.uns["normalized"] = True


# class kj
