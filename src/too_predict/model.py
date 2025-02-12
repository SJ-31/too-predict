#!/usr/bin/env ipython
from abc import abstractmethod

import anndata as ad
import numpy as np
import pandas as pd
import skbio.stats.composition as comp
from scipy import sparse


class Model:
    def __init__(
        self, adata: ad.AnnData, normalization: str, imputation: str, features=None
    ) -> None:
        self.ad = adata  # Training data
        self.n_method = normalization
        self.i_method = imputation
        pass

    @abstractmethod
    def predict(self, data=None) -> pd.Series:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    # def impute(self)

    def _alr(self, by: int | str, var_col: str = None) -> None:
        """Normalize counts in adata using ALR, with the counts of a specific
        gene `by` as the reference.

        :param: by name of gene to normalize by, or the index of the gene in adata.var
            if the name is provided, the index is looked up automatically.
        :param: var_col column in adata.var containing the gene name.
        """
        if isinstance(by, str) and var_col:
            query = np.where(self.ad.var[var_col] == by)
            by = query[0][0]
            if len(query) > 1:
                raise ValueError("Key `by` is not unique!")
        elif isinstance(by, str):
            by = self.ad.var.index.get_loc(by)
            if len(by) > 1:
                raise ValueError("Key `by` is not unique!")
        # <2025-02-11 Tue> TODO: figure out handling of zeros
        counts = self.ad.X.toarray()
        self.ad = ad.concat([self.ad[:, :by], self.ad[:, by + 1 :]], axis="var")
        self.ad.layers["counts"] = self.ad.X
        normalized = comp.alr(counts + 1, by)
        self.ad.X = sparse.csr_matrix(normalized)

    def _clr(self):
        normalized = comp.clr(self.ad.X.toarray() + 1)
        self.ad.layers["counts"] = self.ad.X
        self.ad.X = sparse.csc_matrix(normalized)

    def normalize(self, method: str, **kwargs) -> None:
        if method == "alr":
            self._alr(**kwargs)
        elif method == "clr":
            self._clr()
        self.ad.uns["normalized"] = True
