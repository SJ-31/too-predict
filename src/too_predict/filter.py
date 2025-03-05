#!/usr/bin/env ipython

import anndata as ad
import numpy as np
import scanpy as sc


class Filter:
    def __init__(self, features=None, min_cells=2, feature_col="GENENAME") -> None:
        self.min_cells = min_cells
        self.features = features  # Requested features to subset data by
        self.feature_col = feature_col
        self.discarded_features = None  # Any features discarded during preprocessing e.g.  # due to not being in enough samples

        self.missing_features = []

    def fit(self, adata: ad.AnnData) -> None:
        self.adata = adata

    def fit_transform(self, adata: ad.AnnData) -> ad.AnnData:
        self.fit(adata)
        return self.transform()

    def transform(self) -> ad.AnnData:
        passed_filter: np.ndarray = sc.pp.filter_genes(
            self.adata, min_cells=self.min_cells, inplace=False
        )  # Genes must be nonzero in at least two samples
        self.discarded_features = self.adata.var.loc[~(passed_filter[0]), :]
        self.adata = self.adata[:, passed_filter[0]].copy()
        if self.features is not None:
            self.adata = self.adata[
                :, self.adata.obs[self.feature_col] == self.features
            ]
            missing = set(self.features) - set(self.adata.obs[self.feature_col])
            self.missing_features = missing
            if len(missing) > 0:
                print("--- WARNING: Missing features!")
                print(missing)
                print("---")
        return self.adata
