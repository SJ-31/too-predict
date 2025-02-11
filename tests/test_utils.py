#!/usr/bin/env ipython

import importlib.resources as res
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scanpy as sc
import skbio.stats.composition as comp
import too_predict
from pyhere import here
from rpy2.rinterface import SexpVector
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr
from scipy import sparse
from too_predict.utils import (
    add_gene_metadata,
    df_from_r,
    dgelist2anndata,
    get_data,
    r_cleanup,
)

base = importr("base")
ensembldb = importr("ensembldb")
datadir = here("data", "tests")
hcc = here(datadir, "tcga_hcc.rds")

adata = dgelist2anndata(str(hcc))

hg38 = here("data", "Homo_sapiens.GRCh38.113.sqlite")

adata: ad.AnnData = dgelist2anndata(str(hcc))
adata.var.index = adata.var.index.to_series().str.replace("\\..*", "", regex=True)
add_gene_metadata(adata)
adata.var = adata.var.drop("gene_id", axis="columns")
# #  --- CODE BLOCK ---


class Model:
    def __init__(self, adata: ad.AnnData) -> None:
        self.ad = adata
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

    # def tts() -> test, train


P = Model(adata)
# P.normalize("alr", by="BRSK2", var_col="GENENAME")
P.normalize("clr")


# #  --- CODE BLOCK ---
