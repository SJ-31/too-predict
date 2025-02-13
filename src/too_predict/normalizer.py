#!/usr/bin/env ipython
from typing import Callable

import anndata as ad
import numpy as np
import rpy2.robjects as ro
import skbio.stats.composition as comp
from rpy2.robjects import default_converter, numpy2ri
from scipy import sparse

from too_predict.utils import library, r_cleanup

IMPLEMENTED_NORMALIZATION = {"clr", "tmm"}


class Normalizer:
    """Class for normalizing counts in the given adata object. Inplace by default"""

    def __init__(
        self, adata: ad.AnnData, method: str, impute_fn: Callable, inplace=True
    ) -> None:
        self.ad = adata if inplace else adata.copy()
        if method.lower() not in IMPLEMENTED_NORMALIZATION:
            raise ValueError(f"Normalization method {method} not implemented!")
        self.method = method
        self.ad.layers["counts"] = adata.X
        self.counts = impute_fn(adata.X.copy())

    def alr(self, by: int | str, var_col: str = None) -> None:
        """Normalize counts in adata using ALR, with the counts of a specific
        gene `by` as the reference.

        :param: by name of gene to normalize by, or the index of the gene in adata.var
            if the name is provided, the index is looked up automatically.
        :param: var_col column in adata.var containing the gene name.
        """
        index: int
        if isinstance(by, str) and var_col:
            query = np.where(self.ad.var[var_col] == by)
            index = query[0][0]
            if len(query) > 1:
                raise ValueError("Key `by` is not unique!")
        elif isinstance(by, str):
            index = self.ad.var.index.get_loc(by)
            if len(by) > 1:
                raise ValueError("Key `by` is not unique!")
        else:
            index = by
        self.ad = ad.concat([self.ad[:, :index], self.ad[:, index + 1 :]], axis="var")
        normalized = comp.alr(self.counts.toarray(), by)
        self.ad.X = sparse.csr_matrix(normalized)

    @r_cleanup
    def tmm(self) -> None:
        np_cv_rules = default_converter + numpy2ri.converter
        with np_cv_rules.context():
            ro.globalenv["mat"] = np.transpose(self.counts.toarray())
        ro.r("dge <- edgeR::DGEList(mat)")
        ro.r("edgeR::normLibSizes(dge)")
        ro.r("counts <- edgeR::cpm(dge, log = TRUE)")
        normalized = np.transpose(np.asarray(ro.r("counts")))
        self.ad.X = sparse.csc_matrix(normalized)

    def clr(self) -> None:
        normalized = comp.clr(self.counts.toarray())
        self.ad.X = sparse.csc_matrix(normalized)

    def run(self) -> ad.AnnData:
        match self.method:
            case "clr":
                self.clr()
            case "tmm":
                self.tmm()
        return self.ad
