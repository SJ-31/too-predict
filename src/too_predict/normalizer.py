#!/usr/bin/env ipython
from typing import Callable

import anndata as ad
import numpy as np
import rpy2.robjects as ro
import skbio.stats.composition as comp
from rpy2.robjects import default_converter, numpy2ri
from scipy import sparse, stats

from too_predict.utils import library, r_cleanup

IMPLEMENTED_NORMALIZATION = {"clr", "tmm", "alr", "dirichlet_mc"}


class Normalizer:
    """Class for normalizing counts in the given adata object. Inplace by default
    Count data are temporarily converted to a numpy array for normalization if necessary

    """

    def __init__(
        self,
        adata: ad.AnnData,
        method: str,
        impute_fn: Callable,
        inplace=True,
        make_sparse=True,
    ) -> None:
        self.ad = adata if inplace else adata.copy()
        if method.lower() not in IMPLEMENTED_NORMALIZATION:
            raise ValueError(f"Normalization method {method} not implemented!")
        self.inplace = inplace
        self.method = method
        self.make_sparse = make_sparse
        self.ad.layers["counts"] = adata.X
        if not (isinstance(adata.X, np.ndarray)):
            self.counts = impute_fn(adata.X.toarray().copy())
        else:
            self.counts = impute_fn(adata.X.copy())

    def alr(self, by: int | str, var_col: str = None) -> np.ndarray:
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
        self.ad = ad.concat(
            [self.ad[:, :index], self.ad[:, index + 1 :]],
            axis="var",
            merge="same",
            uns_merge="same",
        )
        return comp.alr(self.counts, index)

    def dirichlet_mc(self, n: int, prior: float = 0.5, prefix="mc_"):
        """
        Obtain random dirichlet instances from read counts (based on the ALDEx2 R package)
        For each sample, use the vector of read counts as the concentration parameter

        TODO: how to define the prior?
        TODO: this isn't very performant, look into Rust (rand_distr) or parallel

        Parameters:
        -----------
        prefix: prefix for the layer in the resulting adata object containing each instance
        """
        arr = self.counts + prior
        instances = np.apply_along_axis(lambda x: stats.dirichlet.rvs(x, n), 1, arr)
        for i in range(instances.shape[1]):
            inst = instances[:, i, :]
            self.ad.layers[f"{prefix}{i}"] = (
                inst if not self.make_sparse else sparse.csr_matrix(inst)
            )

    @r_cleanup
    def tmm(self) -> np.ndarray:
        np_cv_rules = default_converter + numpy2ri.converter
        with np_cv_rules.context():
            ro.globalenv["mat"] = np.transpose(self.counts)
        ro.r("dge <- edgeR::DGEList(mat)")
        ro.r("edgeR::normLibSizes(dge)")
        ro.r("counts <- edgeR::cpm(dge, log = TRUE)")
        return np.transpose(np.asarray(ro.r("counts")))

    def clr(self) -> np.ndarray:
        return comp.clr(self.counts)

    def run(self, **kwargs) -> ad.AnnData | None:
        match self.method:
            case "clr":
                normalized = self.clr()
            case "tmm":
                normalized = self.tmm()
            case "alr":
                normalized = self.alr(**kwargs)
            case "dirichlet_mc":
                self.dirichlet_mc(**kwargs)
                if not self.inplace:
                    return self.ad
                return
            case _:
                normalized = np.array()
        self.ad.X = sparse.csc_matrix(normalized) if self.make_sparse else normalized
        if not self.inplace:
            return self.ad
