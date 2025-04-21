#!/usr/bin/env ipython

import anndata as ad
import numpy as np
import rpy2.robjects as ro
from inmoose import pycombat
from rpy2.robjects import default_converter, numpy2ri
from scipy import sparse

import too_predict.utils as ut

IMPLEMENTED_CORRECTION = {
    "pycombat_seq",
}


class Corrector:
    """
    Class for applying corrections to count data i.e. batch effect correction
    Unlike `Transformer`, methods here must preserve integer property
    [2025-04-21 Mon] Try use this to remove noise between sample types
    """

    def __init__(
        self,
        method: str,
        inplace=False,
        layer=None,
        **kwargs,
    ):
        if self.method not in IMPLEMENTED_CORRECTION:
            raise ValueError("Method not supported!")
        self.method = method
        self.inplace = inplace
        self.layer = layer
        self.kwargs: dict = kwargs
        self.counts: np.ndarray

    def fit(self, adata: ad.AnnData, batch: str | np.ndarray | list[str]) -> None:
        self.adata = adata.copy() if not self.inplace else adata
        if isinstance(batch, str):
            self.batch = list(self.adata.obs[batch].astype(str))
        else:
            self.batch = batch
        self.counts = adata.X if self.layer is None else adata.layers[self.layer]
        self.was_sparse = sparse.issparse(self.counts)
        self.counts = self.counts.toarray() if self.was_sparse else self.counts

    def _get_obs_col(self, col: str | list[str] | np.ndarray) -> list[str]:
        if isinstance(col, str):
            return list(self.adata.obs[col].astype(str))
        elif isinstance(col, np.ndarray):
            return list(col)
        return col

    def _pycombat_seq(
        self, covar_mod: list[str] | str | None = None, **kwargs
    ) -> np.ndarray:
        if covar_mod is not None:
            covar_mod = self._get_obs_col(covar_mod)
        pyc = pycombat.pycombat_seq(
            np.transpose(self.counts), batch=self.batch, covar_mod=covar_mod, **kwargs
        )
        return np.transpose(pyc)

    @ut.r_cleanup
    def _remove_batch_effect(
        self,
        batch2: list[str] | str | None = None,
        group: list[str] | str | None = None,
    ) -> np.ndarray:
        ro.r("library(limma)")
        logged = np.log1p(self.counts)
        ro.globalenv["batch"] = ro.StrVector(self.batch)
        if group is not None:
            ro.globalenv["group"] = ro.StrVector(self._get_obs_col(group))
        else:
            ro.r("group <- NULL")
        if batch2 is not None:
            ro.globalenv["batch2"] = ro.StrVector(self._get_obs_col(batch2))
        else:
            ro.r("batch2 <- NULL")
        ut.np_to_r(logged, r_symbol="counts")
        ro.r("""
        corrected <- removeBatchEffect(counts, batch = batch, batch2 = batch2, group = group)
        """)
        corrected = np.expm1(ut.np_from_r(ro.globalenv["corrected"]))
        corrected[corrected < 0] = 0
        corrected = corrected.astype(np.int64)
        return corrected

    def transform(self, _=None) -> ad.AnnData | None:
        corrected: np.ndarray
        match self.method:
            case "pycombat_seq":
                corrected = self._pycombat_seq(**self.kwargs)
            case "removeBatchEffect":
                corrected = self._remove_batch_effect(**self.kwargs)
            case _:
                raise ValueError
        self.adata.X = corrected if not self.was_sparse else sparse.csr_array(corrected)
        return self.adata if not self.inplace else None

    def fit_transform(
        self, data: ad.AnnData, batch: str
    ) -> ad.AnnData | None | np.ndarray:
        self.fit(data, batch=batch)
        return self.transform()
