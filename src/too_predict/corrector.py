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
    "removeBatchEffect",
    "combat_ref",
    "combat_seq",
    "deseq2",
}
# Both combat_seq and combat_ref preserve integer count data

# [2025-04-21 Mon] Try to save the batch correction factor used in train and
# apply it to the test data
# this definitely constitutes information leakage


class Corrector:
    """
    Class for applying corrections to count data i.e. batch effect correction
    Unlike `Transformer`, methods here must preserve integer property
    [2025-04-21 Mon] Try use this to remove noise between sample types
    """

    def __init__(
        self,
        method: str,
        batch: str,
        inplace=False,
        layer=None,
        **kwargs,
    ):
        self.method = method
        if self.method not in IMPLEMENTED_CORRECTION:
            raise ValueError("Method not supported!")
        self.batch_key = batch
        self.inplace = inplace
        self.layer = layer
        self.kwargs: dict = kwargs
        self.counts: np.ndarray

    def fit(self, adata: ad.AnnData) -> None:
        self.adata = adata.copy() if not self.inplace else adata
        if isinstance(self.batch_key, str):
            self.batch = list(self.adata.obs[self.batch_key].astype(str))
        else:
            self.batch = self.batch_key
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
        """Batch-effect correction with pycombat_seq

        Parameters
        ----------
        covar_mod : sample-level experimental conditions to preserve.
            Basically what you want to compare in a DE analysis after the correction
        """
        if covar_mod is not None:
            covar_mod = self._get_obs_col(covar_mod)
        pyc = pycombat.pycombat_seq(
            np.transpose(self.counts), batch=self.batch, covar_mod=covar_mod, **kwargs
        )
        return np.transpose(pyc)

    @ut.r_cleanup
    def _combat_seq(
        self,
        group: list[str] | str | None = None,
        covar_mod: list[str] | str | None = None,
        full_mod: bool = True,
        shrink: bool = False,
        shrink_disp: bool = False,
        gene_subset_n: int | None = None,
    ) -> np.ndarray:
        ut.np_to_r(np.transpose(self.counts), "counts")
        ro.globalenv["batch"] = ro.StrVector(self.batch)
        ro.globalenv["full_mod"] = full_mod
        ro.globalenv["shrink"] = shrink
        ro.globalenv["shrink_disp"] = shrink_disp
        if gene_subset_n is not None:
            ro.globalenv["gene_subset_n"] = ro.IntVector(gene_subset_n)
        else:
            ro.r("gene_subset_n <- NULL")
        if group is not None:
            ro.globalenv["group"] = ro.StrVector(self._get_obs_col(group))
        else:
            ro.r("group <- NULL")
        if covar_mod is not None:
            ro.globalenv["covar_mod"] = ro.StrVector(self._get_obs_col(covar_mod))
        else:
            ro.r("covar_mod <- NULL")
        ro.r("""
        corrected <- sva::ComBat_seq(
            counts, batch, group = group, covar_mod = covar_mod,
            full_mod = full_mod, shrink = shrink, shrink.disp = shrink_disp,
            gene.subset.n = gene_subset_n
        )
        """)
        corrected = np.transpose(ut.np_from_r(ro.globalenv["corrected"]))
        return corrected

    @ut.r_cleanup
    def _deseq2(
        self, group: list[str] | str | None = None, full: bool = True
    ) -> np.ndarray:
        ut.source("correction.R", in_r=True)
        ut.counts_into_r(self.adata, counts=self.counts)
        ro.globalenv["batch"] = ro.StrVector(self.batch)
        ro.globalenv["full_mod"] = full
        if group is not None:
            ro.globalenv["group"] = ro.StrVector(self._get_obs_col(group))
        else:
            ro.r("group <- NULL")
        ro.r("""
        corrected <- deseq2_batch(counts, batch, group = group, full_mod = full_mod)
        """)
        corrected = np.transpose(ut.np_from_r(ro.globalenv["corrected"]))
        return corrected

    @ut.r_cleanup
    def _combat_ref(
        self,
        group: list[str] | str | None = None,
        covar_mod: list[str] | str | None = None,
        full: bool = True,
        genewise_disp: bool = False,
    ) -> np.ndarray:
        """Batch-effect correction with pycombat_seq

        Parameters
        ----------
        covar_mod : sample-level experimental conditions to preserve.
            Basically what you want to compare in a DE analysis after the correction
        """
        ut.source("combat_ref.R", in_r=True)
        ut.np_to_r(np.transpose(self.counts), "counts")
        ro.globalenv["batch"] = ro.StrVector(self.batch)
        ro.globalenv["full_mod"] = full
        ro.globalenv["genewise_disp"] = genewise_disp
        if group is not None:
            ro.globalenv["group"] = ro.StrVector(self._get_obs_col(group))
        else:
            ro.r("group <- NULL")
        if covar_mod is not None:
            ro.globalenv["covar_mod"] = ro.StrVector(self._get_obs_col(covar_mod))
        else:
            ro.r("covar_mod <- NULL")
        ro.r("""
        corrected <- ComBat_ref(counts, batch, group = group, covar_mod = covar_mod,
            full_mod = full_mod, genewise.disp = genewise_disp)
        """)
        corrected = np.transpose(ut.np_from_r(ro.globalenv["corrected"]))
        return corrected

    @ut.r_cleanup
    def _remove_batch_effect(
        self,
        batch2: list[str] | str | None = None,
        group: list[str] | str | None = None,
    ) -> np.ndarray:
        """Model batch and experimental effects with linear model, then subtracts
        former from expression data

        See limma::removeBatchEffect

        Parameters
        ----------
        batch2 : second series of batch effects that are independent of first
        group : sample-level experimental conditions to preserve.
            Basically what you want to compare in a DE analysis after the correction
        """
        message: str = "Applying correction: removeBatchEffect"
        ro.r("library(limma)")
        logged = np.transpose(np.log1p(self.counts))
        ro.globalenv["batch"] = ro.StrVector(self.batch)
        if batch2 is not None:
            message = f"{message}\n\tbatch2: {batch2}"
            ro.globalenv["batch2"] = ro.StrVector(self._get_obs_col(batch2))
        else:
            ro.r("batch2 <- NULL")
        if group is not None:
            message = f"{message}\n\tgroup: {group}"
            ro.globalenv["group"] = ro.StrVector(self._get_obs_col(group))
        else:
            ro.r("group <- NULL")
        ut.np_to_r(logged, r_symbol="counts")
        print(message)
        ro.r("""
        corrected <- removeBatchEffect(counts, batch = batch, batch2 = batch2, group = group)
        """)
        corrected = ut.np_from_r(ro.globalenv["corrected"])
        self.subtracted_effect = np.transpose(logged - corrected)
        corrected = np.expm1(corrected)
        corrected[corrected < 0] = 0
        corrected = corrected.astype(np.int64)
        return np.transpose(corrected)

    def transform(self, _=None) -> ad.AnnData | None:
        corrected: np.ndarray
        match self.method:
            case "pycombat_seq":
                corrected = self._pycombat_seq(**self.kwargs)
                # [2025-04-21 Mon] this is really slow
            case "combat_ref":
                corrected = self._combat_ref(**self.kwargs)
            case "combat_seq":
                corrected = self._combat_seq(**self.kwargs)
            case "deseq2":
                corrected = self._deseq2(**self.kwargs)
            case "removeBatchEffect":
                corrected = self._remove_batch_effect(**self.kwargs)
            case _:
                raise ValueError
        self.adata.X = corrected if not self.was_sparse else sparse.csr_array(corrected)
        return self.adata if not self.inplace else None

    def fit_transform(self, data: ad.AnnData) -> ad.AnnData | None | np.ndarray:
        self.fit(data)
        return self.transform()
