#!/usr/bin/env ipython
from typing import Callable

import anndata as ad
import numpy as np
import rpy2.robjects as ro
import skbio.stats.composition as comp
from rpy2.robjects import default_converter, numpy2ri
from scipy import sparse, stats

from too_predict.utils import library, r_cleanup

IMPLEMENTED_NORMALIZATION = {"clr", "tmm", "alr"}

"""
References
[1] Pachter, L. (2011). Models for transcript quantification from RNA-Seq. ArXiv. https://arxiv.org/abs/1104.3889
[2] Bennett, A. R., Lundstrøm, J., Chatterjee, S., & Bojar, D. (2025). Compositional data analysis enables statistical rigor in comparative glycomics. Nature Communications, 16(1), 1-15. https://doi.org/10.1038/s41467-025-56249-3
[3] Godichon-Baggioni, A., Maugis-Rabusseau, C., & Rau, A. (2018). Clustering transformed compositional data using K-means, with applications in gene expression and bicycle sharing system data. Journal of Applied Statistics, 46(1), 47–65. https://doi.org/10.1080/02664763.2018.1454894
"""


class Normalizer:
    """Class for normalizing counts in the given adata object. Inplace by default
    Count data are temporarily converted to a numpy array for normalization if necessary
    """

    def __init__(
        self,
        adata: ad.AnnData,
        method: str,
        impute_fn: Callable | None = None,
        inplace=True,
        make_sparse=True,
        supported_methods=IMPLEMENTED_NORMALIZATION,
    ) -> None:
        self.ad = adata if inplace else adata.copy()
        if method.lower() not in supported_methods:
            raise ValueError(f"Method {method} not implemented!")
        self.inplace = inplace
        self.method = method
        self.make_sparse = make_sparse
        self.ad.layers["counts"] = adata.X
        if not (isinstance(adata.X, np.ndarray)):
            self.counts = adata.X.toarray().copy()
        else:
            self.counts = adata.X.copy()  # A sample x gene ndarray
        if impute_fn:
            self.counts = impute_fn(self.counts)

    def alr(
        self,
        by: int | str,
        var_col: str = None,
        condition_col: str = "",
        scales: dict | list = None,
        gamma: float = 0,
    ) -> np.ndarray:
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
        if gamma and condition_col and scales:
            return self._alr_scale(index, gamma, scales, condition_col)
        return comp.alr(self.counts, index)

    def _alr_scale(
        self, by: int, gamma, scales: dict | list, condition_col: str
    ) -> np.ndarray:
        """ALR with informed scale adjustment, adapted from [2]

        Parameters
        ----------
        scales : a dictionary mapping condition names to custom scale values for
           those conditions

        Returns
        -------


        Notes
        -----

        """
        gamma = max(gamma, 0.1)
        rng = np.random.default_rng(1)
        indices = np.arange(self.ad.shape[0])
        uniques = self.ad.obs[condition_col].unique()
        if not isinstance(scales, dict):
            lookup: dict = {u: scales[i] for i, u in enumerate(uniques)}
        else:
            lookup: dict = scales
        alr_adjusted = np.empty((self.counts.shape[0], self.counts.shape[1] - 1))
        ref_vals = self.conts[:, by]
        self.counts = np.delete(self.counts, (by), axis=1)
        for u in uniques:
            scale_factor = np.log2(lookup.get(u, 1))
            locs = indices[self.ad.obs[condition_col] == u]
            draws = rng.normal(scale_factor, gamma, len(locs)).reshape((-1, 1))
            ref_adj = np.log2(ref_vals) - draws
            adj = np.log2(self.counts[locs, :]) - ref_adj
            alr_adjusted[locs, :] = adj
        return alr_adjusted

    @r_cleanup
    def tmm(self) -> np.ndarray:
        np_cv_rules = default_converter + numpy2ri.converter
        with np_cv_rules.context():
            ro.globalenv["mat"] = np.transpose(self.counts)
        ro.r("dge <- edgeR::DGEList(mat)")
        ro.r("edgeR::normLibSizes(dge)")
        ro.r("counts <- edgeR::cpm(dge, log = TRUE)")
        return np.transpose(np.asarray(ro.r("counts")))

    def tpm(
        self, length_col: str = "SEQLENGTH", avg_fragment_size: float = 0
    ) -> np.ndarray:
        """Transcripts per million

        Notes
        -----
        Calculation adapted from [1], but using gene length instead of effective length
        if average fragment size isn't given
        """
        lengths = self.ad.var[length_col]
        if avg_fragment_size:
            lengths = lengths - avg_fragment_size + 1
        numer = np.log(self.counts) - np.reshape(np.log(lengths), (1, -1))
        denom = np.log(np.nansum(np.exp(numer), axis=1)).reshape(-1, 1)
        tpm = np.exp(numer - denom + np.log(1e6))
        return np.nan_to_num(tpm, neginf=0)

    def fkpm(
        self, length_col: str = "SEQLENGTH", avg_fragment_size: float = 0
    ) -> np.ndarray:
        """Fragments/reads per kilobase million

        Notes
        -----
        Calculation adapted from [1], but using gene length instead of effective length
        if average fragment size isn't given
        """
        lengths = self.ad.var[length_col]
        if avg_fragment_size:
            lengths = lengths - avg_fragment_size + 1
        numer = np.log(self.counts) - np.log(lengths).values.reshape(1, -1)
        denom = np.log(np.nansum(self.counts, axis=1).reshape(-1, 1))
        fpkm = np.exp(numer - denom + np.log(1e9))
        return np.nan_to_num(fpkm, neginf=0)

    def _clr_scale(self, gamma, scales: dict | list, condition_col: str) -> np.ndarray:
        """CLR with informed scale model, adapted from [2]

        Parameters
        ----------
        scales : a dictionary mapping condition names to custom scale values for
            those conditions

        Returns
        -------


        Notes
        -----

        """
        gamma = max(gamma, 0.1)
        rng = np.random.default_rng(1)
        indices = np.arange(self.ad.shape[0])
        clr_adjusted = np.empty(self.ad.shape)
        uniques = self.ad.obs[condition_col].unique()
        if not isinstance(scales, dict):
            lookup: dict = {u: scales[i] for i, u in enumerate(uniques)}
        else:
            lookup: dict = scales
        for u in uniques:
            scale_factor = np.log2(lookup.get(u, 1))
            locs = indices[self.ad.obs[condition_col] == u]
            draws = rng.normal(scale_factor, gamma, len(locs)).reshape((-1, 1))
            adj = np.log2(self.counts[locs, :]) + draws
            clr_adjusted[locs, :] = adj
        return clr_adjusted

    def log_clr(self, features=None, feature_col="gene_id") -> np.ndarray:
        """Implementation of logCLR [3]

        Notes
        -----
        <2025-02-20 Thu> This extension of CLR was developed in the context of
        k-means clustering, unsure of its performance for machine learning
        """
        normalized = np.empty_like(self.counts.shape)
        if features:
            gmean = stats.gmean(
                self.counts[:, self.ad.var[feature_col].isin(features)],
                axis=1,
                nan_policy="omit",
            ).reshape((-1, 1))
        else:
            gmean = stats.gmean(self.counts, axis=1, nan_policy="omit").reshape((-1, 1))
        log_gmean = np.log(gmean)

        def helper(index: int) -> None:
            cur = self.counts[index, :]
            ratio = cur / gmean[index]
            log_cur = np.log(cur)
            less_than = ratio <= 1
            normalized[index, less_than] = -(
                -(np.log(1 - log_cur[less_than] - log_gmean[index]) ** 2)
            )
            normalized[index, ~less_than] = (
                log_cur[~less_than] - log_gmean[index]
            ) ** 2

        _ = [helper(i) for i in range(self.counts.shape[0])]

        return normalized

    def clr(
        self, features=None, gamma=None, scales=None, feature_col: str = "gene_id"
    ) -> np.ndarray:
        if gamma and scales:
            return self._clr_scale(gamma=gamma, scales=scales)
        elif features:
            gmean = stats.gmean(
                self.counts[:, self.ad.var[feature_col].isin(features)],
                axis=1,
                nan_policy="omit",
            ).reshape((-1, 1))
            clr = np.log(self.counts) - np.log(gmean)
            return clr
        return comp.clr(self.counts)

    def run(self, **kwargs) -> ad.AnnData | None:
        match self.method:
            case "clr":
                normalized = self.clr()
            case "tmm":
                normalized = self.tmm()
            case "alr":
                normalized = self.alr(**kwargs)
            case "tpm":
                normalized = self.tpm(**kwargs)
            case _:
                normalized = np.array()
        self.ad.X = sparse.csc_matrix(normalized) if self.make_sparse else normalized
        if not self.inplace:
            return self.ad
