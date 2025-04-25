#!/usr/bin/env ipython
from typing import Callable, Iterable

import anndata as ad
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import skbio.stats.composition as comp
from rpy2.robjects import default_converter, numpy2ri
from scipy import sparse, stats

from too_predict.simulation import IMPLEMENTED_SIMULATION, Simulator
from too_predict.utils import (
    add_gc_content,
    counts_into_r,
    np_from_r,
    np_to_r,
    r_cleanup,
)

IMPLEMENTED_TRANSFORMATION = {
    "clr",
    "tmm",
    "alr",
    "tpm",
    "fpkm",
    "cqn",
    "robust_clr",
    "none",
    "qsmooth",
}
IMPLEMENTED_TRANSFORMATION = IMPLEMENTED_TRANSFORMATION | IMPLEMENTED_SIMULATION

"""
References
[1] Pachter, L. (2011). Models for transcript quantification from RNA-Seq. ArXiv. https://arxiv.org/abs/1104.3889
[2] Bennett, A. R., Lundstrøm, J., Chatterjee, S., & Bojar, D. (2025). Compositional data analysis enables statistical rigor in comparative glycomics. Nature Communications, 16(1), 1-15. https://doi.org/10.1038/s41467-025-56249-3
[3] Godichon-Baggioni, A., Maugis-Rabusseau, C., & Rau, A. (2018). Clustering transformed compositional data using K-means, with applications in gene expression and bicycle sharing system data. Journal of Applied Statistics, 46(1), 47–65. https://doi.org/10.1080/02664763.2018.1454894
[4] Martino C, Shenhav L, Marotz CA, Armstrong G, McDonald D, Vázquez-Baeza Y, Morton JT, Jiang L, Dominguez-Bello MG, Swafford AD, Halperin E, Knight R. Context-aware dimensionality reduction deconvolutes gut microbial community dynamics. Nat Biotechnol. 2021 Feb;39(2):165-168. doi: 10.1038/s41587-020-0660-7. Epub 2020 Aug 31. PMID: 32868914; PMCID: PMC7878194.
"""


class Transformer:
    """Class for transforming counts in the given adata object. Inplace by default
    Count data are temporarily converted to a numpy array for normalization if necessary
    """

    def fit(self, data: ad.AnnData | np.ndarray | pd.DataFrame) -> None:
        if isinstance(data, ad.AnnData):
            self.counts_only = False
            self.adata = data if self.inplace else data.copy()
            self.adata.layers["counts"] = data.X.copy()
            if sparse.issparse(data.X):
                self.counts = data.X.toarray().copy()
            else:
                self.counts = data.X.copy()  # A sample x feature ndarray
        else:
            self.adata = None
            self.counts_only = True
            self.counts = data.toarray() if sparse.issparse(data) else data

    def __init__(
        self,
        method: str,
        impute_fn=None,
        inplace=False,
        make_sparse=True,
        supported_methods=IMPLEMENTED_TRANSFORMATION,
        **kwargs,
    ) -> None:
        self.counts: np.ndarray | pd.DataFrame
        self.adata: ad.AnnData | None
        self.inplace = inplace
        self.method = method
        self.make_sparse = make_sparse
        if method is not None and method.lower() not in supported_methods:
            raise ValueError(f"Method {method} not implemented!")
        self.kwargs = kwargs
        self.impute: Callable[[np.ndarray], np.ndarray] = impute_fn

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
        if isinstance(by, str) and self.counts_only:
            query = np.where(self.counts.columns == by)
            index = query[0][0]
        elif isinstance(by, str) and var_col:
            query = np.where(self.adata.var[var_col] == by)
            index = query[0][0]
            if len(query) > 1:
                raise ValueError("Key `by` is not unique!")
        elif isinstance(by, str):
            index = self.adata.var.index.get_loc(by)
            if len(by) > 1:
                raise ValueError("Key `by` is not unique!")
        else:
            index = by
        if not self.counts_only:
            self.adata = ad.concat(
                [self.adata[:, :index], self.adata[:, index + 1 :]],
                axis="var",
                merge="same",
                uns_merge="same",
            )
        if gamma and condition_col and scales:
            return self._alr_scale(index, gamma, scales, condition_col)
        return comp.alr(self.counts, index)

    def _alr_scale(
        self,
        by: int,
        gamma,
        scales: dict | list,
        condition_col: str = "",
        conditions: np.ndarray | None = None,
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
        indices = np.arange(self.counts.shape[0])

        if not self.counts_only:
            uniques = self.adata.obs[condition_col].unique()
        elif conditions is not None and len(conditions) == self.counts.shape[0]:
            uniques = set(conditions)
        else:
            raise ValueError(
                "Conditions for each sample must be provided if passing only counts!"
            )

        if not isinstance(scales, dict):
            lookup: dict = {u: scales[i] for i, u in enumerate(uniques)}
        else:
            lookup: dict = scales
        alr_adjusted = np.empty((self.counts.shape[0], self.counts.shape[1] - 1))
        ref_vals = self.counts[:, by]
        self.counts = np.delete(self.counts, (by), axis=1)
        for u in uniques:
            scale_factor = np.log2(lookup.get(u, 1))
            if not self.counts_only:
                locs = indices[self.adata.obs[condition_col] == u]
            else:
                locs = indices[conditions == u]
            draws = rng.normal(scale_factor, gamma, len(locs)).reshape((-1, 1))
            ref_adj = np.log2(ref_vals) - draws
            adj = np.log2(self.counts[locs, :]) - ref_adj
            alr_adjusted[locs, :] = adj
        return alr_adjusted

    @r_cleanup
    def cqn(
        self,
        length_col: str = "SEQLENGTH",
        gc_col: str | None = None,
        size_factors: str | None = None,
    ):
        # [2025-04-25 Fri]
        # BUG: could not find function "mclustBIC"
        na_lengths = self.adata.var[length_col].isna()
        print(f"WARNING {na_lengths.sum()} genes have no length data! Removing...")
        self.adata = self.adata[:, ~na_lengths]
        self.counts = self.counts[:, ~na_lengths]
        ro.globalenv["lengths"] = ro.IntVector(self.adata.var[length_col])
        if gc_col is None:
            add_gc_content(self.adata, id_col="GENEID")
            gc_col = "gc_content"
        na_gc = self.adata.var[gc_col].isna()
        if any(na_gc):
            print(f"WARNING {na_gc.sum()} genes have no GC content data! Removing...")
            self.adata = self.adata[:, ~na_gc]
            self.counts = self.counts[:, ~na_gc]
        ro.globalenv["gc_content"] = ro.FloatVector(self.adata.var[gc_col])
        if size_factors is None:
            ro.r("size_factors <- NULL")
        else:
            ro.globalenv["size_factors"] = ro.FloatVector(self.adata.var[size_factors])

        counts_into_r(self.adata, counts=self.counts)
        ro.r("norm <- cqn::cqn(counts, gc_content, lengths)")
        return np.transpose(np_from_r(ro.globalenv["norm"]))

    @r_cleanup
    def tmm(self, log=True) -> np.ndarray:
        np_cv_rules = default_converter + numpy2ri.converter
        with np_cv_rules.context():
            ro.globalenv["mat"] = np.transpose(self.counts)
        ro.r("dge <- edgeR::DGEList(mat)")
        ro.r("edgeR::normLibSizes(dge)")
        if log:
            ro.r("counts <- edgeR::cpm(dge, log = TRUE)")
        else:
            ro.r("counts <- edgeR::cpm(dge, log = FALSE)")
        return np.transpose(np.asarray(ro.r("counts")))

    @r_cleanup
    def qsmooth(self) -> np.ndarray:
        # TODO: Very slow, if it works well try own implementation
        np_to_r(np.transpose(self.counts), "matrix")
        ro.globalenv["labels"] = ro.FactorVector(self.adata.obs["tumor_type"])
        ro.r("library(qsmooth)")
        ro.r("normed <- qsmooth(object = matrix, group_factor = labels)")
        ro.r("data <- qsmoothData(normed)")
        return np.transpose(np_from_r(ro.globalenv["data"]))

    def tpm(
        self,
        length_col: str = "SEQLENGTH",
        avg_fragment_size: float = 0,
        gene_lengths: Iterable | None = None,
    ) -> np.ndarray:
        """Transcripts per million

        Notes
        -----
        Calculation adapted from [1], but using gene length instead of effective length
        if average fragment size isn't given
        """
        if not self.counts_only:
            lengths = self.adata.var[length_col]
        elif not isinstance(gene_lengths, np.ndarray) and gene_lengths is not None:
            lengths = np.array(gene_lengths)
        elif gene_lengths is not None:
            lengths = gene_lengths
        else:
            raise ValueError("Gene lengths not provided")
        if avg_fragment_size:
            lengths = lengths - avg_fragment_size + 1  # Get effective length
        numer = np.log(self.counts) - np.reshape(np.log(lengths), (1, -1))
        denom = np.log(np.nansum(np.exp(numer), axis=1)).reshape(-1, 1)
        tpm = np.exp(numer - denom + np.log(1e6))
        return np.nan_to_num(tpm, neginf=0)

    def fpkm(
        self,
        length_col: str = "SEQLENGTH",
        avg_fragment_size: float = 0,
        gene_lengths: Iterable | None = None,
    ) -> np.ndarray:
        """Fragments/reads per kilobase million

        Notes
        -----
        Calculation adapted from [1], but using gene length instead of effective length
        if average fragment size isn't given
        """
        if not self.counts_only:
            lengths = self.adata.var[length_col]
        elif not isinstance(gene_lengths, np.ndarray) and gene_lengths is not None:
            lengths = np.array(gene_lengths)
        elif gene_lengths is not None:
            lengths = gene_lengths
        else:
            raise ValueError("Gene lengths not provided")
        if avg_fragment_size:
            lengths = lengths - avg_fragment_size + 1
        numer = np.log(self.counts) - np.log(lengths).values.reshape(1, -1)
        denom = np.log(np.nansum(self.counts, axis=1).reshape(-1, 1))
        fpkm = np.exp(numer - denom + np.log(1e9))
        return np.nan_to_num(fpkm, neginf=0)

    def _clr_scale(
        self,
        gamma,
        scales: dict | list,
        condition_col: str,
        conditions: np.ndarray | None = None,
    ) -> np.ndarray:
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
        indices = np.arange(self.counts.shape[0])

        if not self.counts_only:
            uniques = self.adata.obs[condition_col].unique()
        elif conditions is not None and len(conditions) == self.counts.shape[0]:
            uniques = set(conditions)
        else:
            raise ValueError(
                "Conditions for each sample must be provided if passing only counts!"
            )
        clr_adjusted = np.empty_like(self.counts)

        if not isinstance(scales, dict):
            lookup: dict = {u: scales[i] for i, u in enumerate(uniques)}
        else:
            lookup: dict = scales
        for u in uniques:
            scale_factor = np.log2(lookup.get(u, 1))
            if not self.counts_only:
                locs = indices[self.adata.obs[condition_col] == u]
            else:
                locs = indices[conditions == u]
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
        <2025-03-04 Tue> Buggy, don't use
        """
        normalized = np.empty_like(self.counts)
        if features:
            if not self.counts_only:
                subset = self.adata.var[feature_col].isin(features)
            else:
                subset = self.counts.columns.isin(features)
            gmean = stats.gmean(
                self.counts[:, subset], axis=1, nan_policy="omit"
            ).reshape((-1, 1))
        else:
            gmean = stats.gmean(self.counts, axis=1, nan_policy="omit").reshape((-1, 1))
        log_gmean = np.log(gmean)

        def helper(index: int) -> None:
            cur = self.counts[index, :]
            ratio = cur / gmean[index]
            log_cur = np.log(cur)
            less_than = ratio <= 1
            normalized[index, less_than] = (
                -np.log(1 - log_cur[less_than] - log_gmean[index])
            ) ** 2  # BUG: this expression produces nans
            normalized[index, ~less_than] = (
                log_cur[~less_than] - log_gmean[index]
            ) ** 2

        _ = [helper(i) for i in range(self.counts.shape[0])]

        return normalized

    def clr(
        self,
        features=None,
        gamma=None,
        scales=None,
        feature_col: str = "GENEID",
        robust: bool = False,
    ) -> np.ndarray:
        if gamma and scales and not robust:
            return self._clr_scale(gamma=gamma, scales=scales)
        elif features is None and not robust:
            return comp.clr(self.counts)

        if not self.counts_only and features is not None:
            subset = self.adata.var[feature_col].isin(features)
        elif features is not None:
            subset = self.counts.columns.isin(features)
        else:
            subset = np.ones(self.counts.shape[1]).astype(bool)

        if robust:
            gmean = [
                stats.gmean(self.counts[i, self.counts[i, :] != 0 & subset])
                for i in range(self.counts.shape[0])
            ]
            gmean = np.array(gmean)
        else:
            gmean = stats.gmean(self.counts[:, subset], axis=1, nan_policy="omit")
        clr = np.log(self.counts) - np.log(gmean.reshape(-1, 1))
        if robust:
            clr[clr == -np.inf] = 0
        return clr

    def transform(self, _=None) -> ad.AnnData | None | np.ndarray:
        if self.impute and self.method != "robust_clr":
            self.counts = self.impute(self.counts)
        if self.method in IMPLEMENTED_SIMULATION:
            sim = Simulator(self.method, self.counts, **self.kwargs)
            normalized = sim()
            if not self.counts_only:
                self.adata.uns["mc_instances"] = normalized
        else:
            match self.method:
                case "robust_clr":
                    self.kwargs.update({"robust": True})
                    normalized = self.clr(**self.kwargs)
                case "cqn":
                    normalized = self.cqn(**self.kwargs)
                case "clr":
                    normalized = self.clr(**self.kwargs)
                case "tmm":
                    normalized = self.tmm()
                case "alr":
                    normalized = self.alr(**self.kwargs)
                case "tpm":
                    normalized = self.tpm(**self.kwargs)
                case "qsmooth":
                    normalized = self.qsmooth(**self.kwargs)
                case "fpkm":
                    normalized = self.fpkm(**self.kwargs)
                case "log_clr":
                    normalized = self.log_clr(**self.kwargs)
                case "none" | _:
                    normalized = self.counts
        if not self.counts_only:
            if self.method not in IMPLEMENTED_SIMULATION:
                self.adata.X = (
                    sparse.csc_matrix(normalized) if self.make_sparse else normalized
                )
            if not self.inplace:
                return self.adata
        else:
            return normalized

    def fit_transform(
        self, data: ad.AnnData | np.ndarray | pd.DataFrame, _=None
    ) -> ad.AnnData | None | np.ndarray:
        self.fit(data)
        return self.transform()
