#!/usr/bin/env ipython

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from scipy import sparse

import too_predict.go_utils as gt
import too_predict.utils as ut

IMPLEMENTED_RECODING = {"GO", "BisqueMarker"}


class Recoder:
    def __init__(self, method: str, layer: str = None, **kwargs) -> None:
        if method not in IMPLEMENTED_RECODING:
            raise ValueError(f"method {method} not supported!")
        self.method = method
        self.layer = layer
        self.kwargs = kwargs

    @ut.r_cleanup
    def _bisque(self, reference_file: Path, mode: str = "marker") -> ad.AnnData:
        ut.source("utils.R", in_r=True)
        if mode == "marker":
            ut.source("utils.R", in_r=True)
            ut.np_to_r(np.transpose(self.counts), r_symbol="counts")
            ro.globalenv["marker_ref"] = str(reference_file.absolute())
            ro.globalenv["samples"] = ro.StrVector(self.adata.obs.index)
            ro.globalenv["vars"] = ro.StrVector(self.adata.var.index)
            ro.r("""
            result <- bisque_marker_wrapper(counts, sample_names = samples,
                var_names = vars, markers = marker_ref)
            """)
            matrix = ut.np_from_r(ro.r("result$bulk.props"))
            genes_used = pd.DataFrame(
                {
                    "set_name": ro.r("names(result$genes.used)"),
                    "genes_used": ro.r("result$genes.used"),
                }
            )
            genes_used.index = genes_used["set_name"]
            genes_used.loc[:, "used_size"] = genes_used["genes_used"].apply(
                lambda x: len(x)
            )
            result = ad.AnnData(
                X=np.transpose(matrix), obs=self.adata.obs, var=genes_used
            )
        else:
            raise ValueError("Reference mode for Bisque not implemented yet!")
        return result

    def fit(self, adata: ad.AnnData) -> None:
        self.adata = adata.copy()
        self.counts = adata.X if self.layer is None else adata.layers[self.layer]
        self.was_sparse = sparse.issparse(self.counts)
        self.counts = self.counts.toarray() if self.was_sparse else self.counts

    def transform(self) -> ad.AnnData:
        match self.method:
            case "GO":
                rg = gt.RecodeGO(**self.kwargs)
                rg.adata = self.adata
                recoded = rg.transform()
            case "BisqueMarker":
                recoded = self._bisque(mode="marker", **self.kwargs)
            case _:
                raise ValueError()
        return recoded

    def fit_transform(self, adata: ad.AnnData) -> ad.AnnData:
        self.fit(adata)
        return self.transform()
