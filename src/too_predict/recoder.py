#!/usr/bin/env ipython

from pathlib import Path

import anndata as ad
import gseapy as gp
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import yaml
from scipy import sparse

import too_predict.go_utils as gt
import too_predict.utils as ut
from too_predict.transformer import Transformer

IMPLEMENTED_RECODING = {"go", "bisquemarker", "plage", "gsva"}


class Recoder:
    def __init__(self, method: str, layer: str = None, **kwargs) -> None:
        if method.lower() not in IMPLEMENTED_RECODING:
            raise ValueError(f"method {method} not supported!")
        self.method = method.lower()
        self.layer = layer
        self.kwargs = kwargs

    def _counts_into_r(self) -> None:
        ut.counts_into_r(self.adata, self.counts)

    @ut.r_cleanup
    def _plage(
        self, reference: Path | dict, metadata: Path | pd.DataFrame | None = None
    ) -> ad.AnnData:
        if isinstance(metadata, Path):
            metadata = pd.read_csv(metadata, sep="\t")
        ro.r("library(GSVA)")
        self._counts_into_r()
        if isinstance(reference, Path):
            ro.r(f"gs <- yaml::read_yaml('{str(reference.absolute())}')")
        else:
            ro.globalenv["gs"] = ro.ListVector(reference)
        ro.r("params <- plageParam(exprData = counts, geneSets = gs)")
        ro.r("plage <- gsva(params)")
        vals: np.ndarray = np.transpose(ut.np_from_r(ro.globalenv["plage"]))
        var = pd.DataFrame(index=ro.r("rownames(plage)"))
        if metadata is not None:
            var = var.merge(metadata, left_index=True, right_index=True, how="left")
        return ad.AnnData(X=vals, var=var, obs=self.adata.obs)

    def _gsva(
        self,
        reference: Path | dict,
        metadata: Path | pd.DataFrame | None = None,
        **kwargs,
    ) -> ad.AnnData:
        if isinstance(metadata, Path):
            metadata = pd.read_csv(metadata, sep="\t")
        if isinstance(reference, Path):
            with open(reference, "r") as f:
                reference = yaml.safe_load(f)
        for_gp = pd.DataFrame(
            np.transpose(self.counts),
            index=self.adata.var.index,
            columns=self.adata.obs.index,
        )
        if not kwargs:
            kwargs = {"kcdf": "Gaussian", "mx_diff": True}
        result = gp.gsva(for_gp, gene_sets=reference, **kwargs)
        vals = result.res2d.pivot(index="Name", columns="Term", values="ES")
        var = pd.DataFrame(index=vals.columns)
        if metadata is not None:
            var = var.merge(metadata, left_index=True, right_index=True, how="left")
        return ad.AnnData(X=vals.values.astype(np.float64), var=var, obs=self.adata.obs)

    @ut.r_cleanup
    def _bisque(self, reference_file: Path, mode: str = "marker") -> ad.AnnData:
        ut.source("utils.R", in_r=True)
        if mode == "marker":
            ut.source("utils.R", in_r=True)
            self._counts_into_r()
            ro.globalenv["marker_ref"] = str(reference_file.absolute())
            ro.r("result <- bisque_marker_wrapper(counts, markers = marker_ref)")
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
        if self.method == "gsva":
            # [2025-04-25 Fri] Original gsva uses cqn but this is too slow
            # use tmm instead
            T = Transformer("tmm", log=False, inplace=False, impute_fn=None)
            self.adata = T.fit_transform(adata)
        else:
            self.adata = adata.copy()
        self.counts = adata.X if self.layer is None else adata.layers[self.layer]
        self.was_sparse = sparse.issparse(self.counts)
        self.counts = self.counts.toarray() if self.was_sparse else self.counts

    def transform(self) -> ad.AnnData:
        match self.method:
            case "go":
                rg = gt.RecodeGO(**self.kwargs)
                rg.adata = self.adata
                recoded = rg.transform()
            case "bisquemarker":
                recoded = self._bisque(mode="marker", **self.kwargs)
            case "plage":
                recoded = self._plage(**self.kwargs)
            case "gsva":
                recoded = self._gsva(**self.kwargs)
            case _:
                raise ValueError()
        if self.was_sparse:
            recoded.X = sparse.csc_array(recoded.X)
        return recoded

    def fit_transform(self, adata: ad.AnnData) -> ad.AnnData:
        self.fit(adata)
        return self.transform()
