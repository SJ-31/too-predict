#!/usr/bin/env ipython

from typing import Callable

import anndata as ad
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import RObject, pandas2ri
from rpy2.robjects.packages import importr
from scipy import sparse

base = None


def df_from_r(robj) -> pd.DataFrame:
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(robj)


def r_cleanup(fn: Callable):
    def wrp(*args, **kwargs):
        val = fn(*args, **kwargs)
        ro.r("rm(list=ls())")
        ro.r("gc()")
        return val

    return wrp


# TODO: make a wrapper for skbio.stats.composition.alr to take a colname if it's been
# given a df or adata
#


@r_cleanup
def dgelist2anndata(rds: str | RObject) -> ad.AnnData:
    global base
    if not base:
        base = importr("base")
    """Read a DGEList object stored in an rds file `rds` into an AnnData object"""
    if isinstance(rds, str):
        ro.globalenv["dge"] = base.readRDS(rds)
        adata = ad.AnnData(
            X=sparse.csr_matrix(ro.r("t(dge$counts)")),
            obs=df_from_r(ro.r("dge$samples")),
            var=df_from_r(ro.r("dge$gene")),
        )
        adata.var.index = adata.var.iloc[:, 0]
    return adata
