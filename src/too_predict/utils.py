#!/usr/bin/env ipython

import importlib.resources as res
from pathlib import Path
from typing import Callable

import anndata as ad
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import RObject, pandas2ri
from rpy2.robjects.packages import STAP, importr
from scipy import sparse

import too_predict

# R libraries
base = None
ensembldb = None


def get_data(path: str) -> Path:
    """Retrieve the path of a file in this package's `data` directory
    :param: path relative path to the desired file
    """
    with res.path(too_predict) as root:
        file = root.parent.parent.joinpath("data").joinpath(path)
        if file.exists():
            return file.absolute()
        raise FileNotFoundError(f"{path} doesn't exist!")


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


def library(r_script: str) -> STAP:
    """Import `r script` in src/R as a STAP"""
    with res.path(too_predict) as root:
        r_src = root.parent.joinpath("R")
        script = r_src.joinpath(r_script)
        if script.exists():
            text = script.read_text()
            return STAP(text, Path(r_script).stem)
        raise FileNotFoundError(f"{r_script} doesn't exist in src/R!")
