#!/usr/bin/env ipython

import importlib.resources as res
from functools import reduce
from pathlib import Path
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2 import rinterface, rinterface_lib
from rpy2.rinterface_lib.sexp import (
    NACharacterType,
    NAComplexType,
    NAIntegerType,
    NALogicalType,
    NARealType,
)
from rpy2.robjects import RObject, pandas2ri
from rpy2.robjects.packages import STAP, InstalledPackage, InstalledSTPackage, importr
from scipy import sparse, stats

import too_predict

NA_TYPES: set = {
    NACharacterType,
    NAIntegerType,
    NALogicalType,
    NARealType,
    NAComplexType,
}


def get_data(path: str) -> Path:
    """Retrieve the path of a file in this package's `data` directory
    :param: path relative path to the desired file
    """
    with res.path(too_predict) as root:
        file = root.parent.parent.joinpath("data").joinpath(path)
        if file.exists():
            return file.absolute()
        raise FileNotFoundError(f"{path} doesn't exist!")


def df_to_r(df: pd.DataFrame, r_symbol: str = ""):
    with (ro.default_converter + pandas2ri.converter).context():
        converted = ro.conversion.get_conversion().py2rpy(df)
        if r_symbol:
            ro.globalenv[r_symbol] = converted
        else:
            return converted


def df_from_r(robj) -> pd.DataFrame:
    with (ro.default_converter + pandas2ri.converter).context():
        converted = ro.conversion.get_conversion().rpy2py(robj)
        # BUG <2025-02-11 Tue>: rpy2 doesn't convert R NAs correctly
        df = converted.map(lambda x: np.nan if type(x) in NA_TYPES else x)
        return df


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
    """Read a DGEList object stored in an rds file `rds` into an AnnData object"""
    base = library("base")
    if isinstance(rds, str):
        ro.globalenv["dge"] = base.readRDS(rds)
        adata = ad.AnnData(
            X=sparse.csr_matrix(ro.r("t(dge$counts)")),
            obs=df_from_r(ro.r("dge$samples")),
            var=df_from_r(ro.r("dge$gene")),
        )
        adata.var.index = adata.var.iloc[:, 0]
    return adata


def source(r_script: str) -> STAP:
    """Import `r script` in src/R as a STAP"""
    with res.path(too_predict) as root:
        r_src = root.parent.joinpath("R")
        script = r_src.joinpath(r_script)
        if script.exists():
            text = script.read_text()
            return STAP(text, Path(r_script).stem)
        raise FileNotFoundError(f"{r_script} doesn't exist in src/R!")


def library(package: str) -> InstalledSTPackage | InstalledPackage:
    """Load an R package into the global environment if it has not been loaded yet"""
    check = globals().get(package)
    if not check or type(check) not in {InstalledPackage, InstalledSTPackage}:
        loaded = importr(package)
        globals()[package] = loaded
        return loaded
    return check


@r_cleanup
def add_gene_metadata(
    adata: ad.AnnData,
    keycol: str = "",
    columns=("GENENAME", "GENEID", "GENEBIOTYPE", "SEQNAME"),
    ensdb_path: str = "",
    keytpe: str = "GENEID",
) -> None:
    """Add gene metadata from the ensembldb object stored at `ensdb_path`"""
    ensembldb = library("ensembldb")
    for c in columns:
        if c in adata.var.columns:
            print(f"WARNING: column {c} already exists in adata.var, removing...")
            adata.var = adata.var.drop(c, index="columns")
    if not ensdb_path:
        ensdb_path = str(get_data("Homo_sapiens.GRCh38.113.sqlite"))
    ro.globalenv["db"] = ensembldb.EnsDb(ensdb_path)
    ro.globalenv["cols"] = ro.StrVector(columns)
    if keycol:
        keys = adata.var[keycol]
    else:
        keys = adata.var.index
    ro.globalenv["keys"] = ro.StrVector(keys)
    ro.globalenv["anno"] = ro.r(
        f"AnnotationDbi::select(db, keys = keys, columns = cols, keytype = '{keytpe}')"
    )
    result: pd.DataFrame = df_from_r(ro.r("anno"))
    if keycol:
        result[keycol] = ro.StrVector(result[keytpe])
    else:
        result.index = ro.StrVector(result[keytpe])

    join_on = keycol if keycol else None
    adata.var = adata.var.join(result, on=join_on, how="left")


def str_mode(array: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Scipy's mode function made compatible with string arrays
    returns a tuple of [mode, count]
    """
    array = array.copy()
    uniques = np.unique(array)
    int2name = {}
    for i, n in enumerate(uniques):
        array[array == n] = i
        int2name[i] = n.item()
    result = stats.mode(array.astype(int), **kwargs)
    names = [int2name[n.item()] for n in result.mode]
    return np.array(names), result.count


def get_star_strandedness(
    sample_name: str,
    df: pd.DataFrame,
    unstranded_col: str = "unstranded",
    forward_col: str = "stranded_first",
    reverse_col: str = "stranded_second",
) -> pd.DataFrame:
    counts: dict = df.loc[:, [unstranded_col, forward_col, reverse_col]].sum().to_dict()
    counts["name"] = sample_name
    return pd.DataFrame(counts, index=[0])


def read_gdc_counts(
    path: str,
    name: str,
    count_col: str,
    var_col: tuple = ("gene_id", "gene_name"),
    var_blacklist: tuple = (
        "N_unmapped",
        "N_multimapping",
        "N_noFeature",
        "N_ambiguous",
    ),
) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(path, sep="\t", comment="#")
    df = df.loc[~df[var_col[0]].isin(var_blacklist), :]
    print(get_star_strandedness(name, df))
    vars: pd.DataFrame = df.loc[:, var_col]
    df = df.drop(columns=list(var_col))
    df = df.loc[:, [count_col]].rename({count_col: name}, axis="columns")
    if len(var_col) > 1:
        df.index = pd.MultiIndex.from_frame(vars)
    else:
        df.index = vars.iloc[:, 0]
    return df


def collect_gdc_counts(
    dir: str,
    count_col: str = "unstranded",
    sample_sheet: str = "",
    case_table: str = "",
    case_cols: tuple = ("primary_site",),
    case_id_col: str = "submitter_id",
) -> ad.AnnData:
    if not sample_sheet:
        sample_sheet = str(next(Path(dir).glob("gdc_sample_sheet*")))
    samples: pd.DataFrame = pd.read_csv(sample_sheet, sep="\t").rename(
        columns=lambda x: x.replace(" ", "_")
    )
    missing_cases: set = set(samples["Case_ID"])
    count_dfs = []
    for d, p, c in zip(samples["File_ID"], samples["File_Name"], samples["Case_ID"]):
        try:
            cur = read_gdc_counts(Path(dir).joinpath(d).joinpath(p), c, count_col)
            missing_cases.remove(c)
            count_dfs.append(cur)
        except FileNotFoundError:
            print(f"WARNING: Case {c} not found")
    joined = reduce(
        lambda x, y: x.join(y, on=["gene_id", "gene_name"], how="outer"), count_dfs
    )
    samples = samples.loc[~samples["Case_ID"].isin(missing_cases), :]
    var_df = joined.index.to_frame(index=False)
    if case_table:
        cases: pd.DataFrame = pd.read_csv(case_table, sep="\t")
        cases = cases.loc[:, [case_id_col] + list(case_cols)].drop_duplicates()
        samples = samples.merge(
            cases, how="left", left_on="Case_ID", right_on=case_id_col
        ).drop(case_id_col, axis="columns")
    return ad.AnnData(X=np.transpose(joined.values), obs=samples, var=var_df)


def read_existing[T](filename: Path, expr: Callable[[Path], T], read_fn) -> T:
    if filename.exists():
        return read_fn(filename)
    else:
        return expr(filename)
