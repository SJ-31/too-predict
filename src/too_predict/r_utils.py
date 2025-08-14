#!/usr/bin/env ipython

import importlib.resources as res
from pathlib import Path
from typing import Callable, Literal

import anndata as ad
import anndata2ri
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scanpy as sc
from rpy2.robjects import RObject, numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import STAP, InstalledPackage, InstalledSTPackage, importr
from scipy import sparse, stats

import too_predict
from too_predict.utils import get_data


def register_biocparallel(workers: int, param="MulticoreParam") -> None:
    ro.r("library(BiocParallel)")
    ro.r(f"register({param}(workers = {workers}))")


def adata_to_r(adata: ad.AnnData, r_symbol: str = "", to_matrix: bool = True):
    with localconverter(anndata2ri.converter):
        ro.globalenv["adata_tmp"] = adata
        id = r_symbol if r_symbol else "sce_tmp"
        ro.r(f"{id} <- as(adata_tmp, 'SingleCellExperiment')")
        if to_matrix:
            ro.r(f"assays({id})$X <- as.matrix(assays({id})$X)")
        ro.r("rm(adata_tmp)")
    if not r_symbol:
        return ro.r(id)


def np_to_r(arr: np.ndarray, r_symbol: str = "") -> None:
    np_cv_rules = ro.default_converter + numpy2ri.converter
    with np_cv_rules.context():
        ro.globalenv[r_symbol] = arr


def np_from_r(robj) -> np.ndarray:
    np_cv_rules = ro.default_converter + numpy2ri.converter
    with np_cv_rules.context():
        return ro.conversion.get_conversion().rpy2py(robj)


def df_to_r(df: pd.DataFrame, r_symbol: str = ""):
    with (ro.default_converter + pandas2ri.converter).context():
        converted = ro.conversion.get_conversion().py2rpy(df)
        if r_symbol:
            ro.globalenv[r_symbol] = converted
        else:
            return converted


def sclr(expr):
    return list(ro.r(expr))[0]


def df_from_r(robj) -> pd.DataFrame:
    with (ro.default_converter + pandas2ri.converter).context():
        ro.globalenv["df_from_r_tmp"] = robj
        if "data.frame" not in set(ro.r("class(df_from_r_tmp)")):
            raise ValueError("The given object is not a data.frame")
        ro.r("rm(df_from_r_tmp)")
        converted = ro.conversion.get_conversion().rpy2py(robj)
        # BUG <2025-02-11 Tue>: rpy2 doesn't convert R NAs correctly
        # df = converted.map(lambda x: np.nan if type(x) in NA_TYPES else x)
        return converted


def r_cleanup(fn: Callable):
    def wrp(*args, **kwargs):
        val = fn(*args, **kwargs)
        if not sclr("exists('NO_CLEANUP')") or not sclr("NO_CLEANUP"):
            ro.r("rm(list=ls())")
            ro.r("gc()")
        return val

    return wrp


@r_cleanup
def dgelist2anndata(rds: str | RObject) -> ad.AnnData:
    """Read a DGEList object stored in an rds file `rds` into an AnnData object"""
    base = library("base")
    if isinstance(rds, str):
        ro.globalenv["dge"] = base.readRDS(rds)
        adata: ad.AnnData = ad.AnnData(
            X=sparse.csc_array(ro.r("t(dge$counts)")),
            obs=df_from_r(ro.r("dge$samples")),
            var=df_from_r(ro.r("dge$gene")),
        )
        adata.var.index = adata.var.iloc[:, 0]
    return adata


def source(r_script: str, in_r=False) -> STAP | None:
    """Import `r script` in src/R as a STAP"""
    root = res.files(too_predict)
    r_src = root.parent.joinpath("R")
    script = r_src.joinpath(r_script)
    if not script.exists():
        raise FileNotFoundError(f"{r_script} doesn't exist in src/R!")
    if in_r:
        ro.r(f"source('{script.absolute()}')")
    else:
        text = script.read_text()
        return STAP(text, Path(r_script).stem)


def library(package: str) -> InstalledSTPackage | InstalledPackage:
    """Load an R package into the global environment if it has not been loaded yet"""
    check = globals().get(package)
    if not check or type(check) not in {InstalledPackage, InstalledSTPackage}:
        loaded = importr(package)
        globals()[package] = loaded
        return loaded
    return check


@r_cleanup
def tximport_salmon(
    files: list[Path], sample_names: list[str] | None = None, column="counts"
) -> pd.DataFrame:
    """Wrapper for importing multiple salmon files with tximport"""
    tx2gene = pd.read_csv(get_data("tx2gene.tsv"), sep="\t")
    df_to_r(tx2gene.loc[:, ["TXID", "GENEID"]], "tx2gene")
    ro.globalenv["paths"] = ro.StrVector([str(f) for f in files])
    # FIXME: would rather have tximport handle the file list automatically, but
    #   it will raise an error if any of the files don't have the same tx
    ro.r(f"""
    library(tidyverse)
    library(tximport)
    imp <- lapply(paths, \\(x) tximport(x, "salmon", tx2gene = tx2gene,
                                                ignoreTxVersion = TRUE))
    df <- lapply(imp, \\(x) rownames_to_column(as.data.frame(x${column}), "id")) |>
        reduce(\\(x, y) full_join(x, y, by = join_by(id))) |>
        column_to_rownames("id")
    colnames(df) <- paste0("V", seq_len(ncol(df)))
    """)
    genes = list(ro.r("rownames(df)"))
    counts = np_from_r(ro.r("as.matrix(df)"))
    sample_names = sample_names if sample_names else list(ro.r("colnames(df)"))
    result = pd.DataFrame(counts, index=genes, columns=sample_names)
    return result


@r_cleanup
def add_gene_metadata(
    adata: ad.AnnData,
    keycol: str = "",
    columns=("GENENAME", "GENEID", "GENEBIOTYPE", "SEQNAME", "SEQLENGTH"),
    ensdb_path: str = "",
    keytype: str = "GENEID",
) -> None:
    """Add gene metadata from the ensembldb object stored at `ensdb_path`"""
    ensembldb = library("ensembldb")
    for c in columns:
        if c in adata.var.columns:
            print(f"WARNING: column {c} already exists in adata.var, removing...")
            adata.var = adata.var.drop(c, index="columns")
    if not ensdb_path:
        ensdb_path = str(get_data("reference/Homo_sapiens.GRCh38.113.sqlite"))
    ro.globalenv["db"] = ensembldb.EnsDb(ensdb_path)
    ro.globalenv["cols"] = ro.StrVector(columns)
    if keycol:
        keys = adata.var[keycol]
    else:
        keys = adata.var.index
    ro.globalenv["keys"] = ro.StrVector(keys)
    ro.globalenv["anno"] = ro.r(
        f"AnnotationDbi::select(db, keys = keys, columns = cols, keytype = '{keytype}')"
    )
    result: pd.DataFrame = df_from_r(ro.r("anno"))
    if keycol:
        result[keycol] = ro.StrVector(result[keytype])
    else:
        result.index = ro.StrVector(result[keytype])

    join_on = keycol if keycol else None
    adata.var = adata.var.join(result, on=join_on, how="left")


def rename_genes(
    data: pd.DataFrame | ad.AnnData,
    old: str = "",
    new: str = "",
    use_ensembl_versions: bool = False,
    remove_versions: bool = False,
    id_col: str = "",
    mapping_file: str = "",
) -> tuple:
    """Rename gene ids between Ensembl ids, Gene symbols and Entrez (NCBI)
    Returns
    -------
    tuple of two dataframes, the first element containing successfully renamed entries,
        the second being failed entries i.e. old name not found in the lookup
    """
    options: set = {"ensembl", "entrez", "symbol"}
    if len({old, new} & options) != 2 and (not remove_versions):
        raise ValueError(f"Only {options} are supported for `old`, `new`")
    if mapping_file:
        id_map = pd.read_csv(mapping_file, sep="\t")
    else:
        id_map = pd.read_csv(get_data("mappings/ensembl_113_id_mapping.tsv"), sep="\t")

    if isinstance(data, ad.AnnData):
        was_adata = True
        df = data.var
    else:
        was_adata = False
        df = data

    old_names: pd.Series = df.index.to_series() if not id_col else df[id_col]
    if not remove_versions:
        if use_ensembl_versions and old == "ensembl":
            old = "ensembl_w_id"
        if old == "symbol":

            def get_correct_symbol(symbol, synonym):
                if symbol in old_names:
                    return symbol
                elif synonym in old_names:
                    return synonym
                return np.nan

            id_map.loc[:, "symbol"] = id_map["symbol"].combine(
                id_map["symbol_synonym"], get_correct_symbol
            )

        lookup: dict = {k: v for k, v in zip(id_map[old], id_map[new])}
        new_names: pd.Series = old_names.apply(lambda x: lookup.get(x, np.nan))
    else:
        new_names = old_names.str.replace("\\..*", "", regex=True)
    row_mask = new_names.notna()
    passed, failed = df.loc[row_mask, :], df.loc[~row_mask, :]
    new_names = new_names.dropna()
    if not id_col:
        passed.index = new_names
    else:
        passed[id_col] = new_names

    if was_adata:
        ad_passed, ad_failed = data[:, row_mask], data[:, ~row_mask]
        ad_passed.var, ad_failed.var = passed, failed
        return ad_passed.copy(), ad_failed.copy()
    return passed, failed


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


def counts_into_r(
    adata: ad.AnnData, counts: np.ndarray | None = None, symbol="counts", transpose=True
) -> None:
    ro.globalenv["sample_names"] = ro.StrVector(adata.obs.index.astype(str))
    ro.globalenv["var_names"] = ro.StrVector(adata.var.index.astype(str))
    to_send = adata.X if counts is None else counts
    to_send = to_send.toarray() if sparse.issparse(to_send) else to_send
    to_send = np.transpose(to_send) if transpose else to_send
    np_to_r(to_send, r_symbol=symbol)
    if transpose:
        ro.r(f"rownames({symbol}) <- var_names")
        ro.r(f"colnames({symbol}) <- sample_names")
    else:
        ro.r(f"colnames({symbol}) <- var_names")
        ro.r(f"rownames({symbol}) <- sample_names")


def r_null_if_none(obj, symbol, conversion=lambda x: x) -> None:
    """Assign `obj` to R object `symbol` iff obj is not None. Otherwise assign
    NULL to symbol
    """
    if obj is None:
        ro.r(f"{symbol} <- NULL")
    else:
        ro.globalenv[symbol] = conversion(obj)


@r_cleanup
def pooled_normalization(
    adata: ad.AnnData,
    cluster_method: Literal["scran", "leiden"] = "leiden",
    cluster_col: str | None = None,
    do_parallel: bool = True,
    n_workers: int = 8,
) -> None:
    if not do_parallel:
        ro.r("bppar <- NULL")
    else:
        ro.r(f"bppar <- BiocParallel::MulticoreParam(workers = {n_workers})")

    if cluster_col is not None:
        clusters = adata.obs[cluster_col]
    elif cluster_method == "leiden":
        adata_c = adata.copy()
        sc.pp.normalize_total(adata_c)
        sc.pp.log1p(adata_c)
        sc.pp.pca(adata_c, n_comps=10)
        sc.pp.neighbors(adata_c)
        sc.tl.leiden(adata_c)
        clusters = adata_c.obs["leiden"]
    elif cluster_method == "scran":
        counts_into_r(adata)
        ro.r("result <- scran::quickCluster(counts, BPPARAM = bppar)")
        clusters = ro.globalenv["result"]

    ro.globalenv["clusters"] = ro.StrVector(clusters)
    counts_into_r(adata)
    adata.obs["size_factors"] = ro.r(
        """scuttle::pooledSizeFactors(counts, clusters = clusters, BPPARAM = bppar)"""
    )
    adata.layers["raw"] = adata.X
    normalized = adata.X / adata.obs["size_factors"].values.reshape((-1, 1))
    adata.X = sparse.csc_array(sc.pp.log1p(normalized))
