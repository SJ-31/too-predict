#!/usr/bin/env ipython

import importlib.resources as res
import itertools
import math
import pickle
from functools import reduce
from pathlib import Path
from typing import Callable

import anndata as ad
import anndata2ri
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scanpy as sc
from pyhere import here
from rpy2 import rinterface, rinterface_lib
from rpy2.rinterface_lib.sexp import (
    NACharacterType,
    NAComplexType,
    NAIntegerType,
    NALogicalType,
    NARealType,
)
from rpy2.robjects import RObject, numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
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

RANDOM_STATE: int = 9874  # Last modified <2025-03-05 Wed>
# Use for CV splitters

RNG = np.random.RandomState(297)  # Last modified <2025-03-05 Wed>
# Use for any relevant estimators


def register_biocparallel(workers: int, param="MulticoreParam") -> None:
    ro.r("library(BiocParallel)")
    ro.r(f"register({param}(workers = {workers}))")


def get_data(path: str) -> Path:
    """Retrieve the path of a file in this package's `data` directory
    :param: path relative path to the desired file
    """
    with res.path(too_predict) as root:
        file = root.parent.parent.joinpath("data").joinpath(path)
        if file.exists():
            return file.absolute()
        raise FileNotFoundError(f"{path} doesn't exist!")


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


def np_to_r(arr: np.ndarray, r_symbol: str = ""):
    np_cv_rules = ro.default_converter + numpy2ri.converter
    with np_cv_rules.context():
        converted = ro.conversion.get_conversion().py2rpy(arr)
        if r_symbol:
            ro.globalenv[r_symbol] = converted
        else:
            return converted


def df_to_r(df: pd.DataFrame, r_symbol: str = ""):
    with (ro.default_converter + pandas2ri.converter).context():
        converted = ro.conversion.get_conversion().py2rpy(df)
        if r_symbol:
            ro.globalenv[r_symbol] = converted
        else:
            return converted


def df_from_r(robj) -> pd.DataFrame:
    with (ro.default_converter + pandas2ri.converter).context():
        ro.globalenv["df_from_r_tmp"] = robj
        if ro.r("class(df_from_r_tmp)")[0] not in {"data.frame"}:
            raise ValueError("The given object is not a data.frame")
        ro.r("rm(df_from_r_tmp)")
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


def filter_by_obs(
    adata: ad.AnnData, keys: list[str], min: int = 0, max: int = np.inf
) -> tuple[ad.AnnData, dict[str, set]]:
    """Filter adata by a category/factor column `key` in adata.obs

    Removes levels of the `key` that do not meet the min and max requirements

    Parameters
    ----------
    min : level must have at least this many observations to be kept
    max : level must have at most this many observations to be kept

    Returns
    -------
    A tuple of [filtered adata, list of discarded levels]

    """
    counts = [adata.obs[key].value_counts() for key in keys]
    count_mask = [((c >= min) & (c < max)) for c in counts]
    to_discard = {
        k: c.index[~c].to_series().to_list() for k, c in zip(keys, count_mask)
    }
    masks = [~adata.obs[k].isin(d) for k, d in to_discard.items()]

    mask = reduce(lambda x, y: x & y, masks).values
    return adata[mask, :], to_discard


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


def adata_x_to_r(adata: ad.AnnData, r_symbol: str = "", layer=None):
    if layer is not None:
        counts = adata.layers[layer]
    else:
        counts = adata.X
    if not isinstance(counts, np.ndarray):
        counts = counts.toarray()
    ro.globalenv["tmp_colnames"] = ro.StrVector(adata.obs.index)
    ro.globalenv["tmp_rownames"] = ro.StrVector(adata.var.index)
    np_to_r(np.transpose(counts), r_symbol="tmp_mat")
    ro.r("rownames(tmp_mat) <- tmp_rownames")
    ro.r("colnames(tmp_mat) <- tmp_colnames")
    ro.r("rm(tmp_colnames)")
    ro.r("rm(tmp_rownames)")
    if r_symbol:
        ro.globalenv[r_symbol] = ro.r("tmp_mat")
        ro.r("rm(tmp_mat)")
    else:
        return ro.globalenv["tmp_mat"]


def source(r_script: str, in_r=False) -> STAP | None:
    """Import `r script` in src/R as a STAP"""
    with res.path(too_predict) as root:
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
def add_gene_metadata(
    adata: ad.AnnData,
    keycol: str = "",
    columns=("GENENAME", "GENEID", "GENEBIOTYPE", "SEQNAME", "SEQLENGTH"),
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
        id_map = pd.read_csv(get_data("ensembl_113_id_mapping.tsv"), sep="\t")

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
        return ad_passed, ad_failed
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
    # print(get_star_strandedness(name, df))
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
    case_cols: tuple = ("primary_site", "disease_type"),
    case_id_col: str = "submitter_id",
    use_dask=True,
    verbose: bool = False,
) -> ad.AnnData:
    if not sample_sheet:
        sample_sheet = str(next(Path(dir).glob("gdc_sample_sheet*")))
    samples: pd.DataFrame = pd.read_csv(sample_sheet, sep="\t").rename(
        columns=lambda x: x.replace(" ", "_")
    )
    missing: set = set(samples["File_ID"])
    not_found: list = []
    count_dfs = []
    for d, p in zip(samples["File_ID"], samples["File_Name"]):
        try:
            cur = read_gdc_counts(Path(dir).joinpath(d).joinpath(p), d, count_col)
            if verbose:
                print(f"Reading {dir} successful")
                print(cur)
            if d in missing:  # Account for multiple samples from same case
                missing.remove(d)
            count_dfs.append(cur)
        except FileNotFoundError:
            print(f"WARNING: File in directory {d} not found")
            not_found.append(d)
    if verbose:
        n_not_found = len(samples["File_ID"])
        print(f"Number of directories not found: {n_not_found}")
        print(not_found)
        print()
    joined = reduce(
        lambda x, y: x.join(y, on=["gene_id", "gene_name"], how="outer"), count_dfs
    )
    print(f"Shape of joined dfs: {joined.shape}")
    samples = samples.loc[~samples["Case_ID"].isin(missing), :]
    var_df = joined.index.to_frame(index=False)
    if case_table:
        cases: pd.DataFrame = pd.read_csv(case_table, sep="\t")
        cases = cases.loc[:, [case_id_col] + list(case_cols)].drop_duplicates()
        samples = samples.merge(
            cases, how="left", left_on="Case_ID", right_on=case_id_col
        ).drop(case_id_col, axis="columns")
    return ad.AnnData(X=np.transpose(joined.values), obs=samples, var=var_df)


def read_existing[T](
    filename: Path,
    expr: Callable[[Path], T],
    read_fn: Callable[[Path], T] | None = None,
) -> T | None:
    if filename.exists() and read_fn is not None:
        return read_fn(filename)
    elif filename.exists():
        return
    else:
        return expr(filename)


def phi_proportionality(x: np.ndarray, y: np.ndarray):
    """Calculate Goodness of Fit to Proportionality []
    Parameters
    ----------
    x, y : Ideally log-ratio transformed data e.g. CLR

    Return
    ------
    Phi(log x, log y)
    The Phi statistic, which is zero when x and y are perfectly proportional
    The closer x and y are to zero, the stronger the proportionality
    """
    log_x = np.log(x)
    return np.var(log_x - np.log(y)) / np.var(log_x)


def adata_from_df(
    df: pd.DataFrame,
    labels: list = None,
    label_col: str = "label",
    var_col: str = "feature",
) -> ad.AnnData:
    adata = ad.AnnData(X=df, var=pd.DataFrame({var_col: df.columns}, index=df.columns))
    if labels is not None and len(labels) == df.shape[0]:
        adata.obs = pd.DataFrame({label_col: labels})
    return adata


def adata_to_df(adata: ad.AnnData, var_col: str = "feature"):
    if not isinstance(adata.X, np.ndarray):
        counts = adata.X.toarray()
    else:
        counts = adata.X
    return pd.DataFrame(counts, columns=adata.var[var_col], index=None)


def into_pseudobulks(adata: ad.AnnData, id_col: str, how="sum") -> ad.AnnData:
    aggregated = sc.get.aggregate(adata, by=id_col, func=[how])
    return ad.AnnData(X=aggregated.layers[how])


def cluster_gini(adata: ad.AnnData, clusters, label_col: str):
    """Calculate Gini index of each cluster with respect to a specific
    label

    Parameters
    ----------
    clusters : either a column in adata.obs containing cluster assignments, or array-like
        with the assignments

    Returns
    -------
    dictionary mapping cluster to its impurity, and the impurity of the entire cluster

    Notes
    -----
    The gini index ranges from 0 - 1; the closer to 0, the purer the cluster
    """
    assignments = adata.obs[clusters] if isinstance(clusters, str) else clusters
    n_samples: int = len(adata)
    ginis = []

    def gini_one(cluster):
        current = adata[np.where(assignments == cluster)]
        n = len(current)
        label_freq = (current.obs[label_col].value_counts() / n).values
        gini = np.sum(label_freq * (1 - label_freq))
        ginis.append(gini)
        return gini * (n / n_samples)

    unique = set(clusters)
    whole_gini = sum(map(gini_one, unique))
    return {k: v for k, v in zip(unique, ginis)}, whole_gini


def find_confounded(x, y) -> list[tuple[str, str]]:
    """Return pairs where all instances of some class in set x
    can only be found in a single instance of a class in set y

    Returns
    -------
    A list of (x_i, y_i) where instance x_i can only be found with y_i
    """
    tabulated: pd.DataFrame = pd.crosstab(x, y)
    problem_pairs = []
    for x_i in tabulated.index:
        if len(indices := np.where(tabulated.loc[x_i, :] > 0)[0]) == 1:
            value = tabulated.columns[indices][0]
            problem_pairs.append((x_i, value))
    return problem_pairs


def training_data_internal(label: str = "tumor_type") -> ad.AnnData:
    public_data = here("remote", "public_data")
    combined_file = here(public_data, "all_tumors_rnaseq.h5ad")
    adata: ad.AnnData = ad.read_h5ad(combined_file)
    adata = adata[adata.obs["tumor_type"] != "Unknown", :]
    old_shape = adata.shape
    print(f"Initial shape: {old_shape}")
    min_samples = round(len(adata) * 0.1)  # Roughly at least 10% of samples
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_genes(adata, min_cells=min_samples)
    sc.pp.filter_genes(adata, min_counts=200)
    adata, discarded_types = filter_by_obs(adata, [label], min=50)
    print(f"Discarded {label}: {discarded_types}")
    print(f"Final shape after filtering: {adata.shape}")
    print(f"N genes removed: {old_shape[1] - adata.shape[1]}")
    print(f"N obs removed: {old_shape[0] - adata.shape[0]}")
    return adata


def training_data_internal_test(
    sample: float = 0.3, label: str = "tumor_type"
) -> ad.AnnData:
    adata = ad.read_h5ad(get_data("tests/all_tumors_rnaseq_TEST.h5ad"))
    sc.pp.subsample(adata, sample)
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_genes(adata, min_counts=200)
    adata, discarded_types = filter_by_obs(adata, [label], min=10)
    print(f"Discarded {label}: {discarded_types}")
    print(f"Test data shape: {adata.shape}")
    return adata


def write_pickle(obj, filename) -> None:
    with open(filename, "wb") as pck:
        pickle.dump(obj, pck, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, "rb") as pck:
        obj = pickle.load(pck)
    return obj


def take_from_ad(
    x: ad.AnnData,
    y: ad.AnnData,
    keymap: list[tuple],
    read_prefix: str = "",
    write_prefix: str = "",
) -> list[tuple]:
    """File `x` with objects from `y` according to keymap"""
    missing: list = []
    for k, v in keymap:
        group = getattr(y, k)
        if (rname := f"{read_prefix}{v}") in group:
            value = group.get(rname)
            getattr(x, k)[f"{write_prefix}{v}"] = value
        else:
            missing.append((k, v))
    return missing


def ref_feature_lists_internal() -> tuple[dict, dict]:
    features, refs = {}, {}
    fs_dir = here("data", "output", "feature_selection")
    for fname, add_to in zip(
        [here(fs_dir, "feature_lists"), here(fs_dir, "reference_lists")],
        [features, refs],
    ):
        for file in fname.iterdir():
            with open(file, "r") as f:
                items = f.read().strip().splitlines()
            name = file.stem
            add_to[name] = items
    features["all_features"] = None
    return refs, features


def comb_pair_at(j, query, n=None) -> tuple[int, int]:
    """


    TODO: write this in rust, accumulate and the loop might get unwieldy
    could there also be an analytic way of calculating `f_offset`?
    """
    n = math.comb(j, 2) if n is None else n
    first: int = j - 2  # Placeholder
    if query >= n:
        raise ValueError("The query is too large!")
    f_offset: int = 0
    first_cutoffs = itertools.accumulate((j - 1 - i for i in range(j - 1)))
    previous = 0
    for index, acc in enumerate(first_cutoffs):
        if query < acc:
            first = index
            f_offset = previous
            break
        if index > 0:
            previous = acc
    second = query - f_offset + first + 1
    return (first, second)
