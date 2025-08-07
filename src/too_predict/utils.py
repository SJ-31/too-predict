#!/usr/bin/env python

import importlib.resources as res
import itertools
import pickle
from collections.abc import Callable, Sequence
from datetime import date, datetime
from functools import reduce, wraps
from pathlib import Path
from typing import Literal

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.spatial.distance as spd
import yaml
from numba import jit
from numpy.random import Generator
from pyhere import here
from rpy2.rinterface_lib.sexp import (
    NACharacterType,
    NAComplexType,
    NAIntegerType,
    NALogicalType,
    NARealType,
)
from scipy import sparse, stats
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import too_predict

NA_TYPES: set = {
    NACharacterType,
    NAIntegerType,
    NALogicalType,
    NARealType,
    NAComplexType,
}
SPARSE_CHUNK_SIZE = 100_000

RANDOM_STATE: int = 9874  # Last modified <2025-03-05 Wed>
# Use for CV splitters

RNG: Generator = np.random.default_rng(297)  # Last modified [2025-03-25 Tue]
# Use for any relevant estimators


def train_test_split_ad(adata: ad.AnnData, **kwargs) -> tuple[ad.AnnData, ad.AnnData]:
    splitter = ShuffleSplit(n_splits=1, **kwargs)
    train, test = next(splitter.split(np.zeros(adata.shape)))
    return adata[train, :], adata[test, :]


def get_data(path: str, must_exist: bool = True) -> Path:
    """Retrieve the path of a file in this package's `data` directory
    :param: path relative path to the desired file
    """
    root = res.files(too_predict)
    file = root.parent.parent.joinpath("data").joinpath(path)
    if must_exist and not file.exists():
        raise FileNotFoundError(f"{path} doesn't exist!")
    return file.absolute()


def xarray_if_sparse(
    x: ad.AnnData, copy: bool = True, dtype: np.dtype = np.float32
) -> np.ndarray:
    was_sparse: bool = sparse.issparse(x.X)
    if was_sparse:
        arr = x.X.toarray()
    elif copy:
        arr = x.X.copy()
    else:
        arr = x.X
    return arr.astype(dtype)


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


def symbol2ensembl(as_df: bool = False) -> dict | pd.DataFrame:
    df = pd.read_csv(get_data("mappings/ensembl_113_id_mapping.tsv"), sep="\t")
    sel = df.loc[:, ["symbol", "ensembl"]].dropna(subset="symbol").drop_duplicates()
    if as_df:
        return sel
    return {k: v for k, v in zip(sel["symbol"], sel["ensembl"])}


def add_gc_content(adata: ad.AnnData, id_col: str = "GENEID") -> None:
    mapping = pd.read_csv(get_data("mappings/ensembl2gc_content.tsv"), sep="\t").rename(
        {"Gene % GC content": "gc_content"}, axis=1
    )
    mapping.loc[:, "gc_content"] = mapping["gc_content"] / 100
    adata.var = adata.var.merge(
        mapping, left_on=id_col, right_on="Gene stable ID", how="left"
    )


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


def gs_internal(meta: bool = False) -> dict | pd.DataFrame:
    if not meta:
        with open(get_data("reference/gene_sets_custom.yaml"), "r") as f:
            data = yaml.safe_load(f)
            return {
                k: set(v) if not isinstance(v, str) else {v} for k, v in data.items()
            }
    return pd.read_csv(get_data("reference/gene_sets_custom_meta.tsv"), sep="\t")


def cell_markers_internal(
    meta: bool = False, file_only=False
) -> pd.DataFrame | dict | Path:
    file = (
        get_data("reference/cell_markers_custom_meta.tsv")
        if meta
        else get_data("reference/cell_markers_custom.yaml")
    )
    if file_only:
        return file
    if meta:
        df = pd.read_csv(file, sep="\t")
        df.loc[:, "set_name"] = df["tissue"].combine(
            df["cell_type"], lambda x, y: f"{x}-{y}"
        )
        df: pd.DataFrame = df.groupby("set_name").agg(
            size=pd.NamedAgg(column="ensembl", aggfunc="count"),
            source=pd.NamedAgg(column="source", aggfunc="first"),
        )
        return df
    with open(file, "r") as f:
        dct = yaml.safe_load(f)
    return dct


def training_data_internal(
    label: str = "tumor_type",
    subset: bool = True,
    backed=None,
    as_dask: bool = False,
    subset_p: float = 0.4,  # Subset this proportion from every tumor type (excluding organoid samples)
) -> ad.AnnData:
    root: Path = res.files(too_predict)
    dir = (
        root.parent.parent.joinpath("remote")
        .joinpath("repos")
        .joinpath("too-predict")
        .joinpath("training")
    )
    combined_file = dir.joinpath("all_tumors_rnaseq.h5ad")
    subset_converted = str(subset_p).replace(".", "_")
    subset_file = dir.joinpath(
        f"all_tumors_rnaseq-{label}_subset-{subset_converted}.h5ad"
    )
    if subset:
        print(f"Subset of {label} at proportion {subset_p}")
    if subset and subset_file.exists():
        if as_dask:
            return adata_as_dask(subset_file)
        return ad.read_h5ad(subset_file, backed=backed)
    elif backed and subset and not subset_file.exists():
        raise ValueError(
            "`backed` must be set False for the initial creation of the subset file"
        )
    if not as_dask:
        adata: ad.AnnData = ad.read_h5ad(combined_file, backed=backed)
        # as of [2025-07-17 Thu], loading the whole thing into memory costs ~ 6.83 GB
    else:
        adata = adata_as_dask(combined_file)
    adata = adata[adata.obs["tumor_type"] != "Unknown", :]
    old_shape = adata.shape
    print(f"Initial shape: {old_shape}")
    min_samples = round(len(adata) * 0.1)  # Roughly at least 10% of samples
    if not backed:
        sc.pp.filter_cells(adata, min_counts=5000)
        sc.pp.filter_genes(adata, min_cells=min_samples)
        sc.pp.filter_genes(adata, min_counts=200)
    if label == "primary_site":
        adata.obs.loc[:, "primary_site"] = adata.obs["primary_site"].replace(
            {
                "bones_joints_and_articular_cartilage_of_limbs": "bones_joints_articular_cartilage",
                "bones_joints_and_articular_cartilage_of_other_and_unspecified_sites": "bones_joints_articular_cartilage",
                # The above includes muscle
                "oropharynx": "hypo_oropharynx",
                "hypopharynx": "hypo_oropharynx",
                "other_and_unspecified_parts_of_mouth": "mouth_tongue",
                "base_of_tongue": "mouth_tongue",
                "tonsil": "mouth_tongue",
                "gum": "mouth_tongue",
                "floor_of_mouth": "mouth_tongue",
                "lip": "mouth_tongue",
                "palate": "mouth_tongue",
                "other_and_ill_defined_sites_in_lip_oral_cavity_and_pharynx": "mouth_tongue",
                "other_and_ill_defined_sites": "unknown",
                "rectum": "colorectal",
                "colon": "colorectal",
                "rectosigmoid_junction": "colorectal",
                "uterus_nos": "uterus_ovary",
                "ovary": "uterus_ovary",
                "corpus_uteri": "uterus_ovary",
                "cervix_uteri": "uterus_ovary",
            }
        )
    if not backed:
        adata, discarded_types = filter_by_obs(adata, [label], min=50)
        print(f"Discarded {label}: {discarded_types}")
        print(f"Final shape after filtering: {adata.shape}")
        print(f"N genes removed: {old_shape[1] - adata.shape[1]}")
        print(f"N obs removed: {old_shape[0] - adata.shape[0]}")
    if subset and not subset_file.exists():
        organoids = adata[adata.obs["Sample_Type"] == "organoid", :]
        adata = adata[adata.obs["Sample_Type"] != "organoid", :]
        adata = preserving_sample(
            adata, key=label, fraction=subset_p, with_replacement=subset_p > 1
        )
        adata = ad.concat([organoids, adata], axis="obs", merge="same")
        adata.write_h5ad(subset_file)
    return adata


def adata_size_of(adata: ad.AnnData) -> None:
    print(f"{adata.__sizeof__() / 1e6:.3} MB")
    print(f"{adata.__sizeof__() / 1e9:.3} GB")


def adata_as_dask(path) -> ad.AnnData:
    with h5py.File(path, "r") as f:
        adata = ad.AnnData(
            obs=ad.io.read_elem(f["obs"]),
            var=ad.io.read_elem(f["var"]),
        )
        adata.X = ad.experimental.read_elem_as_dask(
            f["X"], chunks=(SPARSE_CHUNK_SIZE, adata.shape[1])
        )
    return adata


def training_data_internal_test(
    sample: float = 0.3,
    label: str = "tumor_type",
    minimal: bool = False,
    backed=None,
    as_dask: bool = False,
) -> ad.AnnData:
    if not minimal:
        if not as_dask:
            adata = ad.read_h5ad(
                get_data("tests/all_tumors_rnaseq_TEST.h5ad"), backed=backed
            )
        else:
            adata = adata_as_dask(get_data("tests/all_tumors_rnaseq_TEST.h5ad"))
        if not backed:
            sc.pp.subsample(adata, sample)
            sc.pp.filter_genes(adata, min_cells=10)
            sc.pp.filter_cells(adata, min_counts=5000)
            sc.pp.filter_genes(adata, min_counts=200)
            adata, discarded_types = filter_by_obs(adata, [label], min=20)
            print(f"Discarded {label}: {discarded_types}")
            print(f"Test data shape: {adata.shape}")
    else:
        adata = ad.read_h5ad(
            get_data("tests/all_tumors_rnaseq_TEST_MINIMAL.h5ad"), backed=backed
        )
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


def hugo_ref_internal() -> pd.DataFrame:
    file = get_data("hgnc_complete_set_2025-3-19.tsv")
    if file.exists():
        return pd.read_csv(file, sep="\t")
    raise ValueError("The data file doesn't exist!")


def get_blacklist_internal(
    name: str = "edgeR_median_lfc_feature_list_3000_ZERO.txt",
) -> list:
    file = get_data(f"output/feature_selection/blacklists/{name}")
    if file.exists():
        with open(file, "r") as z:
            zeros = z.read().strip().splitlines()
        return zeros
    raise ValueError(f"Blacklist {name} not found!")


def ref_feature_lists_internal(
    add_all: bool = True, remove_zeros: bool = False
) -> tuple[dict, dict]:
    features, refs = {}, {}
    fs_dir: Path = get_data("output/feature_selection")
    for i, (fname, add_to) in enumerate(
        zip(
            [fs_dir.joinpath("feature_lists"), fs_dir.joinpath("reference_lists")],
            [features, refs],
        )
    ):
        for file in fname.iterdir():
            if file.suffix != ".txt":
                continue
            with open(file, "r") as f:
                items = f.read().strip().splitlines()
            try:
                if i == 0 and (zeros := get_blacklist_internal()) and remove_zeros:
                    # Remove zero-importance features
                    items = list(set(items) - set(zeros))
            except ValueError:
                print("Zeros file not found")
            name = file.stem
            add_to[name] = items
    if add_all:
        features["all_features"] = None
    return refs, features


def record_in_yaml(file: str | Path, record_key: str = "last_ran") -> None:
    file = Path(file) if isinstance(file, str) else file
    with open(file, "w+") as f:
        date = datetime.today().strftime("%Y-%m-%d")
        loaded = yaml.safe_load(f) if file.stat().st_size > 0 else {}
        loaded[record_key] = date
        yaml.safe_dump(loaded, f)


def write_feat_ref_metadata():
    refs, features = ref_feature_lists_internal(False)
    fs_dir = here("data", "output", "feature_selection")
    hugo = hugo_ref_internal()
    var_df = pd.read_csv(here("data", "training_data_var.csv"))
    for dir, group in zip(
        [here(fs_dir, "feature_lists"), here(fs_dir, "reference_lists")],
        [features, refs],
    ):
        for k, v in group.items():
            current = var_df.loc[var_df["GENEID"].isin(v), :]
            with_hugo = current.merge(
                hugo, how="left", left_on="GENENAME", right_on="symbol"
            )
            with_hugo = with_hugo.loc[:, list(hugo.columns) + list(current.columns)]
            with_hugo.to_csv(here(dir, f"{k}_metadata.csv"), index=False)


def adata_sample_by(
    adata,
    label_spec: dict[str, list[tuple[str, int]]],
    rng: np.random.Generator = RNG,
    replace: bool = False,
) -> np.ndarray:
    """Split adata into train and test sets, then randomly draw training
    examples from train and add to test

    Parameters
    ----------
    label_spec : dictionary of
        column in adata.obs -> [(name of label class to draw, n_instances), ...]
        e.g. {"tumor_type" : [("CHOL", 5), ("LIHC", 8)],
              "Project_ID" : [("TARGET", 2)]}

    Returns
    -------
    ndarray of indices with the random selection
    """
    indices = []
    for label, targets in label_spec.items():
        candidates = adata.obs[label]
        for name, n in targets:
            choices = np.where(candidates == name)[0]
            if not replace and n >= len(choices):
                print(
                    f"WARNING: n for target {name} >= the n of target in training data. Taking all samples from trainin..."
                )
                n = len(choices)
            chosen = rng.choice(choices, size=n, replace=replace)
            indices.extend(chosen)
    return np.array(indices)


def split_and_sample(
    adata,
    split_fn,
    label_spec: dict[str, list[tuple[str, int]]],
    rng: np.random.Generator = RNG,
    replace: bool = False,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Split adata into train and test sets, then randomly draw training
    examples from train and add to test
    """
    train, test = split_fn(adata)
    if train.obs.index.isin(test.obs.index).any():
        raise ValueError("The samples in train, test overlap!")
    train_index = pd.Series(range(len(train)))
    indices = adata_sample_by(train, label_spec, rng, replace)
    from_train = train[indices, :]
    train = train[~train_index.isin(indices)]
    test = ad.concat([test, from_train], axis="obs", merge="same")
    return train, test


class RankInterpreter:
    """A class with methods to ease data access and interpretation of the results
    produced by scanpy.tl.rank_genes_groups
    """

    def __init__(self, adata: ad.AnnData, group=None, **kwargs) -> None:
        self.features = adata.var.index
        data = adata.uns["rank_genes_groups"]
        # Each group column is ordered independently of the others
        for k, v in data.items():
            if k != "params":
                self.__setattr__(k, pd.DataFrame(v))
            else:
                self.__setattr__(k, v)
        self.groups = self.names.columns
        self.data = sc.get.rank_genes_groups_df(adata, group=group, **kwargs)
        # attrs include 'params', 'pts' (optional), 'pts_rest' (optional),
        # 'names',
        # 'scores',
        # 'pvals',
        # 'pvals_adj',
        # 'logfoldchanges'

    def feature_stat(
        self,
        stat: str,
        feature_list=None,
        threshold: float = None,
        threshold_requirement: str = "any",
    ) -> pd.DataFrame:
        """Get a dataframe of features x group
        showing the values of each feature in `gene_list` for the given statistic.

        Parameters
        ----------
        threshold : float for adjusted p-value threshold
        """
        feature_list = self.features if feature_list is None else feature_list

        def helper(cur_stat) -> pd.DataFrame:
            if len(self.groups) > 1:
                reshaped = (
                    self.data.filter(items=["group", "names", cur_stat], axis=1)
                    .loc[self.data["names"].isin(feature_list), :]
                    .pivot(columns="group", index="names", values=cur_stat)
                )
            else:
                reshaped = self.data.filter(items=["names", cur_stat], axis=1).loc[
                    self.data["names"].isin(feature_list), :
                ]
                reshaped = reshaped.set_index(reshaped["names"]).drop("names", axis=1)
            return reshaped

        requested = helper(stat)
        if threshold is not None:
            pvals = helper("pvals_adj")
            if threshold_requirement == "any":
                mask = (pvals <= threshold).any(axis=1)
            else:
                mask = (pvals <= threshold).all(axis=1)
            requested = requested.loc[mask, :]
        return requested


def write_tmp_toolkit(
    adata: ad.AnnData, outfile: str | Path, symbol_col: str = "ENTREZ"
) -> None:
    adata = adata[:, ~adata.var[symbol_col].isna()]
    counts = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    sample_names = adata.obs["Project_ID"].combine(
        adata.obs.index.to_series(), lambda x, y: f"{x}_{y}"
    )
    counts = np.transpose(counts)
    df = pd.DataFrame(counts, columns=list(sample_names), index=adata.var[symbol_col])
    df.to_csv(outfile, index_label="Entrez_Gene_Id", sep="\t")


@jit(cache=True)
def pairwise_overlaps(sets: dict, as_matrix: bool = False) -> pd.Series | pd.DataFrame:
    combs = itertools.combinations(sets.keys(), 2)
    result = []
    for comb in combs:
        result.append(np.array([comb[0], comb[1], len(sets[comb[0]] & sets[comb[1]])]))
    overlaps = np.array(result)
    df = pd.DataFrame(overlaps, columns=["x", "y", "intersection"])
    df.loc[:, "intersection"] = df["intersection"].astype(np.int64)
    if as_matrix:
        return df
    series = df["intersection"]
    series.index = pd.MultiIndex.from_frame(df.loc[:, ["x", "y"]])
    return series


def quantify_overlap(sets: dict, method: str) -> pd.Series:
    if method == "overlap":
        result = pairwise_overlaps(sets, do_parallel=True).sort_values(ascending=False)
    else:
        df = (
            pd.DataFrame({"items": sets.values(), "values": True}, index=sets.keys())
            .explode("items")
            .pivot(columns="items")
            .fillna(False)
        )
        df.columns = df.columns.droplevel()
        dist = spd.squareform(spd.pdist(df, "jaccard"))
        long = (
            pd.DataFrame(dist, index=df.index, columns=df.index)
            .melt(ignore_index=False)
            .reset_index()
        )
        long.loc[:, "pair"] = long["index"].combine(
            long["variable"], lambda x, y: {x, y}
        )
        long = long.drop_duplicates(subset="pair").drop("pair", axis=1)
        result = pd.Series(
            long["value"],
            index=pd.MultiIndex.from_frame(long.loc[:, ["index", "variable"]]),
        ).sort_values(ascending=False)
    return result


def scanorama_correct(adata: ad.AnnData, batch_key: str, scale: bool = True, **kwargs):
    """Wrapper for Scanorama batch-effect correction

    Parameters
    ----------
    adata : combined anndata object. Counts should be log-normalized and scaled for
        best results
    scale : whether or not to scale data by batch
    batch_key : column of adata.obs containing batch information
    """
    import scanorama

    defaults = {
        "batch_size": 10,
        "verbose": 2,
        "sigma": 15,
        "alpha": 0.1,
        "knn": 20,
        "approx": True,  # Use approximate nearest neighbors, speeds up runtime
        "hvg": 3000,  # Use this many top hvgs
        "dimred": 100,
    }
    if scale and sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    defaults.update(kwargs)
    splits = []
    for b in adata.obs[batch_key].unique():
        current = adata[adata.obs[batch_key] == b, :].copy()
        if scale:
            scaler = StandardScaler(with_mean=True, with_std=True)  # necessary
            # due to bug in scanpy.pp.scale
            current.X = scaler.fit_transform(current.X)
        splits.append(current)
    corrected = scanorama.correct_scanpy(splits, **defaults)
    result = ad.concat(corrected, axis="obs", join="inner", merge="first")
    if not sparse.issparse(result.X):
        result.X = sparse.csc_array(result.X)
    return result


def preserving_sample(
    adata: ad.AnnData,
    key: str,
    fraction: float = 0.4,
    rng: np.random.Generator = RNG,
    with_replacement: bool = False,
) -> None | ad.AnnData:
    """Helper function for sampling from `adata` in a way that preserves the
    label distributions in `key`
    """
    if not with_replacement and fraction > 1:
        raise ValueError("with_replacement must be True if adding more samples!")
    new_counts: pd.Series = round(adata.obs[key].value_counts() * fraction, 0).astype(
        int
    )
    new_counts[new_counts == 0] = 1
    index_map: pd.Series = pd.Series(range(0, adata.shape[0]), index=adata.obs.index)
    wanted_indices = []
    for k, count in new_counts.items():
        current = adata[adata.obs[key] == k, :]
        wanted_indices.extend(
            rng.choice(current.obs.index, size=count, replace=with_replacement)
        )
    mapped: np.ndarray = index_map[wanted_indices].values
    return adata[mapped, :].copy()


def do_call(fn, pars: dict | None):
    if pars is None:
        return fn(**{})
    else:
        return fn(**pars)


def pca_to_leiden(
    adata, pca_pars=None, neighbor_pars=None, umap_pars=None, leiden_pars=None
):
    """Wrapper function for doing the standard PCA to leiden clustering in scanpy"""
    if "X_pca" not in adata.obsm:
        do_call(lambda **x: sc.pp.pca(adata, **x), pca_pars)
    do_call(lambda **x: sc.pp.neighbors(adata, **x), neighbor_pars)
    do_call(lambda **x: sc.tl.umap(adata, **x), umap_pars)
    do_call(lambda **x: sc.tl.leiden(adata, **x), leiden_pars)


def if_none(dct: dict | None, default: dict, update: bool = True) -> dict:
    """Return `default` if `dct` is None
    If `update`, use `dct` to update `default`, otherwise return `dct` as is
    """
    if dct is None:
        return default
    elif update:
        default.update(dct)
        return default
    return dct


def mad_outliers(
    adata: ad.AnnData,
    columns: str | Sequence[str] = "",
    mode: Literal["cells", "genes"] = "cells",
    n_mads: int | Sequence[int] = 5,
    boolean_mode: Literal["all", "any"] = "any",
    subset: dict[str, Sequence[str]] | None = None,
    mask: np.ndarray | None = None,
    col_added: str = "is_mad_outlier",
) -> None:
    """Remove cells/genes that differ by `n_mads`
    This function computes the median absolute deviation (MAD) of summed expression at
        the cell or gene level, then filters out cells/genes with
        absolute deviation from the median > (n_mads * MAD)

    Parameters
    ----------
    subset : only compute the mad from cells/genes satisfying the subset specification.
        is a dictionary mapping column names in adata.obs or adata.var to permitted
        values e.g. {"cell_type": ["macrophage", "fibroblast"]}
    mask : only compute the mad after applying the mask to adata
    """
    full = adata
    if subset is not None and mode == "cells":
        mask = reduce(
            lambda x, y: x & y, [adata.obs[k].isin(v) for k, v in subset.items()]
        )
        adata = adata[mask, :]
    if subset is not None:
        mask = reduce(
            lambda x, y: x & y, [adata.var[k].isin(v) for k, v in subset.items()]
        )
        adata = adata[:, mask]
    if mask is not None and mode == "cells":
        adata = adata[mask, :]
    elif mask is not None:
        adata = adata[:, mask]

    def get_mad(arr: pd.Series):
        abs_diff = np.abs(arr - np.median(arr))  # Absolute deviation from median
        mad = np.median(abs_diff)
        return abs_diff > mad * n_mads

    def reduce_fn(masks: list[np.ndarray]):
        if boolean_mode == "any":
            return reduce(lambda x, y: x | y, masks)
        else:
            return reduce(lambda x, y: x & y, masks)

    if isinstance(columns, str):
        columns = [columns]

    if mode == "cells":
        full.obs[col_added] = reduce_fn([get_mad(adata.obs[col]) for col in columns])
    else:
        full.var[col_added] = reduce_fn([get_mad(adata.var[col]) for col in columns])


class SaveOrLoad:
    """Decorator class to make single-use data-generating functions (e.g.
        analysis functions) more convenient

    Functions that it wraps are expected to write some output file as the first
    argument, which is passed by the decorator in `out`

    Parameters
    ----------
    out : Output file passed to the wrapped function, or a dictionary of output
        files

    read_fn : Function used to read `out`, if it exists. If it exists, then
        the wrapped function won't be run.
        For multiple outputs, you may also provide a dictionary with the same keys as
        `out` to handle outputs of different formats

    logdir : Path to a directory used to record the date of calls to the
        wrapped function, in ISO format

    TODO: capturing the stdout generated by fn could be useful
    """

    def _get_out(self, file: Path | str) -> Path:
        if isinstance(file, str):
            return Path(file)
        return file

    def __init__(
        self,
        out: Path | str | dict[str, Path | str],
        read_fn: Callable | dict[str, Callable],
        logdir: str | Path | None = None,
    ) -> None:
        if logdir is None:
            self.logdir: Path = Path()
        else:
            self.logdir = self._get_out(logdir)

        self.out: Path | dict[str, Path]
        if isinstance(out, dict):
            self.out = {k: self._get_out(v) for k, v in out.items()}
        else:
            self.out = self._get_out(out)

        self.read_fn: Callable = None
        self.read_fns: dict[str, Callable] = None
        if isinstance(read_fn, dict) and isinstance(self.out, dict):
            self.read_fns = read_fn
        elif isinstance(read_fn, dict):
            raise ValueError("Multiple read functions provided with only one output!")
        else:
            self.read_fn = read_fn

        self.date_str: str = date.today().isoformat()

    def __call__(self, fn) -> Callable:
        @wraps(fn)
        def wrapped(*args, **kwargs):
            log = self.logdir.joinpath(f"{self.date_str}_{fn.__name__}.log")
            if isinstance(self.out, Path) and self.out.exists():
                print("Reading existing file...")
                return self.read_fn(self.out)
            elif isinstance(self.out, dict) and all(
                v.exists() for v in self.out.values()
            ):
                print("Reading existing files...")
                if self.read_fn is not None:
                    return {k: self.read_fn(v) for k, v in self.out.items()}
                return {k: self.read_fns[k](v) for k, v in self.out.items()}
            value = fn(self.out, *args, **kwargs)
            log.write_text("Completed successfully")
            return value

        return wrapped


def xgb_complexity(model: XGBClassifier) -> pd.DataFrame:
    "Produce a dataframe with summary statistics about the properties of an XGBoost model"
    booster = model.get_booster()
    stats = {"stat": [], "value": []}
    for stat, val in zip(
        ["n_trees", "max_depth"], [booster.num_boosted_rounds(), model.max_depth]
    ):
        stats["stat"].append(stat)
        stats["value"].append(val)
    tmp = []
    for itype in ["weight", "gain", "cover", "total_gain", "total_cover"]:
        vals = booster.get_score(importance_type=itype)
        tmp.append(pd.DataFrame({itype: vals.values()}, index=vals.keys()))
    score_df: pd.DataFrame = reduce(
        lambda x, y: pd.merge(x, y, left_index=True, right_index=True), tmp
    )
    tmp = []
    for stat in ["min", "max", "std", "mean"]:
        current = score_df.agg(stat)
        current.name = stat
        tmp.append(current)
    stat_df = (
        reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), tmp)
        .reset_index(names="tmp")
        .melt("tmp")
    )
    stat_df = (
        stat_df.assign(stat=stat_df["tmp"] + "_" + stat_df["variable"])
        .drop(["tmp", "variable"], axis=1)
        .loc[:, ["stat", "value"]]
    )
    return pd.concat([pd.DataFrame(stats), stat_df]).sort_values("stat")
