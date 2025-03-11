#!/usr/bin/env ipython
from pathlib import Path

import anndata as ad
import joblib
import matplotlib.pyplot as plt
from pyhere import here
from too_predict.filter import count_tomek_links
from too_predict.plotting import plot_diagonal_matrix
from too_predict.transformer import IMPLEMENTED_TRANSFORMATION
from too_predict.utils import (
    ref_feature_lists_internal,
    training_data_internal,
    training_data_internal_test,
)

OUTDIR = here("data", "output", "find_overlapping")
OUTDIR.mkdir(exist_ok=True)
STORAGE_DIR = here("remote", "repos", "too-predict", "normalization_comparison")
REF_LISTS, FEATURE_LISTS = ref_feature_lists_internal()
FEATURE_LISTS["all_features"] = None
NORMALIZATION_METHODS = IMPLEMENTED_TRANSFORMATION
IMPUTATION_METHODS = ["plus_one"]


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-d", "--dask", default=False, action="store_true")
    parser.add_argument("-c", "--cores", default=8, type=int)
    return parser.parse_args()


def helper(feature_set, i: str, n: str) -> ad.AnnData | None:
    dir = here(STORAGE_DIR, feature_set)
    output = here(dir, f"{n}-{i}.h5ad")
    outdir: Path = here(OUTDIR, feature_set)
    outdir.mkdir(exist_ok=True)
    paired_file = outdir.joinpath(f"{i}_{n}_tomek_link_pairs.csv")
    counts_file = outdir.joinpath(f"{i}_{n}_tomek_links.csv")
    matrix_file = outdir.joinpath(f"{i}_{n}_tomek_matrix.csv")
    matrix_png = outdir.joinpath(f"{i}_{n}_tomek_matrix.png")
    if not output.exists():
        print(f"{output} isn't finished yet, ignoring")
    else:
        for v in ["tumor_type", "primary_site"]:
            adata = ad.read_h5ad(output)
            class_counts, paired_counts, mat = count_tomek_links(adata, v)
            class_counts.to_csv(counts_file, index=False)
            paired_counts.to_csv(paired_file, index=False)
            mat.to_csv(matrix_file)
            fig, ax = plt.subplots()
            plot_diagonal_matrix(mat, ax, cmap="OrRd")
            fig.savefig(matrix_png, bbox_inches="tight")


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        print("Using test subset")
        adata = training_data_internal_test()
        OUTDIR = OUTDIR.joinpath("test")
        OUTDIR.mkdir(exist_ok=True, parents=True)
    else:
        adata = training_data_internal()
    backend = "dask" if args.dask else "loky"
    par_args = {"wait_for_workers_timeout": 0} if args.dask else {"n_jobs": args.cores}
    with joblib.parallel_backend(backend, **par_args):
        for fname, lst in FEATURE_LISTS.items():
            for i in IMPUTATION_METHODS:
                for n in NORMALIZATION_METHODS:
                    if "alr" in n:  # Temporary
                        continue
                    else:
                        helper(fname, i, n)
