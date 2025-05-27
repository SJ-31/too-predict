#!/usr/bin/env ipython
from datetime import datetime
from pathlib import Path

import anndata as ad
import joblib
import pandas as pd
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from pyhere import here
from scanpy import read_h5ad
from too_predict._train_utils import (
    ADDITIONAL_SPLITS,
    MODELS,
    read_model_spec,
)
from too_predict.utils import (
    RANDOM_STATE,
    training_data_internal,
    training_data_internal_test,
    write_pickle,
)

OUTDIR: Path = here("data", "output", "cross_validation")
DO_CV: bool = True
K: int = 10


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-c", "--cores", default=8)
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-n", "--no_cv", default=False, action="store_true")
    parser.add_argument("-d", "--dask", default=False, action="store_true")
    parser.add_argument("-a", "--cached", default=False, action="store_true")
    parser.add_argument("-l", "--label_class", default="tumor_type")
    return parser.parse_args()


def get_debug_file():
    file = datetime.now().strftime("%y-%m-%d_%H:%M")
    file = here("remote", "repos", "too-predict", f"cross_val_debug_{file}.pkl")
    return file


STORAGE_DIR = here("remote", "repos", "too-predict", "normalization_comparison")
LABEL_CLASSES = ["tumor_type", "primary_site"]
GROUP_CLASSES = ["Project_ID"]
USE_CACHED: bool = True

ADATA: ad.AnnData


def cross_validate_helper(
    lc,
    gc,
    spec: dict,
    result_dir_str,
    trans,
    impute,
    feature_set,
    skip: bool = False,
    which: tuple = ("CV", "additional"),
):
    if skip:
        return
    adata = ADATA.copy()
    filter, model, transformer, balancer, encoder, corrector = read_model_spec(spec)
    print(f"{spec=}")
    print(f"{model=}")
    try:
        if gc is not None:
            result_dir: Path = here(OUTDIR, f"{result_dir_str}_by_group_{gc}")
        else:
            result_dir: Path = here(OUTDIR, result_dir_str)
        result_dir.mkdir(exist_ok=True, parents=True)
        if feature_set is not None:
            dir = STORAGE_DIR.joinpath(feature_set)
        else:
            dir = STORAGE_DIR
        output = here(dir, f"{trans}-{impute}.h5ad")

        if USE_CACHED:
            adata = read_h5ad(output)
        else:
            if encoder is not None:
                adata = encoder.fit_transform(adata)  # TODO: this can leak data
                #
            if filter is not None:
                adata = filter.fit_transform(adata)
            if corrector is not None:
                adata.obs["is_organoid"] = adata.obs["Sample_Type"] == "organoid"
                adata = corrector.fit_transform(
                    adata
                )  # [2025-04-30 Wed] NOTE: temporary
                # Should apply the correction during validation
        if not here(result_dir, f"{lc}-misc.csv").exists() and DO_CV and "CV" in which:
            track_meta = result_dir.joinpath(".metadata")
            track_meta.mkdir(exist_ok=True)
            results = model.cross_validate(
                adata,
                label_col=lc,
                group_col=gc,
                random_state=RANDOM_STATE,
                n_splits=K,
                transformer=transformer,
                corrector=None,  # [2025-04-30 Wed] NOTE: temporary
                record_dir=track_meta,
            )
            write_results(results, result_dir, lc, cm_prefix="fold_")
        result_dir2 = result_dir.joinpath("additional_splits")
        result_dir2.mkdir(exist_ok=True, parents=True)
        if not here(result_dir2, f"{lc}-misc.csv").exists() and "additional" in which:
            results2 = model.holdout(
                adata,
                ADDITIONAL_SPLITS,
                label_col=lc,
                balancer=balancer,
                transformer=transformer,
                corrector=None,  # [2025-04-30 Wed] NOTE: temporary
            )
            write_results(results2, result_dir2, lc)
    except ValueError as e:
        print(f"ValueError encountered with params {spec}")
        print(e)
        print("Writing pckl...")
        write_pickle(data, get_debug_file())


def write_results(results, result_dir, label_col, cm_prefix: str = ""):
    for name, item in results.items():
        if name != "cm" and isinstance(item, pd.DataFrame):
            item.to_csv(result_dir.joinpath(f"{label_col}-{name}.csv"), index=False)
        elif name == "cm":
            for lab, cm in item.items():
                cm.to_csv(
                    result_dir.joinpath(f"{label_col}-{name}_cm-{cm_prefix}{lab}.csv")
                )


if __name__ == "__main__":
    args = parse_args()
    label_class = args.label_class
    DO_CV = not args.no_cv
    if args.test:
        print("Using test subset")
        ADATA = training_data_internal_test(label=label_class)
        OUTDIR = OUTDIR.joinpath("test")
        OUTDIR.mkdir(exist_ok=True, parents=True)
        MODELS = {k: v for k, v in MODELS.items() if k == "clr_random_forest_minfo"}
    else:
        ADATA = training_data_internal(label=label_class)
        OUTDIR.mkdir(exist_ok=True, parents=True)
    if args.dask:
        cluster = SLURMCluster(cores=int(args.cores), memory=f"{args.memory} GB")
        client = Client(cluster)
    backend = "dask" if args.dask else "loky"
    par_args = {"wait_for_workers_timeout": 0} if args.dask else {"n_jobs": args.cores}
    USE_CACHED = args.cached

    with joblib.parallel_backend(backend):
        for name, data in MODELS.items():
            t, i, skip, f, which = (
                data.get("t"),
                data.get("i"),
                data.get("s"),
                data.get("f"),
                data.get("w", ("CV", "additional")),
            )
            cross_validate_helper(
                lc=label_class,
                gc=None,
                spec=data,
                result_dir_str=name,
                trans=t,
                impute=i,
                feature_set=f,
                skip=skip,
                which=which,
            )
