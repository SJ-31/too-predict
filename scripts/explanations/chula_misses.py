#!/usr/bin/env ipython

from pathlib import Path
from typing import Callable

import anndata as ad
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import too_predict.explanation as ex
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from pyhere import here
from too_predict._train_utils import ADDITIONAL_SPLITS, MODELS, read_model_spec
from too_predict.evaluation import get_all_metrics, write_cross_val, write_metrics
from too_predict.filter import Filter
from too_predict.model import PredBase
from too_predict.plotting import plot_diagonal_matrix, plot_instance_dist
from too_predict.utils import (
    RNG,
    read_existing,
    split_and_sample,
    training_data_internal,
    training_data_internal_test,
)

OUTDIR: Path = here("data", "output", "explanations", "chula_misses")
OUTDIR.mkdir(exist_ok=True, parents=True)
STORAGE: Path = here("remote", "repos", "too-predict", "explanations")
STORAGE.mkdir(exist_ok=True, parents=True)

# * Feature importance functions


def get_shap_explanations(explainer: ex.Exp, outdir, set_name, dataset_type):
    val, _ = explainer.shap(
        explain_fn=lambda x: shap.TreeExplainer(x),
        summary_plot=True,
        plot_feature_col="GENENAME",
        plot_directory=here(outdir.joinpath(f"{set_name}_plots")),
    )
    return val


def get_anchors(explainer: ex.Exp, outdir, set_name, dataset_type):
    val, metrics = explainer.anchor()
    metrics.loc[:, "set"] = dataset_type
    metrics.to_csv(here(outdir.joinpath(f"anchor-{set_name}.csv")), index=False)
    return val


# * Higher-level functions


def helper(
    importance_getter: Callable[[ex.Exp, str, str, str], ad.AnnData],
    out: str,
    model: PredBase,
    train: ad.AnnData,
    test: ad.AnnData,
    label_col: str,
    set_name: str,
    n: int = 10,
):
    outdir = OUTDIR.joinpath(out)
    exp = ex.Exp(model, feature_col="GENEID", label_col=label_col)

    def get_test(f):
        exp.fit(test)
        val = importance_getter(exp, outdir, set_name, "test")
        val.write_h5ad(f)

    def get_train(f):
        exp.fit(train)
        val = importance_getter(exp, outdir, set_name, "train")
        val.obs.loc[:, "dataset"] = set_name
        val.write_h5ad(f)

    train_out = here(STORAGE.joinpath(f"{out}{set_name}_train.h5ad"))
    test_out = here(STORAGE.joinpath(f"{out}{set_name}_test.h5ad"))
    test_vals = read_existing(test_out, get_test, ad.read_h5ad)
    train_vals = read_existing(train_out, get_train, ad.read_h5ad)

    neg_contrib = plot_save_helper(
        prefix=out,
        n=n,
        train_imp=train_vals,
        test_imp=test_vals,
        label_col=label_col,
        set_name=set_name,
        outdir=outdir,
    )

    return neg_contrib


def plot_save_helper(
    prefix: str,
    n: int,
    train_imp: ad.AnnData,
    test_imp: ad.AnnData,
    label_col,
    set_name,
    outdir,
):
    interpreter = ex.ExpInterpreter(
        train_importances=train_imp, test_importances=test_imp, label_col=label_col
    )
    neg_contrib, per_label = interpreter.neg_contributions(prefix, n=n)
    plotdir = outdir.joinpath(set_name)
    plotdir.mkdir(exist_ok=True, parents=True)

    train_mat = interpreter.label_distances(prefix, dataset="train", square=True)
    fig, ax = plt.subplots()
    plot_diagonal_matrix(train_mat, ax, cmap="coolwarm")
    fig.savefig(plotdir.joinpath(f"{set_name}_train_dist.png"))

    test_mat = interpreter.label_distances(prefix, dataset="test", square=True)
    fig, ax = plt.subplots()
    plot_diagonal_matrix(test_mat, ax, cmap="coolwarm")
    fig.savefig(plotdir.joinpath(f"{set_name}_test_dist.png"))
    compare_mats = interpreter.instance_distances(prefix, dataset="compare")

    for label, m in compare_mats.items():
        fig, ax = plt.subplots()
        plot_instance_dist(m, ax)
        ax.set(title=f"{label_col}: {label}")
        fig.savefig(plotdir.joinpath(f"{set_name}-{label}_train_test.png"))
    return neg_contrib


def remove_zero_features(
    adata, filter: Filter, model: PredBase, splitter: Callable
) -> tuple[ad.AnnData, ad.AnnData]:
    train, _ = splitter(adata)
    model.fit(train)
    new_filter = filter.copy()
    new_filter.from_feature_importance(model)
    zeros_removed = new_filter.fit_transform(adata)
    train2, test2 = splitter(zeros_removed)
    model.fit(train2)
    return train2, test2


# * Main


def main(args):
    if args.test:
        adata = training_data_internal_test()
    else:
        adata = training_data_internal()
    ignored = args.ignore.split(",")

    spec = MODELS["clr_xgboost_edger"]
    F, M, T, B, E = read_model_spec(spec)

    adata = F.fit_transform(adata)
    adata = T.fit_transform(adata)
    adata.X = adata.X.toarray()

    label_col = args.label_class
    for name, fn in ADDITIONAL_SPLITS.items():
        _, test = fn(adata)
        unique_values = test.obs[label_col].unique()
        spec = {label_col: [(u, 5) for u in unique_values]}
        train, test = remove_zero_features(
            adata,
            F,
            model=M,  # will get fit
            splitter=lambda x: split_and_sample(x, fn, spec, RNG),
        )
        for getter, method in [
            (get_shap_explanations, "shap_"),
            (get_anchors, "anchor_"),
        ]:
            if method in ignored:
                print(f"Ignoring method {method}")
                continue
            for n in [20, 10, 5]:
                n_outdir = OUTDIR.joinpath(method).joinpath(str(n))
                n_outdir.mkdir(exist_ok=True)
                neg_contrib = helper(
                    importance_getter=getter,
                    out=method,
                    model=M,
                    train=train,
                    test=test,
                    label_col=label_col,
                    set_name=name,
                    n=n,
                )
                proba = M.predict_proba(test)
                perf = get_all_metrics(test.obs[label_col], proba, M.classes_)
                before = n_outdir.joinpath(f"{name}_before.txt")
                write_metrics(before, perf)

                # Try hard masking the negatively-contributing features
                test.X[:, test.var["GENEID"].isin(neg_contrib)] = 0
                proba = M.predict_proba(test)
                perf_after = get_all_metrics(test.obs[label_col], proba, M.classes_)
                after = n_outdir.joinpath(f"{name}_after.txt")
                write_metrics(after, perf_after)


# * CLI


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-c", "--cores", default=8)
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-i", "--ignored", default="")
    parser.add_argument("-d", "--dask", default=False, action="store_true")
    parser.add_argument("--no_shap", default=False, action="store_true")
    parser.add_argument("--no_morris", default=False, action="store_true")
    parser.add_argument("-l", "--label_class", default="tumor_type")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dask:
        cluster = SLURMCluster(cores=int(args.cores), memory=f"{args.memory} GB")
        client = Client(cluster)
    backend = "dask" if args.dask else "loky"
    par_args = {"wait_for_workers_timeout": 0} if args.dask else {"n_jobs": args.cores}
    with joblib.parallel_backend(backend, **par_args):
        main(args)
