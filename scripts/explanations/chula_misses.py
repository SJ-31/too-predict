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
from too_predict.plotting import plot_diagonal_matrix
from too_predict.utils import (
    RNG,
    split_and_sample,
    training_data_internal,
    training_data_internal_test,
)

OUTDIR: Path = here("data", "output", "explanations", "chula_misses")
OUTDIR.mkdir(exist_ok=True, parents=True)
STORAGE: Path = here("remote", "repos", "too-predict")


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


def anchor_helper(
    model: PredBase,
    train: ad.AnnData,
    test: ad.AnnData,
    label_col: str,
    set_name: str,
    n: int = 10,
):
    # [2025-04-01 Tue] Use anchors to identify which features are responsible
    # for causing consistent misclassifications
    outdir = OUTDIR.joinpath("anchor")
    exp = ex.Exp(model, adata=test, feature_col="GENEID", label_col=label_col)
    s_test, test_metrics = exp.anchor()
    outfile = here(outdir.joinpath(f"anchor-{set_name}.h5ad"))
    s_train, train_metrics = exp.anchor(train)

    test_metrics.loc[:, "set"] = "test"
    train_metrics.loc[:, "set"] = "train"
    all_metrics = pd.concat([test_metrics, train_metrics])
    all_metrics.to_csv(here(outdir.joinpath(f"anchor-{set_name}.csv")), index=False)

    if not outfile.exists():
        s_test.write_h5ad(here(outdir.joinpath(f"anchor-{set_name}.h5ad")))

    ff = ex.ExpInterpreter(s_train, s_test, label_col=label_col)
    neg_contrib, per_label = ff.neg_contributions(prefix="anchor_", n=n)
    return neg_contrib


def shap_helper(
    model: PredBase,
    train: ad.AnnData,
    test: ad.AnnData,
    label_col: str,
    set_name: str,
    n: int = 10,
):
    outdir = OUTDIR.joinpath("shapley")
    exp_fn = shap.TreeExplainer(model.get_model())
    exp = ex.Exp(model, adata=test, feature_col="GENEID", label_col=label_col)
    s_test, s_vals = exp.shap(
        explain_fn=exp_fn,
        summary_plot=True,
        plot_feature_col="GENENAME",
        plot_directory=here(outdir.joinpath(f"{set_name}_plots")),
    )
    s_test.obs.loc[:, "dataset"] = set_name
    # write_pickle(s_vals, here(outdir.joinpath(f"shapley_explanation-{set_name}.pkl")))
    exp.new_adata(train)
    s_train, _ = exp.shap(
        explain_fn=exp_fn,
        summary_plot=False,
        plot_directory=None,
    )
    outfile = here(outdir.joinpath(f"shapley-{set_name}.h5ad"))
    if not outfile.exists():
        s_test.write_h5ad(here(outdir.joinpath(f"shapley-{set_name}.h5ad")))
    ff = ex.ExpInterpreter(s_train, s_test, label_col=label_col)
    neg_contrib, per_label = ff.neg_contributions("shap_", n=n)
    train_mat = ff.shap_distance(target="train", square=True)
    fig, ax = plt.subplots()
    plot_diagonal_matrix(train_mat, ax, cmap="coolwarm")
    fig.savefig(outdir.joinpath(f"train_dist-{set_name}.png"))

    test_mat = ff.shap_distance(target="test", square=True)
    fig, ax = plt.subplots()
    plot_diagonal_matrix(test_mat, ax, cmap="coolwarm")
    fig.savefig(outdir.joinpath(f"test_dist-{set_name}.png"))

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
        for method, fn in [("shapley", shap_helper), ("anchor", anchor_helper)]:
            if method in ignored:
                print(f"Ignoring method {method}")
                continue
            for n in [20, 10, 5]:
                n_outdir = OUTDIR.joinpath(method).joinpath(str(n))
                n_outdir.mkdir(exist_ok=True)
                neg_contrib = fn(
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


if __name__ == "__main__":
    args = parse_args()
    if args.dask:
        cluster = SLURMCluster(cores=int(args.cores), memory=f"{args.memory} GB")
        client = Client(cluster)
    backend = "dask" if args.dask else "loky"
    par_args = {"wait_for_workers_timeout": 0} if args.dask else {"n_jobs": args.cores}
    with joblib.parallel_backend(backend, **par_args):
        main(args)
