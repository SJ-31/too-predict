#!/usr/bin/env ipython

from collections.abc import Callable
from pathlib import Path

import anndata as ad
import joblib
import pandas as pd
import shap
import too_predict.evaluation as ev
import too_predict.explanation as te
import too_predict.utils as ut
from pyhere import here
from sklearn.base import clone
from too_predict._train_utils import ADDITIONAL_SPLITS, MODELS, read_model_spec
from too_predict.filter import Filter
from too_predict.imbalance import Balancer
from too_predict.model import PredBase

OUTDIR: Path = here("data", "output", "explanations", "global_importance")
OUTDIR.mkdir(parents=True, exist_ok=True)
SPLITTER = ADDITIONAL_SPLITS["CHULA"]


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-c", "--cores", default=8, type=int)
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-i", "--ignore", default="")
    parser.add_argument("-r", "--n_removed", default=200, type=int)
    parser.add_argument("-k", "--n_kept", default=200, type=int)
    parser.add_argument("-l", "--label_class", default="tumor_type")
    return parser.parse_args()


def global_shap(model: PredBase, target, adata):
    train, test = ut.train_test_split_ad(adata, random_state=ut.RANDOM_STATE)
    model.fit(train, y=target)
    exp = te.Exp(model, label_col=target)
    exp.fit(train)
    strain, _ = exp.shap(lambda x: shap.TreeExplainer(x), summary_plot=False)
    exp.fit(test)
    stest, _ = exp.shap(lambda x: shap.TreeExplainer(x), summary_plot=False)
    inter = te.ExpInterpreter(strain, stest, label_col=target)
    g_importance = inter.global_importance("shap_")
    ranked = g_importance.mean(axis=1).sort_values(ascending=False)
    return ranked.index


def compare_best(
    model: PredBase,
    y_array: list[str],
    adatas: list[ad.AnnData],
    importance_fns: dict[str, Callable[[PredBase, str, ad.AnnData], list[str]]],
) -> dict[str, pd.DataFrame]:
    """
    Helper function for comparing the most important features as determined by the
    prediction tasks in `y_array`

    Parameters
    ----------
    importance_fn : Callable taking in model and prediction target, returns
        sorted list of features, ranked in order of decreasing importance
    """
    dfs = {}
    for importance_method, fn in importance_fns.items():
        tmp = {}
        for y, adata in zip(y_array, adatas):
            m = clone(model)
            tmp[y] = fn(m, y, adata)
        tmp_df = pd.DataFrame(tmp)
        dfs[importance_method] = tmp_df
    return dfs


def rfecv_helper(model, target, adata):
    rfecv = model.rfecv(adata, y=target)
    ranks = rfecv.ranking_
    return adata.var["GENEID"][ranks]


def train_without_noisy(
    model: PredBase,
    filter: Filter,
    train: ad.AnnData,
    test: ad.AnnData,
    method: str,
    result_df: pd.DataFrame,  # Must be sorted with most important first
    batch: str = "is_organoid",
    target: str = "tumor_type",
    n=100,
    n_lowest=1000,
):
    """Remove the top n most important features on the `batch` prediction task from adata
    and re-fit the model

    Also try keeping the n_lowest most important features on the batch task instead
    """
    top_noisy = result_df[batch][:n]
    filter_top = filter.copy()
    filter_top.blacklist(top_noisy)
    top_train = filter_top.fit_transform(train)
    top_test = filter_top.fit_transform(test)
    ev.fit_train_write(
        model,
        top_train,
        top_test,
        OUTDIR,
        f"{target}_after_{method}_top_removed.csv",
        y=target,
    )

    lowest = result_df[batch][::-1][:n_lowest]
    filter_low = filter.copy()
    filter_low.blacklist(lowest)
    low_train = filter_low.fit_transform(train)
    low_test = filter_low.fit_transform(test)
    ev.fit_train_write(
        model,
        low_train,
        low_test,
        OUTDIR,
        f"{target}_after_{method}_low_kept.csv",
        y=target,
    )


def main(args):
    if args.test:
        adata = ut.training_data_internal_test()
        adata = adata[:, :20]
        adata.obs.loc[:, "is_organoid"] = ut.RNG.choice([0, 1], size=adata.shape[0])

    else:
        adata = ut.training_data_internal()
        adata.obs.loc[:, "is_organoid"] = (
            adata.obs["Sample_Type"] == "organoid"
        ).astype(int)

    label_col = args.label_class
    spec = MODELS["clr_xgb3_edger"]
    filter, model, trans, _, _ = read_model_spec(spec)

    adata = filter.fit_transform(adata)
    adata = trans.fit_transform(adata)

    train, test = ut.train_test_split_ad(adata, random_state=ut.RANDOM_STATE)

    # Find the top globally most important features for each prediction task
    balancer = Balancer("RandomUnderSampler")
    balanced_adata = balancer.fit_transform(adata, y="is_organoid")
    fns = {"rfecv": rfecv_helper, "global_shap": global_shap}
    targets = ["is_organoid", label_col]
    adatas = [balanced_adata, adata]

    spec = {label_col: [(u, 5) for u in adata.obs[label_col]]}
    train_o, test_o = ut.split_and_sample(adata, SPLITTER, spec, ut.RNG)

    # Write initial performance
    ev.fit_train_write(
        model,
        train=train_o,
        test=test_o,
        outdir=OUTDIR,
        name=f"{label_col}_before.csv",
        y=label_col,
    )

    comparison = compare_best(
        model=model, y_array=targets, importance_fns=fns, adatas=adatas
    )

    # Try to remove noisy features found above
    for method, result in comparison.items():
        result.to_csv(OUTDIR.joinpath(f"{method}_top_features.csv"))
        train_without_noisy(
            model,
            filter,
            train,
            test,
            method=method,
            result_df=result,
            batch="is_organoid",
            target=label_col,
            n=args.n_removed,
            n_lowest=args.n_kept,
        )


if __name__ == "__main__":
    args = parse_args()
    par_args = {"n_jobs": args.cores}
    with joblib.parallel_backend("loky", **par_args):
        main(args)
