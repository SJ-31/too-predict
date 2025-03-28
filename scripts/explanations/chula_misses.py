#!/usr/bin/env ipython

from pathlib import Path

import anndata as ad
import joblib
import pandas as pd
import shap
import too_predict.explanation as ex
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from pyhere import here
from too_predict.evaluation import get_all_metrics, write_cross_val
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.model import PredBase, XGBEstimator
from too_predict.transformer import Transformer
from too_predict.utils import (
    RNG,
    ref_feature_lists_internal,
    split_and_sample,
    training_data_internal,
    training_data_internal_test,
)

OUTDIR: Path = here("data", "output", "explanations", "chula_misses")
OUTDIR.mkdir(exist_ok=True, parents=True)
REFS, FEATURES = ref_feature_lists_internal()

ADDITIONAL_SPLITS: dict = {
    "CHULA": lambda x: (
        x[~x.obs["Project_ID"].str.contains("CHULA"), :],
        x[x.obs["Project_ID"].str.contains("CHULA"), :],
    ),
    "CGCI": lambda x: (
        x[~x.obs["Project_ID"].str.contains("CGCI"), :],
        x[x.obs["Project_ID"].str.contains("CGCI"), :],
    ),
    "CPTAC": lambda x: (
        x[~x.obs["Project_ID"].str.contains("CPTAC"), :],
        x[x.obs["Project_ID"].str.contains("CPTAC"), :],
    ),
    "GEO": lambda x: (
        x[~x.obs["Project_ID"].str.contains("GSE"), :],
        x[x.obs["Project_ID"].str.contains("GSE"), :],
    ),
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-c", "--cores", default=8)
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-d", "--dask", default=False, action="store_true")
    parser.add_argument("--no_shap", default=False, action="store_true")
    parser.add_argument("--no_morris", default=False, action="store_true")
    parser.add_argument("-l", "--label_class", default="tumor_type")
    return parser.parse_args()


def shap_helper(
    model: PredBase,
    train: ad.AnnData,
    test: ad.AnnData,
    label_col: str,
    set_name: str,
):
    outdir = OUTDIR.joinpath("shapley")
    explainer = shap.TreeExplainer(model.get_model())
    s_test, s_vals = ex.get_shap_adata(
        test,
        explainer,
        model,
        label_col,
        feature_col="GENEID",
        summary_plot=True,
        plot_feature_col="GENENAME",
        plot_directory=here(outdir.joinpath(f"{set_name}_plots")),
    )
    s_test.obs.loc[:, "dataset"] = set_name
    # write_pickle(s_vals, here(outdir.joinpath(f"shapley_explanation-{set_name}.pkl")))
    s_train, _ = ex.get_shap_adata(
        train,
        explainer,
        model,
        label_col,
        feature_col="GENEID",
        summary_plot=False,
        plot_directory=None,
    )
    s_test.write_h5ad(here(outdir.joinpath(f"shapley-{set_name}.h5ad")))
    ff = ex.Explain(s_train, s_test, label_col=label_col)
    neg_contrib, per_label = ff.shap_neg_contributions()
    return neg_contrib


def main(args):
    if args.test:
        adata = training_data_internal_test()
    else:
        adata = training_data_internal()

    feature_set = "edgeR_median_lfc_feature_list_3000"
    filter = Filter(feature_col="GENEID", features=FEATURES[feature_set])
    t = Transformer("clr", impute_fn=Imputer("plus_one"), inplace=False)

    adata = filter.fit_transform(adata)
    adata = t.fit_transform(adata)
    adata.X = adata.X.toarray()

    label_col = args.label_class
    if not args.no_shap:
        model = PredBase(model=XGBEstimator())
        for name, fn in ADDITIONAL_SPLITS.items():
            _, test = fn(adata)
            unique_values = test.obs[label_col].unique()
            spec = {label_col: [(u, 5) for u in unique_values]}
            train, test = split_and_sample(adata, fn, spec, RNG)
            model.fit(train)
            neg_contrib = shap_helper(
                model, train, test, label_col=label_col, set_name=name
            )
            new_filter = Filter(
                feature_col="GENEID",
                features=FEATURES[feature_set],
                blacklist=neg_contrib,
            )
            proba = model.predict_proba(test)
            perf = get_all_metrics(test.obs[label_col], proba, model.classes_)
            metrics = (
                f"acc: {perf.get('acc')}\nbalanced_acc: {perf.get('balanced_acc')}\n"
                f"kappa: {perf.get('kappa')}"
            )
            OUTDIR.joinpath("shapley").joinpath(f"{name}_before.txt").write_text(
                metrics
            )

            no_neg_contrib = new_filter.fit_transform(adata)
            # See how the model performs after removing the negatively contributing
            # features
            updated_perf = model.holdout(  # Will refit in here
                no_neg_contrib,
                {k: v for k, v in ADDITIONAL_SPLITS if k == name},
                label_col=label_col,
            )
            write_cross_val(updated_perf, OUTDIR.joinpath("shapley"), prefix=name)


if __name__ == "__main__":
    args = parse_args()
    if args.dask:
        cluster = SLURMCluster(cores=int(args.cores), memory=f"{args.memory} GB")
        client = Client(cluster)
    backend = "dask" if args.dask else "loky"
    par_args = {"wait_for_workers_timeout": 0} if args.dask else {"n_jobs": args.cores}
    with joblib.parallel_backend(backend, **par_args):
        main(args)
