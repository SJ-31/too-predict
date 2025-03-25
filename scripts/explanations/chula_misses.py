#!/usr/bin/env ipython

from pathlib import Path

import anndata as ad
import joblib
import pandas as pd
import shap
import too_predict.evaluation as te
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from pyhere import here
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
    parser.add_argument("-l", "--label_class", default="tumor_type")
    return parser.parse_args()


def shap_helper(
    model: PredBase,
    train: ad.AnnData,
    test: ad.AnnData,
    label_col: str,
    set_name: str,
) -> pd.DataFrame:
    model.fit(train)
    explainer = shap.TreeExplainer(model.get_model())
    s_test = te.get_shapley_adata(
        test, explainer, model, label_col, feature_col="GENEID"
    )
    s_test.loc[:, "dataset"] = set_name
    return s_test


def main(test: bool = True, label_col: str = "tumor_type"):
    if test:
        adata = training_data_internal_test()
    else:
        adata = training_data_internal()

    filter = Filter(
        feature_col="GENEID", features=FEATURES["edgeR_median_lfc_feature_list_3000"]
    )
    t = Transformer("clr", impute_fn=Imputer("plus_one"), inplace=False)

    adata = filter.fit_transform(adata)
    adata = t.fit_transform(adata)
    adata.X = adata.X.toarray()

    model = PredBase(model=XGBEstimator())
    all_shap_tmp = []
    for name, fn in ADDITIONAL_SPLITS.items():
        _, test = fn(adata)
        unique_values = test.obs[label_col].unique()
        spec = {label_col: [(u, 5) for u in unique_values]}
        train, test = split_and_sample(adata, fn, spec, RNG)
        all_shap_tmp.append(
            shap_helper(model, train, test, label_col=label_col, set_name=name)
        )
    all_shap = pd.concat(all_shap_tmp)
    all_shap.to_csv(OUTDIR.joinpath("shapley_results.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    if args.dask:
        cluster = SLURMCluster(cores=int(args.cores), memory=f"{args.memory} GB")
        client = Client(cluster)
    backend = "dask" if args.dask else "loky"
    par_args = {"wait_for_workers_timeout": 0} if args.dask else {"n_jobs": args.cores}
    with joblib.parallel_backend(backend, **par_args):
        main(args.test, args.label_class)
