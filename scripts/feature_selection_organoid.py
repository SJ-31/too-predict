#!/usr/bin/env ipython

from pathlib import Path

import anndata as ad
import joblib
import pandas as pd
import scanpy as sc
import sklearn.feature_selection as fs
import sklearn.metrics as sm
from pyhere import here
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from too_predict._train_utils import ADDITIONAL_SPLITS, MODELS, read_model_spec
from too_predict.evaluation import write_cross_val
from too_predict.filter import Filter
from too_predict.model import PredBase
from too_predict.utils import (
    RANDOM_STATE,
    train_test_split_ad,
    training_data_internal,
    training_data_internal_test,
)

OUTDIR: Path = here("data", "output", "organoid_feature_selection")
OUTDIR.mkdir(exist_ok=True, parents=True)
TEST = True

F, M, T, B, R, C = read_model_spec(MODELS["clr_xgb3_edger"])


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-c", "--cores", default=8, type=int)
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-d", "--dask", default=False, action="store_true")
    parser.add_argument("-a", "--cached", default=False, action="store_true")
    return parser.parse_args()


def get_adata() -> tuple[ad.AnnData, ad.AnnData]:
    if TEST:
        adata = training_data_internal_test()
    else:
        adata = training_data_internal()

    adata.obs["is_organoid"] = adata.obs["Sample_Type"] == "organoid"
    primary = adata.copy()[~adata.obs["is_organoid"], :]
    sc.pp.subsample(primary, random_state=RANDOM_STATE, fraction=0.03)
    organoid = adata.copy()[adata.obs["is_organoid"], :]
    organoid_primary = ad.concat(
        [primary, organoid], axis="obs", join="inner", merge="same"
    )

    adata = T.fit_transform(adata)
    organoid_primary: ad.AnnData = T.fit_transform(organoid_primary)
    adata: ad.AnnData = adata[adata.obs["Sample_Type"].isin(["organoid", "primary"])]
    if TEST:
        adata = adata[:50, :50]
    return adata, organoid_primary


def main2():
    # Identify important features on a organoid vs primary prediction task using
    # a simple model, and see if top 3000 least important features from this task
    # are still useful for predicting tumor type
    adata, organoid_primary = get_adata()
    o_model: PredBase = PredBase(model=LogisticRegressionCV())
    organoid_prediction_scores = o_model.cross_validate(
        organoid_primary, label_col="is_organoid"
    )
    write_cross_val(organoid_prediction_scores, OUTDIR, prefix="organoid_prediction")
    o_train, o_test = train_test_split_ad(organoid_primary)
    o_model.fit(o_train, y="is_organoid")
    with_coefs = pd.Series(o_model.model.coef_[0], index=organoid_primary.var["GENEID"])
    with_coefs = with_coefs[~with_coefs.index.isna()]
    with_coefs = with_coefs.abs().sort_values(ascending=True)
    lowest_n = with_coefs[:3000]
    new_filter = Filter(features=lowest_n.index, feature_col="GENEID")
    print(adata)
    print(len(new_filter.features))
    print(adata.shape)
    adata = new_filter.fit_transform(adata)
    results = M.cross_validate(adata, label_col="tumor_type")

    write_cross_val(results, OUTDIR, prefix="after_removing")
    results2 = M.holdout(adata, ADDITIONAL_SPLITS, label_col="tumor_type")
    write_cross_val(results2, OUTDIR, prefix="after_removing_holdout")


def main():
    adata, organoid_primary = get_adata()

    adata = F.fit_transform(adata)
    organoid_primary = F.fit_transform(organoid_primary)
    labels = organoid_primary.obs["is_organoid"]
    counts = organoid_primary.X.toarray()
    scorer = sm.make_scorer(sm.cohen_kappa_score)
    x_train, x_test, y_train, y_test = train_test_split(counts, labels)
    M.fit(x_train, y_train)
    organoid_primary.var.loc[:, "raw_importance"] = M.feature_importances_
    organoid_primary.var.to_csv(OUTDIR.joinpath("raw_importances.csv"), index=False)

    # [2025-03-13 Thu] Results were all zero with permutation_importance
    rfecv = fs.RFECV(estimator=M, step=1, cv=StratifiedKFold(5), scoring=scorer)
    rfecv.fit(counts, labels)

    cv_score = cross_val_score(M, x_train, y_train)
    print(cv_score)

    df = pd.DataFrame(
        {"GENEID": organoid_primary.var["GENEID"], "ranking": rfecv.ranking_}
    )
    df.to_csv(OUTDIR.joinpath("importances.csv"), index=False)
    score_df = pd.DataFrame(rfecv.cv_results_)
    score_df.to_csv(OUTDIR.joinpath("cv_results.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    TEST = args.test
    backend = "dask" if args.dask else "loky"
    par_args = {"n_jobs": args.cores}
    with joblib.parallel_backend(backend, **par_args):
        # main()
        main2()
