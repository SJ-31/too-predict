#!/usr/bin/env python3

from datetime import date
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from loguru import logger
from pyhere import here
from sklearn.linear_model import LogisticRegression
from too_predict.cache import NamedCache
from too_predict.evaluation import cross_validate
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.model import Pipeline, PredBase, XGBEstimator
from too_predict.transformer import Transformer
from too_predict.utils import adata_sample_by, read_existing, training_data_internal

RANDOM_STATE: int = 242
RNG = np.random.default_rng(RANDOM_STATE)


def get_data(f):
    samples_recode = {"PHcase_21": "CRC"}
    all: ad.AnnData = training_data_internal()
    all.obs = all.obs.loc[:, ["Case_ID", "Project_ID", "tumor_type"]]
    others = all[~all.obs["tumor_type"].isin(["PAAD", "COAD-READ"]), :].copy()
    others = others[adata_sample_by(others, {"tumor_type": 50}, rng=RNG),]
    others.obs["tumor_type"] = "other"
    all = all[all.obs["tumor_type"].isin(["PAAD", "COAD-READ"]), :]
    all.obs = all.obs.replace({"tumor_type": {"COAD-READ": "CRC", "PAAD": "PDAC"}})
    pdac_phcase_counts: pd.DataFrame = pd.read_csv(
        here("remote", "output", "PDAC-RNASEQ", "4-cohort-All_counts.tsv"), sep="\t"
    ).set_index("gene_id")
    pdac_phcase = ad.AnnData(
        X=pdac_phcase_counts.T,
        obs=pd.DataFrame(
            {
                "Case_ID": pdac_phcase_counts.columns,
                "Project_ID": "Chula_PHcase",
                "tumor_type": "PDAC",
            },
            index=pdac_phcase_counts.columns,
        ),
    )
    pdac_phcase.obs["tumor_type"] = pdac_phcase.obs["tumor_type"].combine(
        pdac_phcase.obs["Case_ID"], lambda x, y: samples_recode.get(y, x)
    )
    all = ad.concat([pdac_phcase, all, others], axis="obs", join="inner", merge="first")
    all.write_h5ad(f)
    return all


MODELS = {
    "xgboost_clr": Pipeline(
        [
            Transformer(method="clr", impute_fn=Imputer("plus_one")),
            Filter(method="mutual_information", top=500),
        ],
        PredBase(model=XGBEstimator()),
    ),
    "logistic_clr": Pipeline(
        [
            Transformer(method="clr", impute_fn=Imputer("plus_one")),
            Filter(method="mutual_information", top=500),
        ],
        PredBase(model=LogisticRegression(solver="saga")),
    ),
}

EXCLUDED_FOR_TRAIN = [
    "PHcase_8",
    "PHcase_11",
    "PHcase_18",
    "PHcase_19",
    "PHcase_1",
    "PHcase_2",
]


def do_cross_val(adata: ad.AnnData, outdir: Path):
    for_train = adata[~adata.obs["Case_ID"].isin(EXCLUDED_FOR_TRAIN), :]
    all_results = []
    misses = []
    cache = NamedCache(
        outdir / ".cv_cache",
        writer=lambda x: x.to_csv(index=False),
        reader=pd.read_csv,
        suffix=".csv",
    )
    for mname, pipeline in MODELS.items():
        logger.info("Starting cross validation for {}", mname)
        cv_result = cache(
            cross_validate,
            name=mname,
            pkl=True,
            model=pipeline,
            adata=for_train,
            random_state=RANDOM_STATE,
        )
        all_results.append(cv_result["misc"].assign(model=mname))
        misses.append(cv_result["misses"].assign(model=mname))
    pd.concat(all_results).to_csv(outdir / "cross_validation.csv", index=False)
    pd.concat(misses).to_csv(outdir / "misses.csv", index=False)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", default=None)
    parser.add_argument(
        "-n",
        "--no_cv",
        default=False,
        help="Do not run cross-validation",
        action="store_true",
    )
    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    args = parse_args()
    d = args["date"] or date.today().isoformat()
    workdir: Path = here("scripts", "pdac_crc")
    results_dir = workdir / f"results_{d}"
    adata: ad.AnnData = read_existing(
        here("remote", "repos", "too-predict", "training", "crc-pdac.h5ad"),
        get_data,
        ad.read_h5ad,
    )
    if not results_dir.exists():
        results_dir.mkdir()
    if not args["no_cv"]:
        do_cross_val(adata, results_dir)
