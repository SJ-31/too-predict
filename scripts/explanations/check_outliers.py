#!/usr/bin/env ipython

import joblib
import pandas as pd
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from pyhere import here
from pyod.models.iforest import IForest
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.transformer import Transformer
from too_predict.utils import (
    ref_feature_lists_internal,
    training_data_internal,
    training_data_internal_test,
)

OUTDIR = here("data", "output", "outlier_detection")
OUTDIR.mkdir(exist_ok=True)
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


def get_detector():
    return IForest()


def main(test: bool = True):
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

    # Want to check if the misclassified samples would be considered outliers
    #

    all_outliers = []
    for name, fn in ADDITIONAL_SPLITS.items():
        detector = get_detector()
        train, test = fn(adata)
        detector.fit(train.X)
        train.obs["is_outlier"] = detector.labels_
        train.obs["used_in"] = "train"
        test.obs["is_outlier"] = detector.predict(test.X)
        test.obs["used_in"] = "test"
        grouped = pd.concat(
            [df.loc[df["is_outlier"] == 1, :] for df in (train.obs, test.obs)],
            ignore_index=True,
        )
        grouped["test_set"] = name
        all_outliers.append(grouped)

    final = pd.concat(all_outliers, ignore_index=True)
    final.to_csv(here(OUTDIR, "all_outliers.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    if args.dask:
        cluster = SLURMCluster(cores=int(args.cores), memory=f"{args.memory} GB")
        client = Client(cluster)
    backend = "dask" if args.dask else "loky"
    par_args = {"wait_for_workers_timeout": 0} if args.dask else {"n_jobs": args.cores}
    with joblib.parallel_backend(backend):
        main(test=args.test)
