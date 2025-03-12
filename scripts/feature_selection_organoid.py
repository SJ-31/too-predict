#!/usr/bin/env ipython

from pathlib import Path

import joblib
import pandas as pd
import sklearn.inspection as si
import sklearn.metrics as sm
import too_predict.model as tm
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from pyhere import here
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.transformer import Transformer
from too_predict.utils import (
    RNG,
    ref_feature_lists_internal,
    training_data_internal,
    training_data_internal_test,
)

OUTDIR: Path = here("data", "output", "organoid_feature_selection")
OUTDIR.mkdir(exist_ok=True, parents=True)
REF_LISTS, FEATURE_LISTS = ref_feature_lists_internal()

TEST = True


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-c", "--cores", default=8)
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-d", "--dask", default=False, action="store_true")
    parser.add_argument("-a", "--cached", default=False, action="store_true")
    return parser.parse_args()


def main():
    if TEST:
        adata = training_data_internal_test()
    else:
        adata = training_data_internal()

    model = tm.XGBEstimator()

    filter = Filter(
        feature_col="GENEID",
        features=FEATURE_LISTS["edgeR_median_lfc_feature_list_3000"],
    )
    adata = filter.fit_transform(adata)
    if TEST:
        adata = adata[:50, :50]

    Transformer("clr", impute_fn=Imputer("plus_one"), inplace=True).fit_transform(adata)
    adata.obs["is_organoid"] = adata.obs["Sample_Type"] == "organoid"
    labels = adata.obs["is_organoid"]
    counts = adata.X.toarray()
    scorer = sm.make_scorer(sm.cohen_kappa_score)
    model.fit(counts, labels)

    results = si.permutation_importance(
        model,
        counts,
        labels,
        scoring=scorer,
        n_repeats=3 if not TEST else 1,
        random_state=RNG,
    )
    imp = results["importances"]

    df = pd.DataFrame(
        dict(
            {k: v for k, v in results.items() if k != "importances"},
            **{"GENEID": adata.var["GENEID"]},
        )
    )
    importance = pd.DataFrame(imp, columns=[f"repeat_{i}" for i in range(imp.shape[1])])
    df = (
        pd.concat([df, importance], axis=1, ignore_index=True)
        .reset_index(drop=True)
        .set_axis(list(df.columns) + list(importance.columns), axis=1)
    )
    df.to_csv(OUTDIR.joinpath("importances.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    TEST = args.test
    if args.dask:
        cluster = SLURMCluster(cores=int(args.cores), memory=f"{args.memory} GB")
        client = Client(cluster)
    backend = "dask" if args.dask else "loky"
    par_args = {"wait_for_workers_timeout": 0} if args.dask else {"n_jobs": args.cores}
    with joblib.parallel_backend(backend, **par_args):
        main()
