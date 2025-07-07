#!/usr/bin/env ipython

from pathlib import Path

import anndata as ad
import pandas as pd
import skdim
import too_predict.utils as ut
from pyhere import here
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.transformer import Transformer

OUTDIR: Path = here("data", "output", "deep")
TRANSFORM: Transformer = Transformer(
    "clr", impute_fn=Imputer("plus_one"), inplace=False
)
REF, FEAT = ut.ref_feature_lists_internal()
ESTIMATORS: dict = {
    # "TwoNN": lambda: skdim.id.TwoNN(),  # ignored
    "MLE": lambda: skdim.id.MLE(),
    "MoM": lambda: skdim.id.MOM(),
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    parser.add_argument("-t", "--test", default=False, help="Test", action="store_true")
    args = vars(parser.parse_args())  # convert to dict
    return args


@ut.SaveOrLoad(
    out=here(OUTDIR, "id_estimation.csv"), read_fn=pd.read_csv, logdir=here("log")
)
def get_id(f, x: ad.AnnData, filters: list[str]):
    results = {"feature_set": []}
    for est in ESTIMATORS.keys():
        results[est] = []
    for ref in filters:
        if ref != "all":
            F = Filter(features=FEAT[ref], inplace=False, feature_col="GENEID")
            current = F.fit_transform(x)
        else:
            current = x.copy()
        for name, e_fn in ESTIMATORS.items():
            E = e_fn()
<<<<<<< HEAD
            E.fit(current)
            print(f"{name} complete")
            results[name].append(E.dimension_)
=======
            results[name].append(E.fit_transform(current))
            print(f"{name} complete")
>>>>>>> 3fbdbe7 (chore: disable twonn)
        results["feature_set"].append(ref)
    df = pd.DataFrame(results)
    df.to_csv(f, index=False)


if __name__ == "__main__":
    args = parse_args()
    adata: ad.AnnData
    if args["test"]:
        print("Using test subset")
        adata = ut.training_data_internal_test(minimal=False)
        OUTDIR = OUTDIR.joinpath("test")
        OUTDIR.mkdir(exist_ok=True, parents=True)
    else:
        adata = ut.training_data_internal()
    adata = TRANSFORM.fit_transform(adata)
    adata.X = adata.X.toarray()
    get_id(
        x=adata,
        filters=[
            "edgeR_median_lfc_feature_list_3000",
            "edgeR_15_per_type_ovp_tissue_enriched",
            "edgeR_70_per_type_ovp_.txt",
            "variance_feature_list_3000",
            # "all",
        ],
    )
