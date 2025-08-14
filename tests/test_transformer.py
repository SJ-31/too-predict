#!/usr/bin/env ipython

from pathlib import Path

import anndata as ad
import pandas as pd
import sklearn.metrics as met
import too_predict.model as tm
import too_predict.transformer as tt
import too_predict.utils as ut
from too_predict.filter import Filter
from too_predict.model import Pipeline

# %%

adata = ut.training_data_internal_test(minimal=True)

solo = adata[0, :].copy()
train, test = ut.train_test_split_ad(adata)


def test_leakage():
    """Check that the outcome of transformations is independent from the cohort it was computed on"""
    transforms = ["clr", "tpm", "fpkm", "robust_clr"]
    for t in transforms:
        obj = tt.Transformer(t, "plus_one", make_sparse=False)
        single = obj.transform(solo)
        together = obj.transform(ad.concat([solo, train], merge="same"))
        assert (single.X == together[0, :].X).all()


def test_pipeline():
    pipe = Pipeline(
        [
            Filter(method="mutual_information"),
            tt.Transformer("tpm", "plus_one"),
            tm.PredBase(tm.XGBEstimator()),
        ]
    )
    train_result = pipe.fit_predict(train, y="tumor_type")
    test_result = pipe.predict(test)
    print(met.accuracy_score(test.obs["tumor_type"], test_result))
    print(met.accuracy_score(train.obs["tumor_type"], train_result))
