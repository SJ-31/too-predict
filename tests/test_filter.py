#!/usr/bin/env ipython

from pathlib import Path

import too_predict.filter as fil
import too_predict.utils as ut
from too_predict._train_utils import ADDITIONAL_SPLITS, MODELS, read_model_spec

if "/home/shannc" in str(Path.home()):
    adata = ut.training_data_internal_test()
    # adata = adata[:, :50]
else:
    adata = ut.training_data_internal()


def test_edger():
    split_fn = ADDITIONAL_SPLITS["CHULA"]
    train, test = split_fn(adata)

    cs = fil.CompareSplits(train, test)
    edger = cs.edgeR_lfc()
    return edger


def test_scanpy():
    split_fn = ADDITIONAL_SPLITS["CHULA"]
    spec = MODELS["clr_xgboost_edger"]
    F, M, T, B, E = read_model_spec(spec)
    transformed = T.fit_transform(adata)
    train, test = split_fn(transformed)

    cs = fil.CompareSplits(train, test)
    cs.scanpy_lfc()
    return cs


edger = test_edger()
