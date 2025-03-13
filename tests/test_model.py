#!/usr/bin/env ipython

import anndata as ad
from pyhere import here
from rpy2.robjects.packages import importr
from sklearn.ensemble import RandomForestClassifier
from too_predict.model import AlrBase, RandomForestPred
from too_predict.transformer import Transformer
from too_predict.utils import training_data_internal_test

adata = training_data_internal_test()


def test_base():
    transformed = Transformer("robust_clr", None, inplace=False).fit_transform(adata)
    rf = RandomForestPred()
    results = rf.cross_validate(transformed)
    assert "fold" in results["report"].columns
    assert set(results.keys()) == {"cm", "misc", "roc", "prec_recall"}
    print(results)


def test_holdout():
    model = RandomForestPred()
    holdouts = {
        "chula": lambda x: (
            x[~x.obs["Project_ID"].str.contains("TARGET"), :],
            x[x.obs["Project_ID"].str.contains("TARGET"), :],
        ),
        "gse": lambda x: (x[: round(len(x) * 2 / 3)], x[round(len(x) * 2 / 3) :]),
    }
    hh = model.holdout(adata, holdouts)
    return hh
