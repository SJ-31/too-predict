#!/usr/bin/env ipython
import anndata as ad
from pyhere import here
from too_predict.imputer import Imputer
from too_predict.transformer import IMPLEMENTED_TRANSFORMATION, Transformer
from too_predict.utils import dgelist2anndata, training_data_internal_test

# #  --- CODE BLOCK ---
adata = training_data_internal_test()
impute = Imputer("plus_one")


def test_modification():
    for i in IMPLEMENTED_TRANSFORMATION:
        if i == "alr":
            continue
        old = adata.copy()
        cur = adata.copy()
        Transformer(i, impute.run).fit_transform(cur)
        assert (cur.X != old.X).toarray().any()


def test_creation():
    for i in IMPLEMENTED_TRANSFORMATION:
        if i == "alr":
            continue
        cur = adata.copy()
        new = Transformer(i, impute_fn=impute.run, inplace=False).fit_transform(cur)
        assert (cur.X != new.X).toarray().any()
