#!/usr/bin/env ipython
import anndata as ad
from pyhere import here
from too_predict.imputer import Imputer
from too_predict.normalizer import IMPLEMENTED_NORMALIZATION, Normalizer
from too_predict.utils import dgelist2anndata

# #  --- CODE BLOCK ---
datadir = here("data", "tests")
chol = here(datadir, "tcga_chol.rds")
adata: ad.AnnData = dgelist2anndata(str(chol))
impute = Imputer("plus_one")


def test_modification():
    for i in IMPLEMENTED_NORMALIZATION:
        old = adata.copy()
        cur = adata.copy()
        Normalizer(cur, i, impute.run).run()
        assert (cur.X != old.X).toarray().any()


def test_creation():
    for i in IMPLEMENTED_NORMALIZATION:
        cur = adata.copy()
        new = Normalizer(cur, i, impute.run, inplace=False).run()
        assert (cur.X != new.X).toarray().any()
