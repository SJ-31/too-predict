#!/usr/bin/env ipython

import anndata as ad
from pyhere import here
from rpy2.robjects.packages import importr
from sklearn.ensemble import RandomForestClassifier
from too_predict.model import AlrBase2, RandomForestPred, SimBase
from too_predict.utils import dgelist2anndata

base = importr("base")
ensembldb = importr("ensembldb")
datadir = here("data", "tests")
hcc = here(datadir, "tcga_hcc.rds")
chol = here(datadir, "tcga_chol.rds")
coad = here(datadir, "tcga_coad-read.rds")

test_sets = {"LIHC": hcc, "CHOL": chol, "COAD": coad}
hg38 = here("data", "Homo_sapiens.GRCh38.113.sqlite")


def loader(path, type):
    adata = dgelist2anndata(str(path))
    adata = adata[:100]
    adata.obs["tumor_type"] = type
    return adata


adata: ad.AnnData = ad.concat([loader(t, p) for p, t in test_sets.items()])
adata.var.index = adata.var.index.to_series().str.replace("\\..*", "", regex=True)


def test_base():
    rf = RandomForestPred("clr", "plus_one")
    results = rf.cross_validate(adata)
    print(results)


def test_alr():
    # TODO <2025-02-23 Sun> write this test
    rf = RandomForestPred("clr", "plus_one")
    results = rf.cross_validate(adata)
    print(results)


def test_dirichlet():
    dir = SimBase(
        "clr", None, simulation="dirichlet", model=RandomForestClassifier(), n=2
    )
    results = dir.cross_validate(adata)
    print(results)
