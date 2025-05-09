#!/usr/bin/env ipython

import anndata as ad
import numpy as np
import pytest
import rpy2.robjects as ro
import too_predict.r_utils as ru
import too_predict.recoder as rt
import too_predict.utils as ut
from pyhere import here


def get_ref():
    ref_file = here(
        "data", "tests", "scr_ref", "htca_2025-4-21", "HTCA_ADULT_TESTIS.rds"
    )
    ro.r(f"obj <- readRDS('{str(ref_file)}')")
    mapping = ut.symbol2ensembl()
    x = ru.np_from_r(ro.r("t(as.matrix(SeuratObject::LayerData(obj)))"))
    obs = ru.df_from_r(ro.r("obj[[]]"))

    var = ru.df_from_r(ro.r("obj[['RNA']][[]]"))
    var.loc[:, "ensembl"] = list(map(lambda x: mapping.get(x, np.nan), var.index))

    ref = ad.AnnData(X=x, obs=obs, var=var)
    ref = ref[:, ~ref.var["ensembl"].isna()]
    ref.var = ref.var.set_index("ensembl")
    return ref


ADATA = ut.training_data_internal_test()
REF = get_ref()
MARKERS = ut.cell_markers_internal()


@pytest.mark.skip(reason="Done")
def test_bisque_reference():
    adata = ADATA.copy()[:, :500]
    ref = REF.copy()
    ref.obs.loc[:, "subject"] = ut.RNG.choice([0, 1, 2], size=ref.shape[0])
    recoder = rt.Recoder("bisque_reference", reference=ref, cell_type_col="Cell_Type")
    coded = recoder.fit_transform(adata)
    assert coded.shape[0] == adata.shape[0]


@pytest.mark.skip(reason="Done")
def test_bisque_marker():
    adata = ADATA.copy()
    recoder = rt.Recoder(
        "bisque_marker", markers=ut.cell_markers_internal(file_only=True)
    )
    coded = recoder.fit_transform(adata)
    assert coded.shape[0] == adata.shape[0]


def test_plage():
    adata = ADATA.copy()
    markers = ut.cell_markers_internal()
    recoder = rt.Recoder(
        "plage", reference=markers, metadata=ut.cell_markers_internal(meta=True)
    )
    coded = recoder.fit_transform(adata)
    assert coded.shape[0] == adata.shape[0]
