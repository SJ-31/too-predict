#!/usr/bin/env ipython

from pathlib import Path

import pandas as pd
import too_predict.filter as fil
import too_predict.utils as ut
from too_predict._train_utils import default_filter_transform

if "/home/shannc" in str(Path.home()):
    adata = ut.training_data_internal_test(minimal=True)
    # adata = adata[:, :50]
else:
    adata = ut.training_data_internal()


def test_filter():
    filt, trf = default_filter_transform()
    filtered = filt.transform(adata)
    print(filtered)


test_filter()


def test_variance():
    filter = fil.Filter(
        method="variance_threshold",
        feature_col="GENEID",
        label_col="tumor_type",
    )
    changed = filter.fit_transform(adata)
    return changed


var = test_variance()

# %%


def test_edger():
    filter = fil.Filter(
        method="edgeR", feature_col="GENEID", label_col="tumor_type", n_per=60
    )
    changed = filter.fit_transform(adata)
    return changed, filter


edger, filter = test_edger()
