#!/usr/bin/env ipython

from pathlib import Path

from pyhere import here
from sklearn.ensemble import RandomForestClassifier
from too_predict.evaluation import write_cross_val
from too_predict.model import BatchBase, PredBase, XGBEstimator
from too_predict.utils import training_data_internal

outdir: Path = here("data", "output", "misc")
outdir.mkdir(parents=True, exist_ok=True)


def test_batch():
    adata = training_data_internal()
    adata.obs["is_organoid"] = adata.obs["Sample_Type"] == "organoid"
    bb = BatchBase(
        inner=XGBEstimator(),
        outer=PredBase(RandomForestClassifier()),
        outer_y="is_organoid",
    )
    results = bb.cross_validate_outer(adata, label_col="is_organoid")
    write_cross_val(results, outdir, prefix="batch_outer", cm_prefix="batch_outer")
