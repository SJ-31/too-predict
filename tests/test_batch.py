#!/usr/bin/env ipython

from pathlib import Path

import too_predict.evaluation as te
import too_predict.utils as ut
from pyhere import here
from sklearn.ensemble import RandomForestClassifier
from too_predict._train_utils import MODELS, read_model_spec
from too_predict.corrector import PredWithCorrection
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


def test_wrapper():
    adata = ut.training_data_internal_test()
    spc = MODELS["clr_xgb3_edger_combat_ref"]
    F, M, T, B, E, C = read_model_spec(spc)
    adata.obs.loc[:, "not_primary"] = adata.obs["Sample_Type"] != "primary"

    adata = adata[
        adata.obs["Sample_Type"].isin(["primary", "metastatic", "primary_blood"]), :
    ]
    filtered = F.fit_transform(adata)
    transformed = T.fit_transform(filtered)

    transformed.obs["foo"] = "foo"
    train, test = ut.train_test_split_ad(filtered)

    C.batch_key = "not_primary"
    wrapper = PredWithCorrection(
        model=M, corrector=C, transformer=T, how="fc_mean", give_direct=True
    )
    wrapper.fit(train)

    y_true = test.obs["tumor_type"]
    proba = wrapper.predict_proba(test)
    pred = wrapper.predict(test)
    print((y_true == pred).sum() / len(y_true))
    te.get_all_metrics(y_true, proba, classes=wrapper.classes_)["acc"]
