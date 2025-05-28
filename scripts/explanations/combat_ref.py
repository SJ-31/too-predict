#!/usr/bin/env ipython

from pathlib import Path

import too_predict.evaluation as te
import too_predict.utils as ut
from pyhere import here
from too_predict._train_utils import ADDITIONAL_SPLITS, MODELS, read_model_spec
from too_predict.model import PredWithCorrection

outdir = here("data", "output", "explanations", "batch_correction")
if str(Path.home()) != "/home/shannc":
    adata = ut.training_data_internal()
else:
    adata = ut.training_data_internal_test()
    adata = adata[:, :1000]

adata.obs.loc[:, "is_organoid"] = adata.obs["Sample_Type"] == "organoid"
label_col = "tumor_type"


# Test original
def compare_new_original(adata):
    specs = [
        ("clr_xgb3_edger_combat_ref", "original"),
        ("clr_xgb3_edger_combat_ref_org_rbatch", "org_as_ref"),
    ]
    for model_name, prefix in specs:
        filter, model, transform, _, _, correction = read_model_spec(MODELS[model_name])
        copy = adata.copy()
        copy = filter.fit_transform(copy)
        if prefix == "org_as_ref":
            model = PredWithCorrection(
                model,  # With this setup, fit model to corrected data, but do not
                # let it test on corrected data
                corrector=correction,
                transformer=transform,
                how="none",
                give_direct=True,
            )
            result = model.holdout(copy, ADDITIONAL_SPLITS, label_col=label_col)

        else:
            # Upper limit to test against, can't use this for real
            copy = correction.fit_transform(copy)
            result = model.holdout(
                copy, ADDITIONAL_SPLITS, label_col=label_col, transformer=transform
            )
        te.write_cross_val(result, outdir, prefix=f"{prefix}_")


compare_new_original(adata)
