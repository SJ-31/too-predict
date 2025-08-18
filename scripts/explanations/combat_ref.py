#!/usr/bin/env ipython

from pathlib import Path

import too_predict.evaluation as te
import too_predict.utils as ut
from pyhere import here
from too_predict._train_utils import (
    ADDITIONAL_SPLITS,
    MODELS,
    organoid_test_task,
    read_model_spec,
)
from too_predict.model import PredWithCorrection

outdir = here("data", "output", "explanations", "batch_correction")
if str(Path.home()) != "/home/shannc":
    adata = ut.training_data_internal()
else:
    adata = ut.training_data_internal_test()
    adata = adata[:, :1000]

adata.obs.loc[:, "is_organoid"] = adata.obs["Sample_Type"] == "organoid"
label_col = "tumor_type"

# [2025-05-29 Thu] In outdir, the directory org_as_ref_separate are the results of combat_ref
# batch correction where the correction was applied to the train and test data separately
# i.e. the correction parameters didn't see the organoids
# In practice, they would need to have access to the organoids


# Test original
def compare_new_original(adata):
    specs = [
        ("clr_xgb3_edger_combat_ref", "original", False),
        ("clr_xgb3_edger_combat_ref_org_rbatch", "org_as_ref", False),
        ("clr_xgboost_edger_per_type_ovp_t_enriched", "clr_xgb3", True),
    ]
    for model_name, prefix, skip in specs:
        if skip:
            continue
        filter, model, transform, _, _, correction = read_model_spec(MODELS[model_name])
        copy = adata.copy()
        copy = filter.fit_transform(copy)
        if prefix == "org_as_ref":
            copy = correction.fit_transform(copy)
            # model = PredWithCorrection(
            #     model,  # With this setup, fit model to corrected data, but do not
            #     # let it test on corrected data
            #     corrector=correction,
            #     transformer=transform,
            #     how="none",
            #     give_direct=True,
            # )
            result = te.holdout(
                model=model,
                data=copy,
                split_fns=ADDITIONAL_SPLITS,
                label_col=label_col,
                transformer=transform,
                apply_correction_to="train",
            )
            otest_dir = outdir.joinpath("org_as_ref_organoid_test")
            otest_dir.mkdir(exist_ok=True)
            organoid_test_task(
                adata=adata.copy(),
                model_spec=MODELS[model_name],
                outdir=otest_dir,
                correction_mode="on_train",
                save_split_path=otest_dir,
                with_randoms=False,
            )
        elif prefix == "original":
            # Upper limit to test against, can't use this for real
            otest_dir = outdir.joinpath("original_organoid_test")
            otest_dir.mkdir(exist_ok=True)
            copy = correction.fit_transform(copy)
            result = model.holdout(
                copy, ADDITIONAL_SPLITS, label_col=label_col, transformer=transform
            )
            # [2025-05-29 Thu] So doing it with CLR does produce the right accuracy
            organoid_test_task(
                adata=adata.copy(),
                model_spec=MODELS[model_name],
                outdir=otest_dir,
                correction_mode="before_split",
                save_split_path=otest_dir,
                with_randoms=False,
            )
        else:
            otest_dir = outdir.joinpath(f"{prefix}_organoid_test")
            otest_dir.mkdir(exist_ok=True)
            result = model.holdout(
                copy, ADDITIONAL_SPLITS, label_col=label_col, transformer=transform
            )
            organoid_test_task(
                adata=adata.copy(),
                model_spec=MODELS[model_name],
                outdir=otest_dir,
                with_randoms=False,
                save_split_path=otest_dir,
            )
        te.write_cross_val(
            result, outdir.joinpath("additional_splits"), prefix=f"{prefix}_"
        )


compare_new_original(adata)
