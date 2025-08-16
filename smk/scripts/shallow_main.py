#!/usr/bin/env ipython

from functools import reduce
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import too_predict._train_utils as tt
import too_predict.evaluation as te
import too_predict.filter as fil
import too_predict.model as tm
import too_predict.utils as ut
from snakemake.script import snakemake as smk

REF, FEAT = ut.ref_feature_lists_internal()

TEST = smk.config["test"]
LABEL_COL = smk.config["single_label"]


def get_adata() -> ad.AnnData:
    if TEST:
        adata = ut.training_data_internal_test()
    else:
        adata = ut.training_data_internal(**smk.config["training_data"])
    return adata


def write_results(results, result_dir, label_col, cm_prefix: str = ""):
    for name, item in results.items():
        if name != "cm" and isinstance(item, pd.DataFrame):
            item.to_csv(result_dir.joinpath(f"{label_col}-{name}.csv"), index=False)
        elif name == "cm":
            for lab, cm in item.items():
                cm.to_csv(
                    result_dir.joinpath(f"{label_col}-{name}_cm-{cm_prefix}{lab}.csv")
                )


def make_pipeline(config: dict) -> tm.Pipeline:
    preprocessing = []
    filter = config.get("filter")


def train_test_from_yaml(
    adata: ad.AnnData, spec: dict
) -> tuple[ad.AnnData, ad.AnnData]:
    """Subset adata into train, test sets. The parameters in the config are interpreted
    as the specification for the TEST set. The train set is the inverse of that
    """
    test_masks = []
    for obs, val_list in spec.items():
        for value, match_type in val_list.items():
            if match_type == "exact":
                test_masks.append(adata.obs[obs] == value)
            elif match_type == "contains":
                test_masks.append(adata.obs[obs].str.contains(value))
            else:
                raise ValueError(f"`{match_type}` is an invalid match type!")
    test_mask: np.ndarray = reduce(lambda x, y: x | y, test_masks)
    test = adata[test_mask, :]
    train = adata[~test_mask, :]
    return train, test


# * Cross validation
if smk.rule == "cross_validate":
    cv_kwargs = smk.config["shallow"]["cv"]
    adata = get_adata()
    for i in smk.config["cv_n_repeats"]:
        for model, spec in smk.params["models"]["shallow"].items():
            outdir: Path = Path(smk.params["outdir"].joinpath(model))
            outdir.mkdir(exist_ok=True)
            pipeline = make_pipeline(config=spec)
            result = te.cross_validate(
                model=pipeline,
                adata=adata,
                label_col=LABEL_COL,
                trial=None,
                n_splits=cv_kwargs["n_splits"],
                record_dir=outdir,
                random_state=smk.config["random_state"],
            )
            write_results(result, outdir, cm_prefix="fold_")
elif smk.rule == "holdout":
    adata = get_adata()
    holdout_dct = smk.config["shallow"]["holdout"]
    for model_name, spec in smk.params["models"]:
        outdir = Path(smk.params["outdir"].joinpath(model_name))
        outdir.mkdir(exist_ok=True)
        pipeline: tm.Pipeline = make_pipeline(config=spec)
        for split_name, config in holdout_dct["splits"].items():
            cur_outdir = outdir.joinpath(split_name)
            cur_outdir.mkdir(exist_ok=True)
            train, test = train_test_from_yaml(adata=adata, config=config)
            result = te.holdout(
                pipeline_fn=lambda: pipeline,
                data={split_name: (train, test)},
                label_col=LABEL_COL,
                save_split_path=cur_outdir,
            )
            write_results(result, cur_outdir, cm_prefix=split_name)
        if holdout_dct["organoid_test_task"]["do"]:
            adata.obs.loc[:, "is_organoid"] = adata.obs["Sample_Type"] == "organoid"
            org_outdir = outdir.joinpath("organoid_test")
            org_outdir.mkdir(exist_ok=True)
            _ = tt.organoid_test_task(
                adata=adata,
                model=pipeline,
                organoid_col="is_organoid",
                label_col=LABEL_COL,
                with_randoms=holdout_dct["organoid_test_task"]["with_randoms"],
                save_split_path=org_outdir,
                outdir=org_outdir,
            )
