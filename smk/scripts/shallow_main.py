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
from too_predict.transformer import Transformer

REF, FEAT = ut.ref_feature_lists_internal()

TEST = smk.config["test"]
LABEL_COL = smk.config["single_label"]
S_CONFIG = smk.config["shallow"]


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


def make_pipeline(config) -> tm.Pipeline:
    spec = config.get("config", {})
    params = config.get("params", {})

    #
    if f := spec.get("filter", "variance_threshold"):
        filter = fil.Filter(
            features=spec.get("feature_set", None),
            method=f,
            feature_col=S_CONFIG["filter"]["feature_col"],
            inplace=False,
        )
    else:
        filter = None
    if t := spec.get("transform", "clr"):
        transform = Transformer(
            method=t,
            impute_fn=spec.get("imputation", "plus_one"),
            inplace=False,
            make_sparse=False,
        )
    else:
        transform = None

    filter_before = config.get("filter_before", False)
    if filter is None and transform is None:
        preprocessing = []
    elif filter_before:
        preprocessing = [filter, transform]
    else:
        preprocessing = [transform, filter]

    m = spec.get("model", "XGBoost")
    if m == "XGBoost":
        model = tm.PredBase(tm.XGBClassifier(**params))
    return tm.Pipeline(steps=preprocessing, predictor=model)


# * Cross validation
if smk.rule == "cross_validate":
    cv_kwargs = smk.config["shallow"]["cv"]
    adata = get_adata()
    for i in smk.config["cv_n_repeats"]:
        for model, config in smk.params["models"]["shallow"].items():
            outdir: Path = Path(smk.params["outdir"].joinpath(model))
            outdir.mkdir(exist_ok=True)
            pipeline = make_pipeline(config)
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
    for model_name, config in smk.params["models"]:
        outdir = Path(smk.params["outdir"].joinpath(model_name))
        outdir.mkdir(exist_ok=True)
        pipeline: tm.Pipeline = make_pipeline(config)
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
