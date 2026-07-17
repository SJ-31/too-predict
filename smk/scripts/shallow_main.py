#!/usr/bin/env ipython

import traceback
from pathlib import Path

import anndata as ad
import pandas as pd
import too_predict._train_utils as tt
import too_predict.evaluation as te
import too_predict.utils as ut
from snakemake.script import snakemake as smk
from too_predict.deep.metrics import ConfusionMatrices
from too_predict.model import Pipeline, PredBase

REF, FEAT = ut.ref_feature_lists_internal()

TEST = smk.config["test"]
LABEL_COL = smk.config["single_label"]
S_CONFIG = smk.config["shallow"]
FEATURE_COL = S_CONFIG["filter"]["feature_col"]


def get_adata() -> ad.AnnData:
    if TEST:
        adata: ad.AnnData = ut.training_data_internal_test()
        adata = adata[~adata.obs["Sample_Type"].isin(["organoid", "recurrent"]), :]
        adata.obs["RANDOM"] = ut.RNG.choice(
            [str(s) for s in (range(8))], adata.shape[0]
        )
    else:
        adata = ut.training_data_internal(**smk.config["training_data"])
    adata = adata[:, ~adata.var[FEATURE_COL].isna()]
    return adata


def write_results(results, result_dir, label_col):
    for name, item in results.items():
        if name != "cm" and isinstance(item, pd.DataFrame):
            item.to_csv(result_dir.joinpath(f"{label_col}-{name}.csv"), index=False)
        elif name == "cm":
            for _, cm in item.items():
                cm.to_csv(result_dir.joinpath(f"{label_col}_cm.csv"))


def save_preprocessing(label, pipeline: Pipeline, save_to, override=False):
    "Extract preprocessing from fitted pipeline as a new pipeline and record in `save_to`"
    if override or (label not in save_to):
        save_to[label] = Pipeline(steps=pipeline.preprocessing)


# * Cross validation
if smk.rule == "cross_validate":
    cv_kwargs = smk.config["shallow"]["cv"]
    adata = get_adata()
    seen_preprocessing: dict[tuple, dict[int, Pipeline]] = {}
    for model, config in smk.params["models"].items():
        spec = config.get("spec")
        if not spec:
            print(f"WARNING: pipeline spec not given for {model}")
        outdir: Path = Path(smk.params["outdir"]).joinpath(model)
        outdir.mkdir(exist_ok=True)
        misc_metrics, misses, matrices = [], [], []
        seen = False
        pp = (spec.get("filter"), spec.get("transform"))

        # Use previously-fit preprocessing defined on the same folds to avoid
        # redundant computation
        if pp in seen_preprocessing:
            seen = True
        else:
            seen_preprocessing[pp] = {}

        for i in range(smk.config["cv_n_repeats"]):
            pipeline = tt.make_pipeline(config, FEATURE_COL)
            if seen:
                pipeline = pipeline.predictor
            try:
                result = te.cross_validate(
                    model=pipeline,
                    adata=adata,
                    label_col=LABEL_COL,
                    trial=None,
                    n_splits=cv_kwargs["n_splits"],
                    record_dir=outdir,
                    random_state=smk.config["random_state"],
                    preprocessing=seen_preprocessing[pp],
                    post_fit=lambda f, m: save_preprocessing(
                        f, m, seen_preprocessing[pp]
                    ),
                )
            except ValueError as e:
                print("Command failed with exception", e)
                print("Pipeline:", pipeline)
                print("Configuration:", config)
                print(traceback.format_exc())
                continue

            misc_metrics.append(result["misc"].assign(repeat=i))
            misses.append(result["misses"].assign(repeat=i))
            matrices.extend(result["cm"].values())

        misc_all = pd.concat(misc_metrics)
        misc_all["fold"] = (
            misc_all["fold"].astype(str) + "_" + misc_all["repeat"].astype(str)
        )
        misc_all = misc_all.drop("repeat", axis="columns")
        misc_all.to_csv(outdir.joinpath(f"{LABEL_COL}-misc.csv"), index=False)
        pd.concat(misses).to_csv(
            outdir.joinpath(f"{LABEL_COL}-misses.csv"), index=False
        )
        cm = ConfusionMatrices(matrices)
        ut.write_pickle(cm, outdir.joinpath(f"{LABEL_COL}-cm.pkl"))
        cm.mean().to_csv(outdir.joinpath(f"{LABEL_COL}-mean_cm.csv"), index=False)
        cm.total_correctness().to_csv(
            outdir.joinpath(f"{LABEL_COL}-mean_cm_correctness.csv"), index=False
        )
# * Holdout
elif smk.rule == "holdout":
    adata = get_adata()
    holdout_dct = smk.config["shallow"]["holdout"]
    seen_preprocessing = {}  # Map of preprocessing combinations to dict of split names ->
    # pipeline
    for model_name, m_config in smk.params["models"].items():
        outdir = Path(smk.params["outdir"]).joinpath(model_name)
        outdir.mkdir(exist_ok=True)
        pp = (m_config.get("filter"), m_config.get("transform"))
        if pp not in seen_preprocessing:
            seen_preprocessing[pp] = {}
            seen = False
        else:
            seen = True

        pipeline = tt.make_pipeline(m_config, FEATURE_COL)
        for split_name, split_config in smk.params["split_dct"].items():
            preprocessing: Pipeline | None = seen_preprocessing[pp].get(split_name)
            train, test = ut.train_test_from_yaml(
                adata=adata,
                spec=split_config["spec"],
                mask_or=split_config.get("mask_or", True),
            )
            if seen:
                cur_model = pipeline.predictor
                train = preprocessing.transform(train)
                test = preprocessing.transform(test)
            else:
                cur_model = pipeline

            result = te.holdout(
                pipeline_fn=lambda: cur_model,
                data={split_name: (train, test)},
                label_col=LABEL_COL,
                save_split_path=outdir,
                post_fit=lambda s, m: save_preprocessing(s, m, seen_preprocessing[pp]),
            )
            write_results(result, outdir, label_col=split_name)
