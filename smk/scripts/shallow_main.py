#!/usr/bin/env ipython

from pathlib import Path

import anndata as ad
import pandas as pd
import too_predict._train_utils as tt
import too_predict.evaluation as te
import too_predict.utils as ut
from snakemake.script import snakemake as smk
from too_predict.deep.metrics import ConfusionMatrices

REF, FEAT = ut.ref_feature_lists_internal()

TEST = smk.config["test"]
LABEL_COL = smk.config["single_label"]
S_CONFIG = smk.config["shallow"]


def get_adata() -> ad.AnnData:
    if TEST:
        adata = ut.training_data_internal_test()
        adata = adata[~adata.obs["Sample_Type"].isin(["organoid", "recurrent"]), :]
        adata.obs["RANDOM"] = ut.RNG.choice(
            [str(s) for s in (range(8))], adata.shape[0]
        )
    else:
        adata = ut.training_data_internal(**smk.config["training_data"])
    return adata


def write_results(results, result_dir, label_col):
    for name, item in results.items():
        if name != "cm" and isinstance(item, pd.DataFrame):
            item.to_csv(result_dir.joinpath(f"{label_col}-{name}.csv"), index=False)
        elif name == "cm":
            for _, cm in item.items():
                cm.to_csv(result_dir.joinpath(f"{label_col}_cm.csv"))


# * Cross validation
if smk.rule == "cross_validate":
    cv_kwargs = smk.config["shallow"]["cv"]
    adata = get_adata()
    for model, config in smk.params["models"].items():
        outdir: Path = Path(smk.params["outdir"]).joinpath(model)
        outdir.mkdir(exist_ok=True)
        misc_metrics = []
        misses = []
        matrices = []
        for i in range(smk.config["cv_n_repeats"]):
            pipeline = tt.make_pipeline(config, S_CONFIG["filter"]["feature_col"])
            result = te.cross_validate(
                model=pipeline,
                adata=adata,
                label_col=LABEL_COL,
                trial=None,
                n_splits=cv_kwargs["n_splits"],
                record_dir=outdir,
                random_state=smk.config["random_state"],
            )
            misc_metrics.append(result["misc"].assign(repeat=i))
            misses.append(result["misses"].assign(repeat=i))
            matrices.extend(result["cm"].values())
        pd.concat(misc_metrics).to_csv(
            outdir.joinpath(f"{LABEL_COL}-misc.csv"), index=False
        )
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
    for model_name, m_config in smk.params["models"].items():
        outdir = Path(smk.params["outdir"]).joinpath(model_name)
        outdir.mkdir(exist_ok=True)
        pipeline = tt.make_pipeline(m_config, S_CONFIG["filter"]["feature_col"])
        for split_name, split_config in smk.params["split_dct"].items():
            train, test = ut.train_test_from_yaml(adata=adata, spec=split_config)
            result = te.holdout(
                pipeline_fn=lambda: pipeline,
                data={split_name: (train, test)},
                label_col=LABEL_COL,
                save_split_path=outdir,
            )
            write_results(result, outdir, label_col=split_name)
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
