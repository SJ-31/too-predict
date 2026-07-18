#!/usr/bin/env python3

import pickle
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from pyhere import here
from sklearn.linear_model import LogisticRegression
from too_predict.cache import NamedCache
from too_predict.evaluation import cross_validate, holdout
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.model import Pipeline, PredBase, XGBEstimator
from too_predict.transformer import Transformer
from too_predict.utils import (
    adata_sample_by,
    read_existing,
    read_pickle,
    train_test_split_ad,
    training_data_internal,
)

RANDOM_STATE: int = 242
RNG = np.random.default_rng(RANDOM_STATE)
VAR_COL = "GENEID"
LABEL_COL = "tumor_type"

INTERACTIVE = sys.flags.interactive or hasattr(sys, "ps1")

SPLITS = {
    "CHULA": lambda x: (
        x[~x.obs["Project_ID"].str.contains("CHULA"), :],
        x[x.obs["Project_ID"].str.contains("CHULA"), :],
    )
}

try:
    matplotlib.use("QtAgg")
except ImportError:
    pass


def get_data(f):
    samples_recode = {"PHcase_21": "CRC"}
    all: ad.AnnData = training_data_internal()
    all.obs = all.obs.loc[:, ["Case_ID", "Project_ID", "tumor_type"]]
    others = all[~all.obs["tumor_type"].isin(["PAAD", "COAD-READ"]), :].copy()
    others = others[adata_sample_by(others, {"tumor_type": 50}, rng=RNG),]
    others.obs["tumor_type"] = "other"
    all = all[all.obs["tumor_type"].isin(["PAAD", "COAD-READ"]), :]
    all.obs = all.obs.replace({"tumor_type": {"COAD-READ": "CRC", "PAAD": "PDAC"}})
    pdac_phcase_counts: pd.DataFrame = pd.read_csv(
        here("remote", "output", "PDAC-RNASEQ", "4-cohort-All_counts.tsv"), sep="\t"
    ).set_index("gene_id")
    pdac_phcase = ad.AnnData(
        X=pdac_phcase_counts.T,
        obs=pd.DataFrame(
            {
                "Case_ID": pdac_phcase_counts.columns,
                "Project_ID": "CHULA_PHcase",
                "tumor_type": "PDAC",
            },
            index=pdac_phcase_counts.columns,
        ),
    )
    pdac_phcase.obs["tumor_type"] = pdac_phcase.obs["tumor_type"].combine(
        pdac_phcase.obs["Case_ID"], lambda x, y: samples_recode.get(y, x)
    )
    all = ad.concat([pdac_phcase, all, others], axis="obs", join="inner", merge="first")
    all.obs["sample_type"] = all.obs["Project_ID"].map(
        lambda x: "organoid" if x.startswith("CHULA") else "primary"
    )
    all.write_h5ad(f)
    return all


def get_default_filter():
    return Filter(method="mutual_information", top=500, feature_col="GENEID")


MODELS = {
    "xgboost_clr": lambda: Pipeline(
        [
            Transformer(method="clr", impute_fn=Imputer("plus_one")),
            get_default_filter(),
        ],
        PredBase(model=XGBEstimator()),
    ),
    "mlp_three": "Parallel",
    "logistic_clr": lambda: Pipeline(
        [
            Transformer(method="clr", impute_fn=Imputer("plus_one")),
            get_default_filter(),
        ],
        PredBase(model=LogisticRegression(solver="saga")),
    ),
}

EXCLUDED_FOR_TRAIN = [
    "PHcase_8",
    "PHcase_11",
    "PHcase_18",
    "PHcase_19",
    "PHcase_1",
    "PHcase_2",
]


def build_model(adata: ad.AnnData, outdir: Path, model: str):
    pipeline = MODELS[model]()
    pipeline.fit(adata)
    with open(outdir / f"{model}.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    return pipeline


def deep_eval(
    adata,
    mname: str,
    model_class: str,
    cache: NamedCache,
    config: dict,
    holdout: bool = False,
    split_name: str | None = None,
    split_fn: Callable | None = None,
):
    import too_predict.deep.evaluation as d_ev
    import too_predict.deep.torch_utils as d_ut
    import torch.optim as optim
    import torch.optim.lr_scheduler as schedule
    from lightning.pytorch.callbacks import EarlyStopping
    from too_predict._train_utils import get_model_fn, smk_callbacks
    from too_predict.deep.metrics import multitask_metrics2df
    from too_predict.deep.nns import HardSharer

    device = config.get("device", "cpu")
    labels = (LABEL_COL,)
    n_features, n_classes = d_ut.data_spec(adata, y=labels)
    encoders = d_ut.AnnDataset.fit_encoders(adata, labels)
    valid_split_kws = {"test_size": 0.1, "random_state": RANDOM_STATE}
    if not holdout:
        train, valid = train_test_split_ad(adata, **valid_split_kws)
        test_set = None
    else:
        train, test = split_fn(adata)
        train, valid = train_test_split_ad(adata, **valid_split_kws)
        test_set = d_ut.AnnDataset(
            train, to_encode=labels, device=device, encoders=encoders
        )
    train_set = d_ut.AnnDataset(
        train, to_encode=labels, device=device, encoders=encoders
    )
    valid_set = d_ut.AnnDataset(
        valid, to_encode=labels, device=device, encoders=encoders
    )
    mcfg = d_ut.ModuleConfig(
        cache="val_acc",
        scheduler_fn=lambda x: schedule.ReduceLROnPlateau(x, **config.get("schedule")),
        optimizer_fn=lambda x: optim.Adam(x, **config.get("optimizer", {})),
    )
    kws = config.get("model_kws")
    trainer_kws = config.get("trainer", {}) or {}
    if not holdout:
        cv_result = cache(
            d_ev.cross_validate,
            name=mname,
            pkl=True,
            model_cls=get_model_fn(model_class),
            model_kwargs=kws.get(mname, {}) or {},
            callbacks=smk_callbacks(config),
            logger_fn=lambda x: d_ut.lightning_logger(f"crc-pdac_cv: {mname}", "wandb"),
            trainer_kwargs=trainer_kws,
            adset=train_set,
            model_config=mcfg,
            n_classes=n_classes,
            in_features=n_features,
            validation=valid_set,
            device=device,
            minimal=False,
            **config.get("cv", {}) or {},
        )
        return {"misc": cv_result[0], "cm": cv_result[1]}
    else:
        result = cache(
            train_test_wrapper_torch,
            name=mname,
            pkl=True,
            module_cls=get_model_fn(model_class),
            trainer_kwargs=trainer_kws,
            device=device,
            loader_kwargs=kwargs["loader_kwargs"],
            train_test=(train_set, test_set),
            to_encode=labels,
            validation=valid_set,
            n_classes=n_classes,
            in_features=n_features,
            logger_fn=lambda x: d_ut.lightning_logger(
                f"crc-pdac_holdout_{split_name}: {mname}", "wandb"
            ),
            module_config=mcfg,
            set_label=mname,
        )
        df = multitask_metrics2df(result)
        return {"misc": df}


def do_cross_val(adata: ad.AnnData, outdir: Path, dl_config: dict):
    all_results = []
    misses = []
    cache = NamedCache(
        outdir / ".cv_cache",
        writer=lambda x: x.to_csv(index=False),
        reader=pd.read_csv,
        suffix=".csv",
    )
    for mname, pipeline in MODELS.items():
        logger.info("Starting cross validation for {}", mname)
        if isinstance(pipeline, str):
            cv_result = deep_eval(
                adata,
                model_class=pipeline,
                cache=cache,
                mname=mname,
                config=dl_config,
                holdout=False,
            )
        else:
            cv_result = cache(
                cross_validate,
                name=mname,
                pkl=True,
                model=pipeline(),
                label_col=LABEL_COL,
                adata=adata,
                random_state=RANDOM_STATE,
            )
        all_results.append(cv_result["misc"].assign(model=mname))
        misses.append(cv_result["misses"].assign(model=mname))
    pd.concat(all_results).to_csv(outdir / "cross_validation.csv", index=False)
    pd.concat(misses).to_csv(outdir / "misses.csv", index=False)


def do_holdout(adata: ad.AnnData, outdir: Path, dl_config):
    all_results = []
    cache = NamedCache(
        outdir / ".holdout_cache",
        writer=lambda x: x.to_csv(index=False),
        reader=pd.read_csv,
        suffix=".csv",
    )
    for mname, pipeline in MODELS.items():
        logger.info("Starting holdout for {}", mname)
        if not isinstance(pipeline, str):
            holdout_result = cache(
                holdout,
                split_fns=SPLITS,
                name=mname,
                pkl=True,
                pipeline_fn=pipeline,
                label_col=LABEL_COL,
                data=adata,
            )
            all_results.append(holdout_result["misc"].assign(model=mname))
        else:
            for split_name, fn in SPLITS.items():
                holdout_result = deep_eval(
                    adata,
                    mname=mname,
                    model_class=pipeline,
                    cache=cache,
                    config=dl_config,
                    holdout=True,
                    split_name=split_name,
                    split_fn=fn,
                )
                all_results.append(
                    holdout_result["misc"].assign(model=mname, split=split_name)
                )
    pd.concat(all_results).to_csv(outdir / "holdout.csv", index=False)


def inspect_model(adata, name: str, model: Pipeline, outdir: Path):
    import scanpy as sc
    from too_predict.plotting import plot_adata

    transformed = model.transform(adata)
    sc.pp.pca(transformed)
    sc.pp.neighbors(transformed)
    sc.tl.umap(transformed)
    pca = plot_adata(transformed, [LABEL_COL, "Project_ID", "sample_type"], "pca")
    pca.save(outdir / f"{model}_pca.pdf")
    umap = plot_adata(transformed, [LABEL_COL, "Project_ID", "sample_type"], "umap")
    umap.save(outdir / f"{model}_umap.pdf")

    if isinstance(model.predictor.model, LogisticRegression):
        # TODO: generalize this into a method that extracts feature importance and visualizes the top features by clustering
        weights: pd.DataFrame = pd.DataFrame(model.predictor.model.coef_).T
        weights.columns = [f"lr_weight_{c}" for c in model.predictor.model.classes_]
        weights.index = transformed.var[VAR_COL]
        transformed.var = transformed.var.merge(
            weights, left_on=VAR_COL, right_index=True
        )
        features_plot = {}
        for col in weights.columns:
            features_plot[col] = list(weights.sort_values(col).tail(n=5).index)
        sc.pl.tracksplot(
            transformed,
            features_plot,
            groupby=LABEL_COL,
            gene_symbols=VAR_COL,
            save=f"{model}_tracksplot.pdf",
        )
        # TODO: include a dotplot of the features with the
        # highest coefficients
        # would need to write a custom function cause scanpy's thing is unreliable
    transformed.var.to_csv(outdir / f"{name}-var.csv", index=False)


# TODO: output probability scores as well to analyze
# [2026-07-17 Fri] Logistic regression with SAGA works well,
# can analyze the coefficients next


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", default=None)
    parser.add_argument(
        "-n",
        "--no_cv",
        default=False,
        help="Do not run cross-validation",
        action="store_true",
    )
    parser.add_argument(
        "-u",
        "--holdout",
        default=False,
        help="Include holdout tests",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Name of a specific model (or multiple) to train on the whole data and save as pkl object",
        nargs="*",
        action="extend",
    )
    parser.add_argument("-t", "--test", default=False, help="Test", action="store_true")
    args = vars(parser.parse_args())
    return args


def main(adata, args, dir: Path, dl_config: dict):
    if args["test"]:
        adata = adata[adata_sample_by(adata, {"tumor_type": 100}), :]
    if not dir.exists():
        dir.mkdir()
    if not args["no_cv"]:
        do_cross_val(
            adata[~adata.obs["Case_ID"].isin(EXCLUDED_FOR_TRAIN), :].copy(),
            dir,
            dl_config=dl_config,
        )
    if args["holdout"]:
        do_holdout(adata, dir, dl_config=dl_config)
    if args["model"]:
        model_out = dir / "full"
        model_out.mkdir(exist_ok=True)
        for name in args["model"]:
            fitted = build_model(adata, model_out, name)
            inspect_model(adata, name=name, model=fitted, outdir=model_out)


if __name__ == "__main__" or INTERACTIVE:
    if not INTERACTIVE:
        args = parse_args()
    else:
        args = {"test": True, "date": None}
    d = args["date"] or date.today().isoformat()
    workdir: Path = here("scripts", "pdac_crc")
    results_dir = workdir / f"{'test_' if args['test'] else ''}results_{d}"
    with open(workdir / "deep.yaml") as dc:
        dl_config = yaml.safe_load(dc)
    if not torch.cuda.is_available():
        dl_config["devic"] = "cpu"
    adata: ad.AnnData = read_existing(
        here("remote", "repos", "too-predict", "training", "crc-pdac.h5ad"),
        get_data,
        ad.read_h5ad,
    )
    if not INTERACTIVE:
        main(adata, args, results_dir, dl_config)
