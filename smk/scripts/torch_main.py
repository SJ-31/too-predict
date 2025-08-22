#!/usr/bin/env ipython

from functools import reduce
from pathlib import Path

import anndata as ad
import pandas as pd
import too_predict._train_utils as tt
import too_predict.deep.torch_utils as d_ut
import too_predict.utils as ut
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
import yaml
from sklearn.model_selection import KFold
from too_predict._train_utils import get_model_fn, smk_callbacks
from too_predict.deep.distillation import TeacherResponse
from too_predict.deep.evaluation import (
    cross_validate,
    train_test_split_torch,
    train_test_wrapper_torch,
)
from too_predict.deep.metrics import (
    multitask_acc,
    multitask_all_metrics,
    multitask_metrics2df,
)
from too_predict.deep.nns import Baseline
from too_predict.model import Pipeline
from torch.utils.data import Dataset, Subset

try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = ut.DummySnake(rule="my_rule", configfile="my_config")

torch.set_default_dtype(torch.float32)


LABELS = smk.config["multi_labels"]
DL_CONFIG = smk.config["dl"]
TEST: bool = smk.config["test"]
N_REPEATS = smk.config["cv_n_repeats"] if not TEST else 2
BASELINE_KWARGS = {"max_depth": 2}
DEVICE = "cpu" if TEST else DL_CONFIG["device"]
ROUTINE = "cv" if smk.config["do_cv"] and not smk.config["do_holdout"] else "holdout"

if (mlp := DL_CONFIG["matmul_precision"].lower()) != "none":
    torch.set_float32_matmul_precision(mlp)


# TODO: the transformation NEEDS to be a hyperparameter that you optimize for

MODELS = smk.config["models"]["dl"]


def opt_fn(pars):
    return optim.Adam(pars, **DL_CONFIG["optimizer"])


def get_scheduler(optimizer):
    return schedule.ReduceLROnPlateau(optimizer, **DL_CONFIG["schedule"])


def get_kwargs(model_name):
    model_kwargs = MODELS[model_name].get("params", {})
    model_cls = get_model_fn(model_name)
    trainer_kwargs = DL_CONFIG["trainer"].copy()
    trainer_kwargs["fast_dev_run"] = smk.config["dev_run"]
    if TEST:
        trainer_kwargs["log_every_n_steps"] = 1
        trainer_kwargs["max_epochs"] = 3
    mconfig = d_ut.ModuleConfig(
        cache="val_acc",
        scheduler_fn=get_scheduler,
        optimizer_fn=opt_fn,
        record_norm=smk.config.get("record_norm", True),
        **MODELS[model_name].get("s_params", {}),
    )
    return {
        "model_class": model_cls,
        "model_kwargs": model_kwargs,
        "mconfig": mconfig,
        "trainer_kwargs": trainer_kwargs,
    }

    # TODO: get backed anndata working with anndataset


def baseline_eval(train: Dataset, test: Dataset, n_features, n_classes, outdir: Path):
    base = Baseline(n_features, n_classes, **BASELINE_KWARGS)
    base.fit(train)
    scores = base.predict_proba(test[:])
    metrics = multitask_all_metrics(
        y_true=test[:][1],
        scores=scores,
        task_names=LABELS,
        n_classes=n_classes,
    )
    df = multitask_metrics2df(metrics)
    df.to_csv(outdir.joinpath("baseline.csv"), index=False)


def holdout(file, distillation: bool = False):
    outdir = Path(smk.params["outdir"])
    train_path = Path(file)
    split_name = train_path.stem.replace("_train", "")
    with open(train_path.parent.joinpath(f"{split_name}_spec.yaml"), "r") as f:
        tmp = yaml.safe_load(f)
        n_features = tmp["n_features"]
        n_classes = tmp["n_classes"]
    test_path = train_path.parent.joinpath(f"{split_name}_test.h5ad")
    cur_outdir = outdir.joinpath(split_name)
    train = d_ut.AnnDataset(ad.read_h5ad(train_path), to_encode=LABELS, device=DEVICE)
    train, valid = train_test_split_torch(
        train,
        generator=torch.Generator().manual_seed(smk.config["random_state"]),
        as_dataloader=False,
    )

    test = d_ut.AnnDataset(ad.read_h5ad(test_path), to_encode=LABELS, device=DEVICE)
    baseline_eval(train, test, n_features, n_classes=n_classes, outdir=cur_outdir)
    if distillation:
        baseline = Baseline(
            in_features=n_features, n_classes_per_task=n_classes, **BASELINE_KWARGS
        )
        train = TeacherResponse(data=train, teacher=baseline)
        valid = TeacherResponse(data=valid, teacher=baseline, is_fitted=True)
    for model_name in smk.params["models"]:
        outname = f"{model_name}_kd" if distillation else model_name
        output = cur_outdir.joinpath(f"{outname}.csv")
        if not output.exists():
            kwargs = get_kwargs(model_name)
            trainer_kwargs: dict = kwargs["trainer_kwargs"]
            trainer_kwargs["callbacks"] = smk_callbacks(DL_CONFIG)
            result = train_test_wrapper_torch(
                module_cls=kwargs["model_class"],
                trainer_kwargs=trainer_kwargs,
                device=DEVICE,
                train_test=(train, test),
                to_encode=LABELS,
                validation=valid,
                n_classes=n_classes,
                in_features=n_features,
                logger_fn=lambda x: d_ut.lightning_logger(
                    x,
                    platform="tensorboard",
                    save_dir=cur_outdir.joinpath(f"{outname}_tensorboard"),
                ),
                module_config=kwargs["mconfig"],
                set_label=model_name,
            )
            df = multitask_metrics2df(result)
            df.to_csv(output, index=False)


def cross_val(in_file: str, distillation: bool = False):
    adata = ad.read_h5ad(in_file, backed=True)
    model = Path(in_file).stem
    n_features, n_classes = d_ut.data_spec(adata, y=LABELS)
    train, valid = ut.train_test_split_ad(
        adata, test_size=0.1, random_state=ut.RANDOM_STATE
    )
    train_set = d_ut.AnnDataset(train, to_encode=LABELS, device=DEVICE)
    valid_set = d_ut.AnnDataset(valid, to_encode=LABELS, device=DEVICE)
    if distillation:
        baseline = Baseline(
            in_features=n_features, n_classes_per_task=n_classes, **BASELINE_KWARGS
        )
        train_set = TeacherResponse(data=train_set, teacher=baseline)
        valid_set = TeacherResponse(data=valid_set, teacher=baseline, is_fitted=True)
        outdir = Path(smk.params["outdir"]).joinpath(f"{model}_kd")
    else:
        outdir = Path(smk.params["outdir"]).joinpath(model)
    kwargs = get_kwargs(model)
    dfs = []
    for i in range(N_REPEATS):
        cv: pd.DataFrame = cross_validate(
            model_cls=kwargs["model_class"],
            model_kwargs=kwargs["model_kwargs"],
            model_config=kwargs["mconfig"],
            callbacks=smk_callbacks(DL_CONFIG),
            logger_fn=lambda x: d_ut.lightning_logger(
                f"fold_{x}_repeat_{i}",
                platform="tensorboard",
                save_dir=outdir.joinpath("tensorboard"),
            ),
            trainer_kwargs=kwargs["trainer_kwargs"],
            adset=train_set,
            in_features=n_features,
            n_classes=n_classes,
            validation=valid_set,
            device=DEVICE,
            **DL_CONFIG["cv"],
        )
        cv.loc[:, "repeat"] = i
        dfs.append(cv)
    pd.concat(dfs).to_csv(outdir.joinpath("cv_results.csv"), index=False)


def get_adata():
    if TEST:
        adata = ut.training_data_internal_test(minimal=True)
        adata = adata[~adata.obs["Sample_Type"].isin(["organoid", "redcurrent"]), :]
        adata.obs["RANDOM"] = ut.RNG.choice(
            [str(s) for s in (range(8))], adata.shape[0]
        )
    else:
        adata = ut.training_data_internal()
    masks = []
    for col, allowed in smk.config.get("obs_filters", {}).items():
        if allowed:
            print(f"Only allowing {adata}['{col}'] in {allowed}")
            masks.append(adata[col].isin(allowed))
    if masks:
        mask = reduce(lambda x, y: x | y, masks)
        adata = adata[mask, :]
    return adata


# * Snakemake rules
# ** Preprocess for CV
if smk.rule == "preprocess" and ROUTINE == "cv":
    filter, transform = tt.default_filter_transform(smk.config)
    adata = get_adata()
    for model in smk.output:
        if Path(model).exists():
            continue
        name = Path(model).stem
        spec = smk.config["models"]["dl"].get(name)
        cur: ad.AnnData = adata
        if spec.get("filter"):
            cur = filter.fit_transform(cur)
        if spec.get("transform"):
            cur = transform.fit_transform(cur)
        cur.write_h5ad(model)
# ** Preprocess for holdout
if smk.rule == "preprocess" and ROUTINE == "holdout":
    adata = get_adata()
    feature_col = smk.config["shallow"]["filter"]["feature_col"]
    holdout_config = DL_CONFIG["holdout"]
    split_dct = smk.params["split_dct"]
    for train_file in smk.output["train"]:
        train_file = Path(train_file)
        outdir = train_file.parent
        if train_file.exists():
            continue
        split_name = train_file.stem.replace("_train", "")
        split_config = split_dct.get(split_name)
        if not split_config:
            raise ValueError("split not defined in config!")
        train, test = ut.train_test_from_yaml(adata=adata, spec=split_config["spec"])
        print(split_name, train.shape, test.shape)
        pipeline_name = split_config.get("pipeline", "clr_edgeR_old")
        pipeline_spec = smk.config["models"]["shallow"][pipeline_name]
        preprocessing: Pipeline = tt.make_pipeline(
            pipeline_spec, feature_col=feature_col, with_predictor=False
        )
        n_features, n_classes = d_ut.data_spec(adata, y=LABELS)
        train = preprocessing.fit_transform(train)
        test = preprocessing.transform(test)

        train.write_h5ad(outdir.joinpath(f"{split_name}_train.h5ad"))
        test.write_h5ad(outdir.joinpath(f"{split_name}_test.h5ad"))
        with open(outdir.joinpath(f"{split_name}_spec.yaml"), "w") as f:
            yaml.safe_dump({"n_classes": n_classes, "n_features": n_features}, f)

# ** Cross validate
if smk.rule == "cross_validate":
    for f in smk.input:
        if "baseline.h5ad" not in f:
            cross_val(f)
if smk.rule == "distillation" and ROUTINE == "cv":
    for f in smk.input:
        if "baseline.h5ad" not in f:
            cross_val(f, distillation=True)
# ** Holdout
if smk.rule == "holdout":
    for f in smk.input:
        holdout(f, False)
if smk.rule == "distillation" and ROUTINE == "holdout":
    for f in smk.input:
        holdout(f, True)

# ** Baseline CV
if smk.rule == "baseline" and smk.config["do_cv"]:
    baseline = [x for x in smk.input if "baseline" in str(x)][0]
    batch_size = DL_CONFIG["cv"]["batch_size"]
    adata = ad.read_h5ad(baseline, backed=True)
    adset = d_ut.AnnDataset(adata, to_encode=LABELS)
    cv = KFold(
        n_splits=DL_CONFIG["cv"]["n_splits"], shuffle=True, random_state=ut.RANDOM_STATE
    )
    results: dict = {"fold": []}
    for label in LABELS:
        results[f"{label}_train_acc"] = []
        results[f"{label}_test_acc"] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(adset)):
        train = Subset(adset, train_idx)
        test = Subset(adset, test_idx)
        n_features, n_classes = d_ut.data_spec(train)
        baseline = Baseline(
            in_features=n_features, n_classes_per_task=n_classes, **BASELINE_KWARGS
        )
        baseline.fit(train)
        train_acc = multitask_acc(
            y_true=train[:][1],
            predictions=baseline.predict_step(train[:][0]),
            task_names=LABELS,
            n_classes=n_classes,
        )
        test_acc = multitask_acc(
            y_true=test[:][1],
            predictions=baseline.predict_step(test[:][0]),
            task_names=LABELS,
            n_classes=n_classes,
        )
        for group, acc in zip(["train", "test"], [train_acc, test_acc]):
            for task in LABELS:
                results[f"{task}_{group}_acc"].append(acc[task])
        results["fold"].append(fold)
    df = pd.DataFrame(results)
    df.to_csv(smk.output["cv"], index=False)
    for i, label in enumerate(LABELS):
        ut.xgb_complexity(baseline.models[i]).to_csv(smk.output[label], index=False)
