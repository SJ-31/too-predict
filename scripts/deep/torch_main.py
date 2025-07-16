#!/usr/bin/env ipython

from pathlib import Path

import anndata as ad
import lightning as L
import pandas as pd
import too_predict._train_utils as tt
import too_predict.deep.logistic as d_log
import too_predict.deep.nns as d_nn
import too_predict.deep.torch_utils as d_ut
import too_predict.utils as ut
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pyhere import here
from too_predict.deep.callbacks import AverageBest
from too_predict.deep.evaluation import cross_validate, holdout
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.transformer import Transformer

OUTDIR: Path = here("data", "output", "deep", "cross_validation")
torch.set_default_dtype(torch.float32)

# * Models to test
MODELS = {
    "MultiLevel": (
        lambda **kwargs: d_log.MultiLevel(**kwargs),
        {"skip": True, "filter": True, "holdout": False},
    ),
    "MtcLr": (
        lambda **kwargs: d_log.MtcLr(**kwargs),
        {"skip": True, "filter": True, "holdout": False},
    ),
    "Disyak": (
        lambda **kwargs: d_nn.Disyak(n_hidden=1000, dropout_p=0.2, **kwargs),
        {"skip": True, "filter": True, "cv": True, "holdout": False},
    ),
    "Disyak_All": (
        lambda **kwargs: d_nn.Disyak(n_hidden=1000, dropout_p=0.2, **kwargs),
        {"skip": False, "filter": True, "cv": True, "holdout": False},
    ),
}
TRANSFORM: Transformer = Transformer(
    "clr", impute_fn=Imputer("plus_one"), inplace=False
)
# TODO: the transformation NEEDS to be a hyperparameter that you optimize for
REF, FEAT = ut.ref_feature_lists_internal()
FILTER: Filter = Filter(
    features=FEAT["edgeR_median_lfc_feature_list_3000"],
    inplace=False,
    feature_col="GENEID",
)
LABELS = ("Sample_Type", "tumor_type")

TRAIN_KWARGS: dict = {"max_epochs": 1000}
EARLY_STOP: dict = {"monitor": "val_acc", "patience": 40, "mode": "max"}
CV_KWARGS: dict = {"batch_size": 1024, "n_splits": 5}
N_REPEATS = 3
OPTIMIZATION_KWARGS: dict = {"lr": 0.001}
SCHEDULE_KWARGS: dict = {"patience": 40}


def opt_fn(pars):
    return optim.Adam(pars, **OPTIMIZATION_KWARGS)


def get_scheduler(optimizer):
    return schedule.ReduceLROnPlateau(optimizer, **SCHEDULE_KWARGS)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action="store_true")
    args = vars(parser.parse_args())  # convert to dict
    return args


def cross_val(adata: ad.AnnData):
    for name, (m, pars) in MODELS.items():
        if pars.get("skip"):
            continue
        if pars.get("filter"):
            adata = FILTER.fit_transform(adata)
        n_features, n_classes = d_ut.data_spec(adata, y=LABELS)
        train, valid = ut.train_test_split_ad(
            adata, test_size=0.1, random_state=ut.RANDOM_STATE
        )
        # TODO: don't use random state
        outdir = OUTDIR.joinpath(name)
        outdir.mkdir(exist_ok=True)
        model = m(in_features=n_features, n_classes_per_task=n_classes)
        model.register_optimizers(opt_fn=opt_fn)
        model.register_schedulers(scheduler_fn=get_scheduler)
        kwargs = TRAIN_KWARGS.copy()
        model.set_cache("val_acc")
        kwargs["callbacks"] = [
            EarlyStopping(**EARLY_STOP),
            AverageBest(n_best=5, target="val_acc"),
        ]
        kwargs["default_root_dir"] = outdir
        trainer = L.Trainer(**kwargs)
        if pars.get("cv", True):
            dfs = []
            for i in range(N_REPEATS):
                cv: pd.DataFrame = cross_validate(
                    model=model,
                    trainer=trainer,
                    adset=d_ut.AnnDataset(train, to_encode=LABELS),
                    n_classes=n_classes,
                    validation=d_ut.AnnDataset(valid, to_encode=LABELS),
                    **CV_KWARGS,
                )
                cv.loc[:, "repeat"] = i
                dfs.append(cv)
            pd.concat(dfs).to_csv(outdir.joinpath("cv_results.csv"), index=False)
        if pars.get("holdout"):
            for i in range(N_REPEATS):
                hr_dir = outdir.joinpath(f"additional_splits_{i}")
                hr_dir.mkdir(exist_ok=True)
                _ = holdout(
                    model=model,
                    trainer=trainer,
                    adata=adata,
                    n_classes=n_classes,
                    split_fns=tt.ADDITIONAL_SPLITS,
                    to_encode=LABELS,
                    outdir=hr_dir,
                    minimal=True,
                )
    return


if __name__ == "__main__":
    args = parse_args()
    if args["test"]:
        print("Using test subset")
        adata = ut.training_data_internal_test(minimal=True)
        OUTDIR = OUTDIR.joinpath("test")
        OUTDIR.mkdir(exist_ok=True, parents=True)
        TRAIN_KWARGS["max_epochs"] = 100
    else:
        adata = ut.training_data_internal()
    adata = TRANSFORM.fit_transform(adata)
    cross_val(adata)
