#!/usr/bin/env ipython

from pathlib import Path

import anndata as ad
import pandas as pd
import too_predict._train_utils as tt
import too_predict.deep.logistic as d_log
import too_predict.deep.nns as d_nn
import too_predict.deep.torch_utils as d_ut
import too_predict.utils as ut
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
from pyhere import here
from too_predict.deep.evaluation import cross_validate, holdout
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.transformer import Transformer

OUTDIR: Path = here("data", "output", "deep", "cross_validation")
torch.set_default_dtype(torch.float64)

# * Models to test
MODELS = {
    "MultiLevel": (lambda **kwargs: d_log.MultiLevel(**kwargs), False, True),
    "MtcLr": (lambda **kwargs: d_log.MtcLr(**kwargs), False, True),
    "Disyak": (lambda **kwargs: d_nn.Disyak(n_hidden=3000, **kwargs), False, True),
}
TRANSFORM: Transformer = Transformer(
    "clr", impute_fn=Imputer("plus_one"), inplace=False
)
FILTER: Filter = Filter("edgeR_median_lfc_feature_list_3000", inplace=False)
LABELS = ("Sample_Type", "tumor_type")
TRAIN_KWARGS: dict = {"n_epochs": 1000, "at_batch_level": False}
EARLY_STOP: dict = {"patience": 40, "on_update": False, "higher_better": True}
OPTIMIZATION_KWARGS: dict = {"lr": 0.001}


def get_optimizer(model: nn.Module):
    return optim.Adam(model.named_parameters(), **OPTIMIZATION_KWARGS)


def get_scheduler(optimizer):
    return schedule.ReduceLROnPlateau(optimizer)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action="store_true")
    args = vars(parser.parse_args())  # convert to dict
    return args


def cross_val(adata: ad.AnnData):
    for name, (m, skip, filter) in MODELS.items():
        if skip:
            continue
        if filter:
            adata = FILTER.fit_transform(adata)
        n_features, n_classes = d_ut.data_spec(adata, y=LABELS)
        train, valid = ut.train_test_split_ad(
            adata, test_size=0.1, random_state=ut.RANDOM_STATE
        )
        outdir = OUTDIR.joinpath(name)
        outdir.mkdir(exist_ok=True)
        model = m(in_features=n_features, n_classes_per_task=n_classes)
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer)
        trainer = d_ut.Trainer(
            model,
            **TRAIN_KWARGS,
            optimizer=optimizer,
            scheduler=scheduler,
            record_test_score=True,
        )
        trainer.register_early_stop(d_ut.EarlyStopper(**EARLY_STOP))
        cv: pd.DataFrame = cross_validate(
            trainer,
            d_ut.AnnDataset(train, to_encode=LABELS),
            n_splits=3,
            intermediate_out=outdir,
            save_intermediate=True,
            batch_size=64,
            validation=d_ut.AnnDataset(valid, to_encode=LABELS),
        )
        cv.to_csv(outdir.joinpath("cv_results.csv"), index=False)
        hr_dir = outdir.joinpath("additional_splits")
        hr_dir.mkdir(exist_ok=True)
        _ = holdout(
            trainer,
            adata,
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
        adata = ut.training_data_internal_test()
        OUTDIR = OUTDIR.joinpath("test")
        OUTDIR.mkdir(exist_ok=True, parents=True)
        TRAIN_KWARGS["n_epochs"] = 100
    else:
        adata = ut.training_data_internal()
    adata = TRANSFORM.fit_transform(adata)
    cross_val(adata)
