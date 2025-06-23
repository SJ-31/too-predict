#!/usr/bin/env ipython

from pathlib import Path

import anndata as ad
import too_predict._train_utils as tt
import too_predict.deep.logistic as d_log
import too_predict.deep.torch_utils as d_ut
import too_predict.evaluation as te
import too_predict.utils as ut
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
from pyhere import here
from too_predict.deep.evaluation import holdout

OUTDIR: Path = here("data", "output", "deep", "cross_validation")

# * Models to test
MODELS = {"MultiLevel": d_log.MultiLevel, "MtcLr": d_log.MtcLr}

TRAIN_KWARGS: dict = {"n_epochs": 1000}
OPTIMIZATION_KWARGS: dict = {}


def get_optimizer(model):
    return optim.Adam(model, **OPTIMIZATION_KWARGS)


def get_scheduler(optimizer):
    return schedule.ReduceLROnPlateau(optimizer)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action="store_true")
    args = vars(parser.parse_args())  # convert to dict
    return args


# TODO: check if model init is affected if you use lambdas


def cross_val(adata: ad.AnnData):
    for name, m in MODELS.items():
        outdir = OUTDIR.joinpath(name)
        outdir.mkdir(exist_ok=True)
        model = m()
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer)
        trainer = d_ut.Trainer(
            model,
            **TRAIN_KWARGS,
            optimizer=optimizer,
            scheduler=get_scheduler(optimizer),
        )
        holdout_results = holdout(trainer, adata, split_fns=tt.ADDITIONAL_SPLITS)
        hr_dir = outdir.joinpath("additional_splits")
        hr_dir.mkdir(exist_ok=True)
        te.write_results()
    return


if __name__ == "__main__":
    args = parse_args()
    if args["test"]:
        print("Using test subset")
        adata = ut.training_data_internal_test()
        OUTDIR = OUTDIR.joinpath("test")
        OUTDIR.mkdir(exist_ok=True, parents=True)
    else:
        adata = ut.training_data_internal()
    cross_val(adata)
