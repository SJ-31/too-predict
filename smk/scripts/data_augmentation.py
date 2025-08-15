#!/usr/bin/env python

from pathlib import Path

import anndata as ad
import too_predict.utils as ut
from snakemake.script import snakemake as smk


def get_subset(adata: ad.AnnData, spec: dict) -> ad.AnnData:
    adatas: list = []
    for obs, val_list in spec.items():
        for value, match_type in val_list.items():
            if match_type == "exact":
                adatas.append(adata[adata.obs[obs] == value, :])
            elif match_type == "contains":
                adatas.append(adata[adata.obs[obs].str.contains(value), :])
            else:
                raise ValueError(f"`{match_type}` is an invalid match type!")
    merged = ad.concat(adatas, merge="same")
    merged = merged[merged.obs.duplicated(), :]
    return merged


DA_CONFIG: dict = smk.config["data_augmentation"]
STORAGE: Path = Path(smk.params["storage"])
TEST: bool = smk.config["test"]

# * Augmentation dispatch

# def get_augmentation():

# * Rule handling

if smk.rule == "generate_datasets":
    subsets = []
    if TEST:
        adata = ut.training_data_internal_test()
    else:
        adata = ut.training_data_internal(subset=False)
    for subset, config in DA_CONFIG["subsets"].items():
        outpath = STORAGE.joinpath(subset)
        cur = get_subset(adata, config)
