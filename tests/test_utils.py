#!/usr/bin/env ipython

from functools import reduce

import numpy as np
import pandas as pd
import too_predict.utils as ut
from pyhere import here
from rpy2.robjects.packages import importr
from too_predict._train_utils import (
    MODELS,
    read_model_spec,
)
from too_predict.imbalance import IMPLEMENTED_BALANCE, Balancer

# %%

#
base = importr("base")
ensembldb = importr("ensembldb")
obs = pd.read_csv(here("data", "training_data_obs.csv"))

hg38 = here("data", "Homo_sapiens.GRCh38.113.sqlite")
adata = ut.training_data_internal_test(minimal=True)


spc = MODELS["clr_ranks_mean_xgb_edger_per_type_ovp"]

adata.obs["is_organoid"] = adata.obs["Sample_Type"] != "primary"
adata.obs.loc[:, "not_primary"] = adata.obs["Sample_Type"] != "primary"

adata = adata[
    adata.obs["Sample_Type"].isin(["primary", "metastatic", "primary_blood"]), :
]

F, model, T, B, R, C = read_model_spec(spc, pipeline=False)
filtered = F.fit_transform(adata)
filtered.X = filtered.X.toarray()

train, test = ut.train_test_split_ad(adata[:, :50])

# %%
import anndata as ad
import numpy as np


def get_subset_from_yaml(adata: ad.AnnData, spec: dict) -> ad.AnnData:
    test_masks = []
    for obs, val_dct in spec.items():
        for value, match_type in val_dct.items():
            if match_type == "exact":
                test_masks.append(adata.obs[obs] == value)
            elif match_type == "contains":
                test_masks.append(adata.obs[obs].str.contains(value))
            else:
                raise ValueError(f"`{match_type}` is an invalid match type!")
    test_mask: np.ndarray = reduce(lambda x, y: x | y, test_masks)
    return adata[test_mask, :]


def test_get_subset():
    spec = {
        "tumor_type": {"BRCA": "exact", "COAD_READ": "exact", "DLBC": "exact"},
        "Sample_Type": {"org": "contains"},
    }
    sub = get_subset_from_yaml(adata, spec)
    counts = sub.obs["tumor_type"].value_counts().to_dict()
    original = adata.obs["tumor_type"].value_counts().to_dict()
    print(counts)
    print(original)
    for group in spec["tumor_type"].keys():
        assert counts[group] == original[group]
    assert (
        adata.obs["Sample_Type"].value_counts()["organoid"]
        == sub.obs["Sample_Type"].value_counts()["organoid"]
    )


test_get_subset()
