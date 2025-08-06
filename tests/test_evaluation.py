#!/usr/bin/env ipython
from typing import Callable

import anndata as ad
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import too_predict.evaluation as te
import too_predict.explanation as ee
import too_predict.utils as ut
from pyhere import here
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from too_predict.deep.nns import HardSharer
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.model import (
    PredBase,
)
from too_predict.plotting import plot_diagonal_matrix
from too_predict.transformer import Transformer

# #  --- CODE BLOCK ---
ADATA = ut.training_data_internal_test()


def test_shapley():
    refs, features = ut.ref_feature_lists_internal(False)
    chosen = features["edgeR_median_lfc_feature_list_1000"]

    filter = Filter(feature_col="GENEID", features=chosen)
    adata = filter.fit_transform(ADATA)
    t = Transformer("clr", impute_fn=Imputer("plus_one"), inplace=False)
    result: ad.AnnData = t.fit_transform(adata)
    model = PredBase(model=RandomForestClassifier())

    train, test = ut.train_test_split_ad(result)
    model.fit(train)
    xgb = model.get_model()
    explainer = shap.TreeExplainer(xgb)

    train_shap, train_v = ee.get_shap_adata(train, explainer, model)
    test_shap, test_v = ee.get_shap_adata(test, explainer, model)
    return train_shap, train_v, test_shap, test_v


robust = te.Robustness(
    shifted_test=here("data", "tests", "effective_robustness", "shifted_test.h5ad"),
    standard_test=here("data", "tests", "effective_robustness", "standard_test.h5ad"),
    train=here("data", "tests", "effective_robustness", "train.h5ad"),
)

hard_share = HardSharer()
trainer = L.Trainer(
    max_epochs=3,
    log_every_n_steps=1,
    enable_progress_bar=False,
    enable_checkpointing=False,
    logger=None,
    callbacks=[],
)
trainer.fit(
    model=hard_share, train_dataloaders=
)


hardshare_spec = {
    "model_fn": lambda x: hard_share,
    "pretrained": True,
    "multitask_key": 1,
}
