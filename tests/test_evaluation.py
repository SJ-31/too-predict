#!/usr/bin/env ipython
from typing import Callable

import anndata as ad
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
from too_predict._train_utils import default_filter_transform, get_model_fn
from too_predict.deep.nns import HardSharer
from too_predict.filter import Filter
from too_predict.imputer import Imputer
from too_predict.model import (
    Pipeline,
    PredBase,
)
from too_predict.plotting import plot_diagonal_matrix
from too_predict.transformer import Transformer

import lightning as L

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


def test_holdout():
    dct = {}
    result = te.holdout(
        pipeline_fn=lambda: Pipeline(
            [*default_filter_transform()], predictor=PredBase(RandomForestClassifier())
        ),
        data={"foo": ut.train_test_split_ad(ADATA)},
        label_col="tumor_type",
        save_models=dct,
    )
    print(dct)
    return result


result = test_holdout()

this = Pipeline(
    [*default_filter_transform()], predictor=PredBase(RandomForestClassifier())
)
