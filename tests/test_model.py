#!/usr/bin/env ipython

import anndata as ad
import pytest
from pyhere import here
from rpy2.robjects.packages import importr
from sklearn.ensemble import RandomForestClassifier
from too_predict.model import AlrBase, RandomForestPred
from too_predict.transformer import Transformer
from too_predict.utils import ref_feature_lists_internal, training_data_internal_test

adata = training_data_internal_test()
refs, features = ref_feature_lists_internal()


@pytest.mark.skip(reason="Done")
def test_base():
    transformed = Transformer("robust_clr", None, inplace=False).fit_transform(adata)
    rf = RandomForestPred()
    results = rf.cross_validate(transformed)
    assert "fold" in results["report"].columns
    print(results.keys())
    print(results)
    return results


@pytest.mark.skip(reason="Done")
def test_holdout():
    model = RandomForestPred()
    holdouts = {
        "chula": lambda x: (
            x[~x.obs["Project_ID"].str.contains("TARGET"), :],
            x[x.obs["Project_ID"].str.contains("TARGET"), :],
        ),
        "gse": lambda x: (x[: round(len(x) * 2 / 3)], x[round(len(x) * 2 / 3) :]),
    }
    hh = model.holdout(adata, holdouts)
    return hh


@pytest.mark.skip(reason="Done")
def test_alr_estimator():
    model = AlrBase(
        model=RandomForestClassifier(),
        references=refs["variance_feature_list_lowest_20"],
        imputation="plus_one",
        var_col="GENEID",
    )
    model.fit(adata[:30])
    result = model.predict(adata[:30])
    print(model.model.n_fit)
    model = AlrBase(
        model=RandomForestClassifier(),
        references=refs["variance_feature_list_lowest_20"],
        imputation="plus_one",
        var_col="GENEID",
        n_refs=5,
    )
    print(model.model.n_fit)
    model.fit(adata[:30])
    result = model.predict(adata[:30])
    print(model.classes_)
    print(result)
