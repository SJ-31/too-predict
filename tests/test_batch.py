#!/usr/bin/env ipython

from pathlib import Path
from typing import Literal, override

import anndata as ad
import numpy as np
import scipy.sparse as sparse
import too_predict.evaluation as te
import too_predict.model as tm
import too_predict.utils as ut
from pyhere import here
from sklearn.ensemble import RandomForestClassifier
from too_predict._train_utils import MODELS, read_model_spec
from too_predict.corrector import Corrector
from too_predict.evaluation import write_cross_val
from too_predict.model import BatchBase, PredBase, XGBEstimator
from too_predict.transformer import Transformer
from too_predict.utils import training_data_internal

outdir: Path = here("data", "output", "misc")
outdir.mkdir(parents=True, exist_ok=True)


def test_batch():
    adata = training_data_internal()
    adata.obs["is_organoid"] = adata.obs["Sample_Type"] == "organoid"
    bb = BatchBase(
        inner=XGBEstimator(),
        outer=PredBase(RandomForestClassifier()),
        outer_y="is_organoid",
    )
    results = bb.cross_validate_outer(adata, label_col="is_organoid")
    write_cross_val(results, outdir, prefix="batch_outer", cm_prefix="batch_outer")


class PredWithCorrection(tm.PredBase):
    def __init__(
        self,
        model: tm.PredBase,
        corrector: Corrector,
        transformer: Transformer,
        how: Literal["fc_mean"],
        give_direct: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.give_direct: bool = give_direct
        self.corrector: Corrector = corrector
        self.transformer: Transformer = transformer
        self.genewise_params: np.ndarray
        self.how: str = how

    def _fc_mean_adjust(self, original: ad.AnnData) -> ad.AnnData:
        new = original.copy()
        adj = new.X / self.genewise_params
        new.X = adj.toarray() if sparse.issparse(adj) else adj
        return new

    def _transform(self, original: ad.AnnData) -> ad.AnnData:
        if self.how == "fc_mean":
            adj = self._fc_mean_adjust(original)
        else:
            raise ValueError("Not implemented!")
        adj: ad.AnnData = self.transformer.fit_transform(adj)
        return adj

    @override
    def fit(self, X: ad.AnnData, y="tumor_type") -> None:
        corrected = self.corrector.fit_transform(X)
        if self.how == "fc_mean":
            self.genewise_params = np.mean(X.X / corrected.X, axis=0)
            self.genewise_params[np.isnan(self.genewise_params)] = 0
        if self.give_direct:
            corrected = self.transformer.fit_transform(corrected)
            self.model.fit(corrected, y)
        else:
            x = self._transform(X)
            self.model.fit(x, y)

    @override
    def predict(self, X: ad.AnnData) -> np.ndarray:
        x = self._transform(X)
        return self.model.predict(x)

    @override
    def predict_proba(self, X: ad.AnnData) -> np.ndarray:
        x = self._transform(X)
        return self.model.predict_proba(x)


def test_wrapper():
    adata = ut.training_data_internal_test()
    spc = MODELS["clr_xgb3_edger_combat_ref"]
    F, M, T, B, E, C = read_model_spec(spc)
    adata.obs.loc[:, "not_primary"] = adata.obs["Sample_Type"] != "primary"

    adata = adata[
        adata.obs["Sample_Type"].isin(["primary", "metastatic", "primary_blood"]), :
    ]
    filtered = F.fit_transform(adata)
    transformed = T.fit_transform(filtered)

    transformed.obs["foo"] = "foo"
    train, test = ut.train_test_split_ad(filtered)

    C.batch_key = "not_primary"
    wrapper = PredWithCorrection(
        model=M, corrector=C, transformer=T, how="fc_mean", give_direct=True
    )
    wrapper.fit(train)

    y_true = test.obs["tumor_type"]
    proba = wrapper.predict_proba(test)
    pred = wrapper.predict(test)
    set_trace()  # BUG: tracer here
    print((y_true == pred).sum() / len(y_true))
    te.get_all_metrics(y_true, proba, classes=wrapper.classes_)["acc"]
