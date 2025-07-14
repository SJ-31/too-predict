#!/usr/bin/env ipython

import anndata as ad
import sklearn.metrics as met
import too_predict.model as tm
import too_predict.utils as ut
from pyhere import here

bulkf_file = here("data", "tests", "all_tumors_rnaseq_bulkformer_TEST.h5ad")
bulkf = ad.read_h5ad(bulkf_file)

model = tm.PredBase(model=tm.XGBEstimator())
adata = ut.training_data_internal_test(minimal=True)
train, test = ut.train_test_split_ad(bulkf)
train2, test2 = ut.train_test_split_ad(adata)

model.fit(train, y="tumor_type")
pred = model.predict(test)

acc = met.accuracy_score(y_true=test.obs["tumor_type"], y_pred=pred)

model.fit(train2, y="tumor_type")
pred2 = model.predict(test2)
acc2 = met.accuracy_score(y_true=test2.obs["tumor_type"], y_pred=pred2)
