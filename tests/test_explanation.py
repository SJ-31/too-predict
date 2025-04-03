#!/usr/bin/env ipython

import shap
import too_predict.utils as ut
from too_predict._train_utils import MODELS, read_model_spec
from too_predict.explanation import Exp, ExpInterpreter
from too_predict.utils import training_data_internal_test

spc = MODELS["clr_random_forest_edger"]

adata = training_data_internal_test()

f, m, t, b, e = read_model_spec(spc)
filtered = t.fit_transform(adata)
transformed = f.fit_transform(filtered)

m.fit(transformed)
pred = m.predict(transformed)
print(f"N features in filter {len(f.features)}")
print(f"Length of model feature vector: {len(m.get_model().feature_importances_)}")

f.from_feature_importance(m)
print(f"N features in filter {len(f.features)}")

filtered2 = f.fit_transform(transformed)

train, test = ut.train_test_split_ad(filtered2)
m.fit(train)

exp = Exp(m)
exp.fit(train)
strain, _ = exp.shap(lambda x: shap.TreeExplainer(x), summary_plot=False)
exp.fit(test)
stest, _ = exp.shap(lambda x: shap.TreeExplainer(x), summary_plot=False)

# #  --- CODE BLOCK ---
inter = ExpInterpreter(strain, stest)
tdist = inter.label_distances("shap_", "compare")
idist = inter.instance_distances("shap_", "compare")

# pca, fig = inter.instance_pca(
#     "shap_", plot=True, colors=["tumor_type", "usage"], subset=["LGG", "DLBC"]
# )
# fig.show()
