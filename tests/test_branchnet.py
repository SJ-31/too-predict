#!/usr/bin/env ipython

import too_predict.deep.branchnet as bn
import too_predict.deep.torch_utils as d_ut

#!/usr/bin/env ipython
import too_predict.transformer as tt
import too_predict.utils as ut
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from too_predict.imputer import Imputer
from torch.utils.data import DataLoader

import lightning as L

# %%


torch.set_default_dtype(torch.float32)


adata = ut.training_data_internal_test(minimal=True)  # 1000 features
transformer = tt.Transformer("clr", impute_fn=Imputer("plus_one"), inplace=False)
adata = transformer.fit_transform(adata)

train, test = ut.train_test_split_ad(adata)
encode = ["Sample_Type", "tumor_type"]
train_adset = d_ut.AnnDataset(train, to_encode=encode)
valid_adset = d_ut.AnnDataset(test, to_encode=encode)

trees = ExtraTreesClassifier()
trees.fit(train.X, y=train.obs["tumor_type"])
train_acc = accuracy_score(
    y_pred=trees.predict(train.X), y_true=train.obs["tumor_type"]
)
test_acc = accuracy_score(y_pred=trees.predict(test.X), y_true=test.obs["tumor_type"])

tree = trees.estimators_[0].tree_
all_path_to_parent = []  # list of ...
path_to_parent = []
parents = []


class_prop = []

# trees = bn.BranchNet.fit_trees(train_adset)
# net = bn.BranchNet()
# net.fit(trees, train_adset, epochs=100)

n_features, n_classes = d_ut.data_spec(adata, y=encode)

trainer = L.Trainer(max_epochs=10)

mbn = bn.MultiBranch(n_features, n_classes_per_task=n_classes)

trainer.fit(mbn, train_dataloaders=DataLoader(train_adset))
