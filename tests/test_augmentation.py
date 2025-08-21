#!/usr/bin/env ipython

#!/usr/bin/env ipython
import too_predict.deep.augmentation as d_au
import too_predict.deep.evaluation as d_ev
import too_predict.deep.nns as d_nn
import too_predict.deep.torch_utils as d_ut
import too_predict.transformer as tt
import too_predict.utils as ut
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
from pyhere import here
from too_predict.deep.evaluation import (
    multitask_acc,
    train_test_split_torch,
    train_test_wrapper_torch,
)
from too_predict.imputer import Imputer
from torch.utils.data import DataLoader

import lightning as L

# %%


torch.set_default_dtype(torch.float32)

adata = ut.training_data_internal_test(minimal=True)  # 1000 features
transformer = tt.Transformer("clr", impute_fn=Imputer("plus_one"), inplace=False)
adata = transformer.fit_transform(adata)

# targets = ["Sample_Type", "tumor_type"]

targets = ["tumor_type"]

adset = d_ut.AnnDataset(adata, to_encode=targets)

# train_l, test_l, valid_l = train_test_split_torch(adset, valid=0.1, batch_size=32)
train_l, test_l = train_test_split_torch(adset, batch_size=32)

train, test = ut.train_test_split_ad(adata)
train_adset = d_ut.AnnDataset(train, to_encode=targets)
valid_adset = d_ut.AnnDataset(test, to_encode=targets)

n_features, n_classes = d_ut.data_spec(train_l)


def base_perf(train_l: DataLoader, test_l: DataLoader):
    n_features, n_classes = d_ut.data_spec(train_l)
    base = d_nn.Baseline(n_features, n_classes, max_depth=1)
    base.fit(train_l.dataset)
    res = base.predict_step(test_l.dataset[:][0])
    base_acc = multitask_acc(
        test_l.dataset[:][1],
        res,
        task_names=targets,
        n_classes=n_classes,
    )
    print(f"Base acc: {base_acc}")


base_perf(train_l, test_l)
# %%


def test_cvae():
    trainer = L.Trainer(
        max_epochs=100,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        logger=d_ut.lightning_logger(
            "test_cvae",
            platform="tensorboard",
            save_dir=here("tests", "lightning_logs"),
        ),
    )
    model = d_au.cVAE(in_features=n_features, n_classes_per_task=n_classes)
    trainer.fit(model, train_dataloaders=DataLoader(adset, batch_size=20))
    generated = model.sample_to_dataset(labels=adset[:][1])
    base_perf(DataLoader(generated), DataLoader(adset))
    return model


result = test_cvae()
