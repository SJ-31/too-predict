#!/usr/bin/env ipython

from collections.abc import Callable

import anndata as ad
import lightning as L
import pandas as pd
import too_predict.utils as ut
import torch
from pyhere import here
from too_predict._train_utils import MODELS, DummySnake, get_model_fn, read_model_spec
from too_predict.deep.evaluation import Baseline
from too_predict.deep.torch_utils import AnnDataset, data_spec
from too_predict.evaluation import Robustness
from torch.utils.data import DataLoader

try:
    from snakemake.script import snakemake as smk
except ImportError:
    rule = "get_beta"
    if rule == "get_beta":
        smk = DummySnake(
            rule=rule,
            configfile=here("smk", "env.yaml"),
            input=[
                here("data", "tests", "effective_robustness", "train.h5ad"),
                here("data", "tests", "effective_robustness", "standard_test.h5ad"),
                here("data", "tests", "effective_robustness", "shifted_test.h5ad"),
            ],
            output=here("data", "tests", "effective_robustness", "out.pkl"),
        )

torch.set_default_dtype(torch.float32)

CONFIG = smk.config

DEFAULTS = CONFIG["defaults"]

MODEL_SPEC: dict = CONFIG["models"]["dl"]
DEEP_MODELS = [m for m in MODEL_SPEC.keys() if not MODEL_SPEC[m].get("skip")]
LOADER_KWARGS = smk.config["defaults"]["dl"]["dataloader"]


ID_KWARGS = {"y_col": "tumor_type", "y_idx": 1}

# * Prepare
if smk.rule == "prep":
    if smk.config["test"]:
        adata = ut.training_data_internal_test(minimal=True)
    else:
        adata = ut.training_data_internal()
    shifted_test = adata[adata.obs["Sample_Type"] == "organoid", :].copy()
    adata = adata[adata.obs["Sample_Type"] != "organoid", :]
    train, test = ut.train_test_split_ad(adata)
    train.write_h5ad(smk.output[0])
    shifted_test.write_h5ad(smk.output[1])
    test.write_h5ad(smk.output[2])
# * Get beta
elif smk.rule == "get_beta":
    train = ad.read_h5ad(smk.input[0])
    shifted_test = ad.read_h5ad(smk.input[1])
    standard_test = ad.read_h5ad(smk.input[2])
    n_features, n_classes = data_spec(train, y=["tumor_type"])
    eff = Robustness(
        train_ad=train,
        shifted_test_ad=shifted_test,
        standard_test_ad=standard_test,
        n_classes=n_classes,
        **ID_KWARGS,
    )
    if smk.rule == "get_beta":
        spec = [
            {
                "name": n,
                "model_fn": lambda: read_model_spec(x, pipeline=True),
                "train_fn": "fit",
                "adata": True,
            }
            for n, x in MODELS.items()
        ]
        if smk.config["test"]:
            spec = spec[:2]
        eff.get_beta(spec, save_to=str(smk.output))
# * Fit DL models and save ckpts
elif smk.rule == "fit_deep":
    train = AnnDataset(
        ad.read_h5ad(smk.input["train"], backed=True),
        to_encode=smk.config["multi_labels"],
    )
    n_features, n_classes = data_spec(train)
    for name in DEEP_MODELS:
        if name == "baseline":
            continue
        spec = MODEL_SPEC[name]
        kwargs = spec.get("params")
        kwargs.update({"n_classes_per_task": n_classes, "in_features": n_features})
        model = get_model_fn(name)(**kwargs)
        trainer_kwargs = smk.config["defaults"]["dl"]["trainer"]
        trainer_kwargs["enable_checkpointing"] = False
        if smk.config["test"]:
            trainer_kwargs["max_epochs"] = 1
        trainer = L.Trainer(**trainer_kwargs)
        trainer.fit(model, train_dataloaders=DataLoader(train, **LOADER_KWARGS))
        print(model.state_dict().keys())
        torch.save(model, smk.output.get(name))
        # trainer.save_checkpoint(smk.output.get(name))
# * Evaluate saved DL models
elif smk.rule == "evaluate":
    train = ad.read_h5ad(smk.input["train"], backed=True)
    n_features, n_classes = data_spec(train, y=["tumor_type"])
    shifted_test_ad = ad.read_h5ad(smk.input["shifted_test"], backed=True)
    standard_test_ad = ad.read_h5ad(smk.input["standard_test"], backed=True)
    train_adset = AnnDataset(train, to_encode=smk.config["multi_labels"])
    eff = Robustness(
        train=train_adset,
        shifted_test=AnnDataset(shifted_test_ad, to_encode=smk.config["multi_labels"]),
        standard_test=AnnDataset(
            standard_test_ad, to_encode=smk.config["multi_labels"]
        ),
        n_classes=n_classes[0],
        beta_path=smk.input["beta_path"],
        **ID_KWARGS,
    )
    n_features, n_classes = data_spec(train, y=CONFIG["multi_labels"])
    baseline = Baseline(in_features=n_features, n_classes_per_task=n_classes)
    baseline.fit(train_adset)
    baseline_spec = {
        "model_fn": lambda: baseline,
        "name": "baseline",
        "pretrained": True,
        "multitask_key": 1,
    }
    data = {"name": [], "effective_robustness": [], "relative_robustness": []}
    for name, ckpt_path in smk.params.items():
        # cls: Callable = get_model_fn(name)
        # model: L.LightningModule = cls(
        #     in_features=n_features, n_classes_per_task=n_classes
        # )
        # trainer = L.Trainer(fast_dev_run=1)
        # trainer.fit(model, train_dataloaders=DataLoader(train_adset))
        #
        # BUG: would rather load the state dict, but been having problems with
        # lazy initialization
        model = torch.load(ckpt_path, weights_only=False)
        spec = {
            "name": name,
            "model_fn": lambda: model,
            "pretrained": True,
            "multitask_key": 1,
        }
        data["effective_robustness"].append(eff.effective_robustness(spec))
        data["relative_robustness"].append(
            eff.relative_robustness(ispec=baseline_spec, mspec=spec)
        )
        data["name"].append(name)

    pd.DataFrame(data).to_csv(smk.output[0])
