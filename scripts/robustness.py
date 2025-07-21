#!/usr/bin/env ipython

import anndata as ad
import too_predict.utils as ut
from pyhere import here
from too_predict._train_utils import MODELS, DummySnake, read_model_spec
from too_predict.deep.torch_utils import data_spec
from too_predict.evaluation import Robustness

try:
    from snakemake.script import snakemake as smk
except ImportError:
    rule = "eff_get_beta"
    if rule == "eff_get_beta":
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

CONFIG = smk.config

DEFAULTS = CONFIG["defaults"]


def prep():
    if smk.config["test"]:
        adata = ut.training_data_internal_test(minimal=True)
    else:
        adata = ut.training_data_internal()
    shifted_test = adata[adata.obs["Sample_Type"] == "organoid", :].copy()
    train, test = ut.train_test_split_ad(adata)
    train.write_h5ad(smk.output[0])
    shifted_test.write_h5ad(smk.output[1])
    test.write_h5ad(smk.output[2])


if smk.rule == "prep":
    prep()
elif smk.rule == "get_beta":
    train = ad.read_h5ad(smk.input[0])
    shifted_test = ad.read_h5ad(smk.input[1])
    standard_test = ad.read_h5ad(smk.input[2])
    n_features, n_classes = data_spec(train, y="tumor_type")
    eff = Robustness(
        train_ad=train,
        shifted_test_ad=shifted_test,
        standard_test_ad=standard_test,
        n_classes=n_classes,
        y_col="tumor_type",
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
        eff.get_beta(spec, save_to=str(smk.output))

    # if smk.rule == "evaluate":
    # TODO: fit, then measure the effective robustness of deep learning models
