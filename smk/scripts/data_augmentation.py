#!/usr/bin/env python

from functools import reduce
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import sklearn.preprocessing as sp
import too_predict.deep.evaluation as d_ev
import too_predict.deep.torch_utils as d_ut
import too_predict.model as tm
import too_predict.utils as ut
from snakemake.script import snakemake as smk
from too_predict._train_utils import default_filter_transform, get_model_fn
from too_predict.deep.nns import Disyak
from too_predict.evaluation import train_test_wrapper
from too_predict.imbalance import TORCH_METHODS, Balancer

DA_CONFIG: dict = smk.config["data_augmentation"]
DL_CONFIG: dict = smk.config["dl"]
TEST: bool = smk.config["test"]
LABEL_COL: str = smk.config["shallow"]["filter"]["label_col"]
USE_TORCH = DA_CONFIG["model"] != "PredBase"


# * Utilities


def get_subset_from_yaml(adata: ad.AnnData, spec: dict) -> ad.AnnData:
    test_masks = []
    for obs, val_dct in spec.items():
        for value, match_type in val_dct.items():
            if match_type == "exact":
                test_masks.append(adata.obs[obs] == value)
            elif match_type == "contains":
                test_masks.append(adata.obs[obs].str.contains(value))
            else:
                raise ValueError(f"`{match_type}` is an invalid match type!")
    test_mask: np.ndarray = reduce(lambda x, y: x | y, test_masks)
    return adata[test_mask, :]


# You could do this all in `holdout`, but don't want to read everything in all at once
def evaluate(
    name: str, train: ad.AnnData, test, validation, encoders: dict[str, sp.LabelEncoder]
):
    result: dict
    if not USE_TORCH:
        model = tm.PredBase(tm.XGBEstimator())
        result, _ = train_test_wrapper(
            model=model,
            maybe_split=(train, test),
            label_col=LABEL_COL,
            set_label=name,
            pre_split=True,
            minimal=False,
        )
    else:
        result = d_ev.train_test_wrapper_torch(
            module_cls=get_model_fn(DA_CONFIG["model"]),
            maybe_split=(train, test),
            to_encode=LABEL_COL,
            validation=validation,
            n_classes=outpath.joinpath("n_classes.txt").read_text(),
            in_features=outpath.joinpath("in_features.txt").read_text(),
            set_label=name,
            trainer_kwargs=DL_CONFIG["trainer"],
            loader_kwargs=DL_CONFIG["dataloader"],
            module_kwargs=DL_CONFIG["models"]["dl"]
            .get(DA_CONFIG["model"], {})
            .get("params", {}),
        )
    return result


# * Rule handling

# ** Generate data
if smk.rule == "generate_datasets":
    subsets = []
    storage: Path = Path(smk.params["store"])
    result_dir: Path = Path(smk.params["result_dir"])
    if TEST:
        adata = ut.training_data_internal_test()
    else:
        adata = ut.training_data_internal(subset=False)
    for subset_name, config in smk.params["subsets"].items():
        outpath = storage.joinpath(subset_name)
        subset_ad = get_subset_from_yaml(adata, config)
        preprocess = tm.Pipeline(steps=[*default_filter_transform(smk.config)])
        adata, test = ut.train_test_split_ad(
            subset_ad, random_state=smk.config["random_state"]
        )
        adata = preprocess.fit_transform(adata)
        test = preprocess.transform(test)
        in_features, n_classes = d_ut.data_spec(
            adata, y=adata.obs[LABEL_COL].astype(str)
        )
        adata.write_h5ad(outpath.joinpath("baseline_train.h5ad"))
        test.write_h5ad(outpath.joinpath("test.h5ad"))

        outpath.joinpath("in_features.txt").write_text(str(in_features))
        outpath.joinpath("n_classes.txt").write_text(str(n_classes))

        for method, conf in DA_CONFIG["augmentations"].items():
            if method == "baseline" or (
                conf is not None and not conf.get("skip", True)
            ):
                continue
            if method in TORCH_METHODS:
                dl_config: dict = smk.config["dl"]
                trainer_kwargs = dl_config["trainer"]
                loader_kwargs = dl_config["dataloader"]
                logger_kwargs = {
                    "exp_name": f"{method}_log",
                    "platform": "tensorboard",
                    "save_dir": result_dir.joinpath(subset_name),
                }
            else:
                trainer_kwargs = None
                loader_kwargs = None
                logger_kwargs = None
            conf = ut.if_none(conf, {})
            params: dict = ut.if_none(conf.get("params", {}), {})
            balancer = Balancer(
                method,
                trainer_kwargs=trainer_kwargs,
                logger_kwargs=logger_kwargs,
                loader_kwargs=loader_kwargs,
                sampling_strategy=conf.get("sampling_strategy", "auto"),
                n=conf.get("n", None),
                **params,
            )
            augmented = balancer.fit_transform(adata, y=LABEL_COL)
            augmented.write_h5ad(outpath.joinpath(f"{method}_train.h5ad"))

# ** Evaluate
if smk.rule == "evaluate":
    # TODO: need to update this to use shared encoders, or the metrics won't be correct
    encoders = d_ut.AnnDataset.fit_encoders(
        ut.training_data_internal(backed=True, only_obs=True)
        if not TEST
        else ut.training_data_internal_test(),
        to_encode=[LABEL_COL],
    )
    wanted_metrics = ["acc", "kappa", "auroc", "aupr"]
    results = {"dataset": [], "augmentation": []}
    results.update({w: [] for w in wanted_metrics})
    dfs = []
    outdir: Path = Path(smk.params["outdir"])
    subset_dirs = {Path(p).parent for p in smk.input}
    subset2train: dict = {
        p: [
            Path(train_file) for train_file in smk.input if Path(train_file).parent == p
        ]
        for p in subset_dirs
    }

    for subset_path, train_paths in subset2train.items():
        subset_name = subset_path.stem
        test = ad.read_h5ad(subset_path.joinpath("test.h5ad"))
        cur_outdir = outdir.joinpath(subset_name)
        if USE_TORCH:
            test = d_ut.AnnDataset(test, to_encode=LABEL_COL, encoders=encoders)
        for train_path in train_paths:
            name = train_path.stem.replace("_train", "")
            cur_outfile = cur_outdir.joinpath(f"{name}_result.csv")
            if cur_outfile.exists():
                df = pd.read_csv(cur_outfile)
            else:
                adata = ad.read_h5ad(train_path)
                train, valid = ut.train_test_split_ad(
                    adata,
                    random_state=smk.config["random_state"],
                    test_size=0.1,
                )
                if USE_TORCH:
                    train, valid = (
                        d_ut.AnnDataset(train, to_encode=LABEL_COL, encoders=encoders),
                        d_ut.AnnDataset(valid, to_encode=LABEL_COL, encoders=encoders),
                    )
                current = results.copy()
                current["dataset"].append(subset_name)
                current["augmentation"].append(name)
                metrics = evaluate(
                    name, train=train, test=test, validation=valid, encoders=encoders
                )
                for m in wanted_metrics:
                    current[m].append(metrics[m])
                df = pd.DataFrame(current)
                if USE_TORCH:
                    df = d_ut.tensor_cols_to_float(pd.DataFrame(current))
                df.to_csv(cur_outfile, index=False)
            dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(smk.output["final_csv"], index=False)
