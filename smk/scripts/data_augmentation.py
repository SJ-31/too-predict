#!/usr/bin/env python

from pathlib import Path

import anndata as ad
import pandas as pd
import too_predict.deep.evaluation as d_ev
import too_predict.deep.torch_utils as d_ut
import too_predict.model as tm
import too_predict.utils as ut
from snakemake.script import snakemake as smk
from too_predict._train_utils import default_filter_transform, get_model_fn
from too_predict.deep.nns import Disyak
from too_predict.evaluation import train_test_wrapper
from too_predict.imbalance import Balancer

DA_CONFIG: dict = smk.config["data_augmentation"]
DL_CONFIG: dict = smk.config["dl"]
STORAGE: Path = Path(smk.params["storage"])
TEST: bool = smk.config["test"]
LABEL_COL: str = smk.config["shallow"]["filter"]["label_col"]
USE_TORCH = DA_CONFIG["model"] != "PredBase"


# * Utilities


def get_subset_from_yaml(adata: ad.AnnData, spec: dict) -> ad.AnnData:
    adatas: list = []
    for obs, val_list in spec.items():
        for value, match_type in val_list.items():
            if match_type == "exact":
                adatas.append(adata[adata.obs[obs] == value, :])
            elif match_type == "contains":
                adatas.append(adata[adata.obs[obs].str.contains(value), :])
            else:
                raise ValueError(f"`{match_type}` is an invalid match type!")
    merged = ad.concat(adatas, merge="same")
    merged = merged[merged.obs.duplicated(), :]
    return merged


# You could do this all in `holdout`, but don't want to read everything in all at once
def evaluate(name: str, train: ad.AnnData, test, validation):
    result: dict
    if not USE_TORCH:
        model = tm.Pipeline(
            [*default_filter_transform(smk.config)], predictor=tm.XGBClassifier()
        )
        result = train_test_wrapper(
            pipeline=model,
            maybe_split=(train, test),
            label_col=LABEL_COL,
            set_label=name,
            pre_split=True,
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
    if TEST:
        adata = ut.training_data_internal_test()
    else:
        adata = ut.training_data_internal(subset=False)
    for subset_name, config in DA_CONFIG["subsets"].items():
        outpath = STORAGE.joinpath(subset_name)
        subset_ad = get_subset_from_yaml(adata, config)
        preprocess = tm.Pipeline(steps=[*default_filter_transform(smk.config)])
        adata, test = ut.train_test_split_ad(
            subset_ad, random_state=smk.config["random_state"]
        )
        adata = preprocess.fit_transform(adata)
        test = preprocess.transform(test)
        in_features, n_classes = d_ut.data_spec(adata, y=adata.obs[LABEL_COL])
        adata.write_h5ad(outpath.joinpath("baseline_train.h5ad"))
        test.write_h5ad(outpath.joinpath("test.h5ad"))

        outpath.joinpath("in_features.txt").write_text(in_features)
        outpath.joinpath("n_classes.txt").write_text(n_classes)

        for method, conf in DA_CONFIG["augmentations"].items():
            if not conf.get("enabled", True) or method == "baseline":
                continue
            balancer = Balancer(method, **conf.get("params", {}))
            augmented = balancer.fit_transform(adata, y=LABEL_COL)
            augmented.write_h5ad(outpath.joinpath(f"{method}_train.h5ad"))

# ** Evaluate
if smk.rule == "evaluate":
    wanted_metrics = ["acc", "kappa", "mcc", "fowlkes_mallows", "auroc"]
    results = {"dataset": [], "augmentation": []}
    results.update({w: [] for w in wanted_metrics})
    dfs = []
    for dir in smk.input:
        path: Path = Path(str(dir))
        subset_name = path.stem
        test = ad.read_h5ad(path.joinpath("test.h5ad"))
        if USE_TORCH:
            test = d_ut.AnnDataset(test, to_encode=LABEL_COL)
        for adata_path in path.glob(pattern="*_train.h5ad"):
            name = adata_path.stem.replace("_train", "")
            cur_outfile = Path(f"{name}_result.csv")
            if cur_outfile.exists():
                df = pd.read_csv(cur_outfile)
            else:
                adata = ad.read_h5ad(adata_path)
                train, valid = ut.train_test_split_ad(
                    adata,
                    random_state=smk.config["random_state"],
                    test_size=0.1,
                )
                current = results.copy()
                current["dataset"].append(subset_name)
                current["augmentation"].append(name)
                metrics = evaluate(name, train=train, test=test, validation=valid)
                for m in wanted_metrics:
                    current[m].append(metrics[m])
                df = pd.DataFrame(current)
                df = d_ut.tensor_cols_to_float(pd.DataFrame(results))
                df.to_csv(path.joinpath(cur_outfile, index=False))
            dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(smk.output["final_csv"], index=False)
