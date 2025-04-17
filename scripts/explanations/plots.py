#!/usr/bin/env ipython

import scanpy as sc
import too_predict.utils as ut
from pyhere import here
from too_predict._train_utils import MODELS, read_model_spec
from too_predict.plotting import plot_adata

OUTDIR = here("data", "output", "explanations")

DEFAULT_SUBSET = (
    "PAAD",
    "COAD-READ",
    "BLCA",
    "CHOL",
    "PAAD",
    "LUAD",
    "ESCA",
    "THYM",
    "STAD",
)

DEFAULT_STYLE = ["Sample_Type", "from_chula"]


def get_project_str(project_id):
    cases = {"CHULA", "TARGET", "GSE", "TCGA", "CPTAC", "CGCI"}
    for c in cases:
        if c in project_id:
            return c
    return "OTHER"


def plot_helper(model_key, adata, style, subset, y, colors):
    spec = MODELS[model_key]
    F, M, T, B, E = read_model_spec(spec)
    adata = F.fit_transform(adata)
    adata = T.fit_transform(adata)
    adata.obs.loc[:, "from_chula"] = adata.obs["Project_ID"].str.contains("CHULA")
    adata.obs.loc[:, "Project_Group"] = adata.obs["Project_ID"].apply(get_project_str)
    adata.obs.loc[:, "size"] = adata.obs["from_chula"].replace({True: 3, False: 1})
    for plot_style in ["pca", "umap"]:
        name = f"{model_key}_{plot_style}.png"
        fig = plot_adata(
            adata,
            y=y,
            subset=subset,
            plot_together=True,
            plot_mode=plot_style,
            style=style,
            alpha=0.8,
            size="size",
            colors=colors,
        )
        fig.tight_layout()
        fig.set_size_inches((15, 10))
        fig.savefig(OUTDIR.joinpath(name))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", default=False, action="store_true")
    args = vars(parser.parse_args())  # convert to dict
    return args


if __name__ == "__main__":
    args = parse_args()
    to_plot = {
        "clr_xgboost_edger": {
            "style": ["Sample_Type"],
            "subset": DEFAULT_SUBSET,
            "colors": ("tumor_type", "Project_Group"),
        },
        "fpkm_random_forest_edger": {},
    }
    if args["test"]:
        adata = ut.training_data_internal_test()
    else:
        adata = ut.training_data_internal()

    for name, data in to_plot.items():
        plot_helper(
            name,
            adata,
            style=data.get("style", DEFAULT_STYLE),
            subset=data.get("subset", DEFAULT_SUBSET),
            y=data.get("y", "tumor_type"),
            colors=data.get("colors"),
        )
