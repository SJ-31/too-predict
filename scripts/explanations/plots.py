#!/usr/bin/env ipython

import too_predict.utils as ut
from pyhere import here
from too_predict._train_utils import MODELS, read_model_spec
from too_predict.plotting import plot_adata

OUTDIR = here("data", "output", "explanations")


def plot_helper(model_key, adata):
    spec = MODELS[model_key]
    F, M, T, B, E = read_model_spec(spec)
    adata = F.fit_transform(adata)
    adata = T.fit_transform(adata)
    adata.obs.loc[:, "from_chula"] = adata.obs["Project_ID"].str.contains("CHULA")
    adata.obs.loc[:, "size"] = adata.obs["from_chula"].replace({True: 3, False: 1})
    for plot_style in ["pca", "umap"]:
        name = f"{model_key}_{plot_style}.png"
        fig = plot_adata(
            adata,
            y="tumor_type",
            subset=(
                "PAAD",
                "COAD-READ",
                "BLCA",
                "CHOL",
                "PAAD",
                "LUAD",
                "ESCA",
                "THYM",
                "STAD",
            ),
            plot_together=True,
            plot_mode=plot_style,
            style=["Sample_Type", "from_chula"],
            alpha=0.8,
            size="size",
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
    to_plot = ["clr_xgboost_edger"]
    if args["test"]:
        adata = ut.training_data_internal_test()
    else:
        adata = ut.training_data_internal()

    for name in to_plot:
        plot_helper(name, adata)
