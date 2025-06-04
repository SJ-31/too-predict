#!/usr/bin/env ipython

import anndata as ad
import pandas as pd
import scanpy as sc
import too_predict.plotting as tp
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
    F, M, T, B, E, _ = read_model_spec(spec)
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


def plot_de_enrichment(adata):
    wanted_ttype = ["PAAD", "COAD_READ", "LIHC", "CHOL"]
    wanted_stype = ["primary", "organoid"]
    adata = adata[
        (
            adata.obs["Sample_Type"].isin(wanted_stype)
            & adata.obs["tumor_type"].isin(wanted_ttype)
        )
        | adata.obs["Project_ID"].str.contains("CHULA") :,
    ]
    organoid_compare_dir = here("data", "output", "chula_organoid_comparison")

    de_enrich_dir = here(organoid_compare_dir, "de_enrichment")
    de_df = pd.read_csv(here(de_enrich_dir, "sample_type_top_tags.tsv"), sep="\t")
    de_df.loc[:, "absLogFC"] = de_df["logFC"].abs()

    sig_pathways = list()

    gsa_df = pd.read_csv(here(de_enrich_dir, "gene_sets", "gsa.tsv"), sep="\t")
    gs = ut.gs_internal()

    filtered = gsa_df.query(
        "(`p-value:up-regulated in primary` <= 0.05) | (`p-value:up-regulated in organoid` <= 0.05)"
    )
    sig_pathways.extend(filtered["set_name"][:10])

    n = 20
    top_n = (
        de_df.sort_values("absLogFC", axis=0, ascending=False)
        .dropna(subset="GENENAME", axis="index")["GENEID"][:n]
        .to_list()
    )

    heatmap = tp.mp_plot(
        adata,
        genes=top_n,
        gene_symbols_to_show="GENENAME",
        method="heatmap",
        sample_groupings=["Project_ID", "Sample_Type"],
        var_spacing=0.01,
        var_groupings="GENEBIOTYPE",
        height=9,
        width=9,
    )
    heatmap.figure.savefig(
        here(organoid_compare_dir, "expr_heatmap.png"), bbox_inches="tight"
    )


def plot_shap(adata):
    shap_dir = here("data", "output", "explanations", "shap_importance_25-5-31")
    train_shaps = ad.read_h5ad(shap_dir.joinpath("train_data_shap.h5ad"))
    spec = MODELS["clr_xgboost_edger_per_type_ovp_t_enriched"]
    filter, model, trans, _, _, _ = read_model_spec(spec)
    adata = adata[~adata.obs["Sample_Type"].isna(), :]
    adata.obs.loc[:, "is_organoid"] = adata.obs["Sample_Type"] == "organoid"
    transformed = trans.fit_transform(filter.fit_transform(adata))
    wanted_types = ["PAAD", "CHOL", "LIHC", "COAD_READ", "LUAD", "BRCA"]
    all_ttypes = wanted_types + [""]
    transformed = transformed[transformed.obs["tumor_type"].isin(all_ttypes), :]
    n: int = 30
    for w in wanted_types:
        mean = (
            train_shaps.obsm[f"shap_{w}"]
            .abs()
            .mean(axis=0)
            .sort_values(ascending=False)
        )
        top_n = mean[:n].index
        hmap = tp.mp_plot(
            transformed,
            genes=top_n,
            sample_groupings=["tumor_type", "is_organoid"],
            method="heatmap",
            gene_symbols="GENEID",
            gene_symbols_to_show="GENENAME",
        )
        hmap.figure.set_size_inches(12, 15)
        hmap.figure.savefig(
            shap_dir(f"{w}_train_most_important.png"), bbox_inches="tight"
        )


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

    plot_shap(adata)
    # for name, data in to_plot.items():
    #     # Plot some count data
    #     plot_helper(
    #         name,
    #         adata,
    #         style=data.get("style", DEFAULT_STYLE),
    #         subset=data.get("subset", DEFAULT_SUBSET),
    #         y=data.get("y", "tumor_type"),
    #         colors=data.get("colors"),
    #     )
    # plot_de_enrichment(adata)
