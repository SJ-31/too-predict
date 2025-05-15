#!/usr/bin/env ipython
import itertools
import math
from abc import abstractmethod
from functools import reduce
from os import replace
from pathlib import Path
from typing import Literal, Sequence, override

import anndata as ad
import h5py
import marsilea as ma
import marsilea.plotter as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import rpy2.robjects as ro
import scanpy as sc
import scipy.cluster as cluster
import scipy.cluster.hierarchy as sch
import scipy.optimize as opt
import scipy.spatial.distance as spd
import seaborn as sns
import seaborn.objects as so
import shap
import skbio.stats.composition as comp
import sklearn.feature_selection as fs
import sklearn.metrics as sm
import sklearn.neighbors as sn
import sklearn.preprocessing as sp
import too_predict._rust_helpers as rh
import too_predict.evaluation as te
import too_predict.explanation as te
import too_predict.filter as fil
import too_predict.go_utils as gu
import too_predict.model as tm
import too_predict.plotting as tp
import too_predict.r_utils as ru
import too_predict.recoder as rt
import too_predict.utils as ut
from joblib import Parallel, delayed, parallel
from matplotlib.figure import Figure
from pyhere import here
from rpy2.robjects.packages import importr
from scipy import sparse
from scipy.stats import mode
from sklearn.linear_model import LogisticRegressionCV
from too_predict._train_utils import MODELS, read_model_spec
from too_predict.corrector import Corrector
from too_predict.transformer import Transformer

# #  --- CODE BLOCK ---
#
base = importr("base")
ensembldb = importr("ensembldb")
obs = pd.read_csv(here("data", "training_data_obs.csv"))

hg38 = here("data", "Homo_sapiens.GRCh38.113.sqlite")
adata = ut.training_data_internal_test()


# #  --- CODE BLOCK ---

spc = MODELS["clr_random_forest_edger"]

F, M, T, B, E, C = read_model_spec(spc)
adata.obs.loc[:, "not_primary"] = adata.obs["Sample_Type"] != "primary"

adata = adata[
    adata.obs["Sample_Type"].isin(["primary", "metastatic", "primary_blood"]), :
]

# filtered = F.fit_transform(adata)
# transformed = T.fit_transform(filtered)

# transformed.obs["foo"] = "foo"
# train, test = ut.train_test_split_ad(transformed)

# counts = adata.X.toarray()

organoid_compare_dir = here("data", "output", "chula_organoid_comparison")

de_enrich_dir = here(organoid_compare_dir, "de_enrichment")
de_df = pd.read_csv(here(de_enrich_dir, "sample_type_top_tags.tsv"), sep="\t")
de_df.loc[:, "absLogFC"] = de_df["logFC"].abs()

sig_pathways = list()
gsa_df = pd.read_csv(here(de_enrich_dir, "gene_sets", "gsa.tsv"), sep="\t")
filtered = gsa_df.query(
    "(`p-value:up-regulated in primary` <= 0.05) | (`p-value:up-regulated in organoid` <= 0.05)"
)
sig_pathways.extend(filtered["set_name"][:10])

top_tags = pd.read_csv(
    here(
        "data",
        "output",
        "chula_organoid_comparison",
        "de_enrichment",
        "sample_type_top_tags.tsv",
    ),
    sep="\t",
)

markers = ut.cell_markers_internal()


def marker_auroc_score(
    adata: ad.AnnData,
    target: str,
    markers: Sequence,
    label_col: str,
    marker_col: str | None = None,
    style: Literal["ovr", "ovo"] = "ovr",
) -> pd.DataFrame:
    labels: pd.Series = adata.obs[label_col]
    counts: pd.Series = labels.value_counts()
    mmask = (
        adata.var[marker_col].isin(markers)
        if marker_col is not None
        else adata.var.index.isin(markers)
    )
    expr_all: np.ndarray = adata.X[:, mmask].toarray()
    pmask: np.ndarray = adata.obs[label_col] == target

    n_positives: int = pmask.sum()
    denoms: dict[str, float] = {}
    rhs: float = n_positives * (n_positives * (n_positives + 1)) / 2
    unique_labels: np.ndarray = labels.unique()

    result_tmp: dict = {}
    for i, marker in enumerate(markers):
        expr: pd.Series = pd.Series(expr_all[:, i], index=labels)
        # Rank by expression in ascending order
        if style == "ovr":
            denoms["ovr"] = denoms.get(
                "ovr", 1 / (counts[target] * (counts.sum() - counts[target]))
            )
            ranks = np.argwhere(expr.sort_values().index == target).flatten()
            result_tmp[marker] = denoms["ovr"] * (ranks.sum() - rhs)
        elif style == "ovo":
            all_aurocs = []
            for u in unique_labels:
                denoms[u] = denoms.get(u, 1 / (counts[target] * counts[u]))
                subset = expr[expr.index == u | expr.index == target]
                ranks = np.argwhere(subset.sort_values().index == target).flatten()
                all_aurocs.append(denoms[u] * (ranks.sum() - rhs))
            result_tmp[marker] = np.mean(all_aurocs)
    return pd.DataFrame(result_tmp)


class MetaMarkers:
    def __init__(
        self,
        datasets: list[ad.AnnData] | dict[str, ad.AnnData],
        known_markers: dict | None | pd.DataFrame = None,
    ) -> None:
        self.marker_df: pd.DataFrame
        if known_markers is not None and isinstance(known_markers, dict):
            self.marker_df = (
                pd.DataFrame(
                    {"target": known_markers.keys(), "gene_id": known_markers.values()}
                )
                .explode("gene_id")
                .set_index("gene_id")
            )
        elif known_markers is not None and isinstance(known_markers, pd.DataFrame):
            self.marker_df = known_markers
        else:
            self.marker_df = pd.DataFrame(
                columns="target", index=pd.Index(name="gene_id")
            )

    def find_markers(self, method: Literal["edgeR", "scanpy"], **kwargs) -> None: ...

    def plot_auroc(
        self,
        targets: Sequence | None = None,
    ) -> None | Figure: ...
