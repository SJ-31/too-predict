#!/usr/bin/env ipython
#
import numba as nb
import numpy as np
import pandas as pd
import pytest
import too_predict.meta_markers as meta
import too_predict.utils as ut
from pyhere import here
from sklearn.preprocessing import LabelEncoder

# #  --- CODE BLOCK ---
#
adata = ut.training_data_internal_test()
ENCODER = LabelEncoder()

@pytest.mark.skip(reason="Done")
def test_plot():
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
    top_tags = top_tags.loc[top_tags["GENENAME"] != "NA_character_", :]

    test_markers = top_tags["GENENAME"][:50]
    scores = meta.marker_auroc_score(
        adata,
        target="BRCA",
        markers=test_markers,
        label_col="tumor_type",
        marker_col="GENENAME",
    )
    scores = scores.merge(top_tags, left_index=True, right_on="GENENAME")

    rand_scores = {"main": scores}
    names = ["foo", "bar", "baz", "bat"]
    for i in range(4):
        copy = pd.DataFrame({"GENENAME": ut.RNG.choice(top_tags["GENENAME"], size=50)})
        copy.loc[:, "AUROC"] = ut.RNG.random(size=copy.shape[0])
        copy.loc[:, "logFC"] = ut.RNG.integers(low=-30, high=30, size=copy.shape[0])
        rand_scores[names[i]] = copy

    pfront = meta.maximal_points(scores, "AUROC", "logFC", "GENENAME")
    scores.loc[:, "is_pareto"] = scores["GENENAME"].isin(pfront)

    fig, p = meta.plot_aurocs(rand_scores, id_col="GENENAME", palette="AirNomads")
    fig.show()


# #  --- CODE BLOCK ---
def get_ranks_dummy(
    markers: np.ndarray,
    expr_all: np.ndarray,
    labels: np.ndarray,
    target: str,
    n_pos: int,
    n_neg: int,
):
    result_tmp = {"rank_sum": [], "tie_correction": []}
    for i, _ in enumerate(markers):
        expr: pd.Series = pd.Series(expr_all[:, i], index=labels)
        # Rank by expression in ascending order
        ranks = np.argwhere(expr.sort_values().index == target).flatten()
        result_tmp["rank_sum"].append(ranks.sum())

        unique_expr: pd.Series = expr.value_counts()
        tie_correct: float = (
            np.array([(v**3) - v for v in unique_expr])
            / ((n_pos + n_neg) * (n_pos + n_neg + 1))
        ).sum()
        result_tmp["tie_correction"].append(tie_correct)
    return pd.DataFrame(result_tmp, index=markers)


@pytest.mark.skip(reason="Done")
def test_get_ranks():
    labels = adata.obs["tumor_type"]
    print(nb.typeof(labels.values.astype(str)))
    n_pos = (labels == "BRCA").sum()
    n_neg = len(labels) - n_pos
    rs, tc = meta._get_ranks(
        markers=np.array(range(adata.shape[1])),
        all_expression=adata.X.toarray(),
        labels=ENCODER.fit_transform(labels),
        target=0,
        n_pos=n_pos,
        n_neg=n_neg,
    )
    nb_ans = pd.DataFrame({"a": rs, "b": tc})
    print("nb done")
    old_ans = get_ranks_dummy(
        markers=adata.var.index.values,
        expr_all=adata.X.toarray(),
        labels=labels.values,
        target="BRCA",
        n_pos=n_pos,
        n_neg=n_neg,
    )


def test_calc_auroc():
    MM = meta.MetaMarkers(datasets=[adata], label_col="tumor_type")
    MM.add_markers(adata.var.index[:50])
    aurocs = MM.calc_auroc("dataset_0")
    print(aurocs)
    return aurocs
