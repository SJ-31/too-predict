#!/usr/bin/env ipython
#
import pandas as pd
import too_predict.meta_markers as meta
import too_predict.utils as ut
from pyhere import here

adata = ut.training_data_internal_test()


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
