#!/usr/bin/env ipython

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.neighbors as sn


class Filter:
    def __init__(
        self, features=None, min_cells=2, feature_col="GENENAME", inplace=False
    ) -> None:
        self.min_cells = min_cells
        self.features = features  # Requested features to subset data by
        self.feature_col = feature_col
        self.discarded_features = None  # Any features discarded during preprocessing e.g.  # due to not being in enough samples
        self.inplace = inplace
        self.missing_features = []

    def fit(self, adata: ad.AnnData) -> None:
        self.adata = adata.copy() if not self.inplace else adata

    def fit_transform(self, adata: ad.AnnData) -> ad.AnnData:
        self.fit(adata)
        return self.transform()

    def transform(self) -> ad.AnnData:
        passed_filter: np.ndarray = sc.pp.filter_genes(
            self.adata, min_cells=self.min_cells, inplace=False
        )  # Genes must be nonzero in at least two samples
        self.discarded_features = self.adata.var.loc[~(passed_filter[0]), :]
        self.adata = self.adata[:, passed_filter[0]].copy()
        if self.features is not None:
            self.adata = self.adata[
                :, self.adata.obs[self.feature_col] == self.features
            ]
            missing = set(self.features) - set(self.adata.obs[self.feature_col])
            self.missing_features = missing
            if len(missing) > 0:
                print("--- WARNING: Missing features!")
                print(missing)
                print("---")
        return self.adata


def count_tomek_links(
    adata: ad.AnnData, target_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Count the number of tomek links ocurring in each class of adata.obs['target_col'],
    A tomek link is when two samples of different classes are nearest neighbors

    Returns
    -------
    tuple of [per-class counts of tomek links, pairwise-counts of tomek-links,
        matrix of pairwise counts between classes]

    """
    target = adata.obs[target_col]
    counts = adata.X.toarray()
    neighbors = sn.NearestNeighbors(n_neighbors=2)  # Must get the second closest point
    neighbors.fit(counts)
    _distances, nearest = neighbors.kneighbors(counts)
    nearest_labelled = target.iloc[nearest[:, 1]]

    neighbor_pairs = pd.DataFrame(
        {
            "x": target.index.to_series().reset_index(drop=True),
            "x_label": target.reset_index(drop=True),
            "y": nearest_labelled.index.to_series().reset_index(drop=True),
            "y_label": nearest_labelled.reset_index(drop=True),
        },
    )
    tomek_links = neighbor_pairs.loc[
        neighbor_pairs["x_label"] != neighbor_pairs["y_label"], :
    ]
    suffixes = ["", "_label"]
    for s in suffixes:
        tomek_links.loc[:, f"pair{s}"] = tomek_links[f"x{s}"].combine(
            tomek_links[f"y{s}"], lambda x, y: {x, y}
        )
    tomek_links = tomek_links.loc[~tomek_links["pair"].duplicated(), :]
    # Do not overcount pairs e.g. if point A's nearest neighbor is B, and B's
    # nearest neighbor is A, then they would contribute twice to the class count

    class_counts = tomek_links["x_label"].value_counts()
    # Class counts with samples that are participating in at least one tomek link

    class_totals = target.value_counts()
    formatted_class = pd.DataFrame(
        {
            "class": class_counts.index.to_series(),
            "count": class_counts,
            "class_total": class_totals,
            "percentage": class_counts / class_totals * 100,
        }
    ).reset_index(drop=True)
    pair_counts = tomek_links["pair_label"].value_counts()
    formatted_pairs = pd.DataFrame(
        {
            "pair": pair_counts.index.to_series().apply(lambda x: list(x)),
            "count": pair_counts,
        }
    ).reset_index(drop=True)
    count_matrix = pd.DataFrame(0, index=target.unique(), columns=target.unique())

    for p, c in formatted_pairs.itertuples(index=False):
        x, y = p
        count_matrix.loc[x, y] = c
        count_matrix.loc[y, x] = c

    return formatted_class, formatted_pairs, count_matrix
