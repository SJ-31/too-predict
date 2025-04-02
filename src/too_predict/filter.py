#!/usr/bin/env ipython

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.neighbors as sn
from scipy import sparse


class Filter:
    """Class for filtering features (genes) in adata objects to a requested subset
    Also re-orders the features in the adata to that of the feature list, filling
    in missing features with 0s.
    """

    def __init__(
        self, features, feature_col="GENENAME", inplace=False, blacklist=None
    ) -> None:
        self.features = features  # Requested features to subset data by
        if blacklist is not None:
            self.features = [f for f in features if f not in blacklist]
        self.feature_col = feature_col
        self.discarded_features = None  # Any features discarded during preprocessing e.g.  # due to not being in enough samples
        self.inplace = inplace
        self.missing_features = []

    def fit(self, adata: ad.AnnData) -> None:
        self.adata = adata.copy() if not self.inplace else adata

    def fit_transform(self, adata: ad.AnnData, _=None) -> ad.AnnData:
        self.fit(adata)
        return self.transform()

    def transform(self, _=None) -> ad.AnnData:
        to_fill = np.zeros((self.adata.shape[0], len(self.features)))
        new_var = pd.DataFrame(index=self.features).merge(
            self.adata.var, how="left", left_index=True, right_on=self.feature_col
        )
        new_var.index = self.features
        lookup: pd.Index = pd.Index(self.adata.var[self.feature_col])
        missing = []
        is_array = isinstance(self.adata.X, np.ndarray)
        counts: np.ndarray = self.adata.X.toarray() if not is_array else self.adata.X
        for i, f in enumerate(self.features):
            try:
                to_fill[:, i] = counts[:, lookup.get_loc(f)]
            except KeyError:
                missing.append(f)
                continue
        transformed = ad.AnnData(
            X=sparse.csc_matrix(to_fill) if not is_array else to_fill,
            var=new_var,
            obs=self.adata.obs,
            uns=self.adata.uns,
            obsm=self.adata.obsm,
        )
        self.missing_features = missing
        if len(missing) > 0:
            print(f"--- WARNING: {len(missing)} missing features!")
        return transformed


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
    counts = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
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


def get_redundant_features(
    adata,
    height: int | float,
    method: str = "correlation",
    col: str = "cluster",
    n_cell_col: str = "n_cells",
) -> tuple[list, list, pd.DataFrame]:
    """Detect and remove potentially redundant features in `adata` by clustering on
        an association metric and selecting a single feature from each

    Parameters
    ----------
    method : association metric to use, includes `correlation`
        (recommended after a log-transformation), `phi_prop` and `rho_prop`.
        The phi and rho proportionality should be used on raw counts
    height : the cutoff at which to determine clusters

    Returns
    -------
    An tuple of [filtered features, features removed,
            updated adata.var df with cluster assignments]

    Notes
    -----
    The representative of a cluster is chosen as the feature found in the most samples
    of `adata`
    """
    if method == "correlation":
        matrix = spd.pdist(np.transpose(adata.X.toarray()), "correlation")  #
        # transpose because we want correlation between features, not the samples
        height = 1 - height  # because correlation distance is 1 - correlation
    elif method == "phi_prop":
        matrix = rh.phi_matrix(adata.X.toarray(), True)
    elif method == "rho_prop":
        matrix = rh.rho_matrix(adata.X.toarray(), True)
    else:
        raise ValueError(f"Method {method} not supported!")
    link_mat = sch.linkage(matrix, method="average")
    clusters = sch.fcluster(link_mat, t=height, criterion="distance")
    adata.var.loc[:, col] = clusters
    print(f"N clusters: {len(set(clusters))}")
    most_cells = (
        adata.var.loc[:, [col, n_cell_col]]
        .groupby(col)
        .idxmax()
        .loc[:, n_cell_col]
        .to_list()
    )
    removed = list(set(adata.var.index) - set(most_cells))
    return most_cells, removed, adata.var.copy()
