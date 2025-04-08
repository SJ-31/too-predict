#!/usr/bin/env ipython

#!/usr/bin/env python
import json
from collections import ChainMap
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import obonet
import pandas as pd
import polars as pl
import requests
import scipy.sparse as sparse

import too_predict.utils as ut


class SubsetGO:
    """Class for subsetting the GO DAG based on the GO terms in a sample"""

    def __init__(
        self,
        subset: list[str],
        go_path: str | None = None,
        metadata_path: str | None = None,
    ) -> None:
        mpath = (
            metadata_path
            if metadata_path is not None
            else ut.get_data("go_meta_2025-4-8.csv")
        )
        gpath = go_path if go_path is not None else ut.get_data("go.obo")
        metadata = pd.read_csv(mpath)
        self.roots = {"BP": "GO:0008150", "CC": "GO:0005575", "MF": "GO:0003674"}
        self.sample_gos = subset
        GO: nx.MultiDiGraph = obonet.read_obo(gpath)
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()
        root_map: dict = dict(zip(metadata["accession"], metadata["domain"]))
        self.G.add_nodes_from(self.roots.values())
        for go in self.sample_gos:
            if go in GO:
                paths: list = nx.all_simple_edge_paths(
                    GO, source=go, target=self.roots[root_map[go]]
                )
                is_a = relation_path(paths, "is_a")
                if is_a:
                    self.G.add_edges_from(is_a)
        nd = self._get_node_data()
        self.metadata: pd.DataFrame = pd.merge(nd, metadata, on="accession")
        self.metadata.index = self.metadata["accession"]

    def _get_node_data(self):
        self.successors: dict = {}  # Map of GO_IDs to list of child terms
        level_map: dict = {}
        # GO_IDs found in the sample are True, others (used to link GO terms back to their roots) are False
        # Produces a data frame that maps GO terms to the number of children they have
        node_series = pd.Series(list(self.G.nodes))
        in_sample = node_series.isin(self.sample_gos)
        in_sample.index = node_series
        from_sample = in_sample.to_dict()

        nx.set_node_attributes(self.G, {"from_sample": from_sample})
        for root in self.roots.values():
            current = nx.bfs_tree(self.G, root)
            current.graph["root"] = root
            self.successors = ChainMap(self.successors, all_successors(current))
            level_map = ChainMap(level_map, get_level_map(current))

        df = pd.DataFrame({"accession": in_sample.index, "in_sample": in_sample})
        df.loc[:, "n_children"] = df["accession"].apply(
            lambda x: len(self.successors[x])
        )
        df.loc[:, "level"] = df["accession"].apply(lambda x: level_map[x])
        return df

    def aggregate_to_level(
        self,
        level: int,
        adata: ad.AnnData,
        summarize_method: str = "sum",
        only_in_data: bool = False,
    ) -> ad.AnnData:
        if only_in_data:
            # Requires that terms aggregated to are explicitly observed in adata
            # Might not be necessary since summing the children of a term can
            # give counts to a term that is considered zero
            meta_tmp = self.metadata.loc[self.metadata.index.isin(adata.var.index), :]
        else:
            meta_tmp = self.metadata
        mask = meta_tmp["level"] > level
        agg_to = meta_tmp.loc[mask, :]["accession"]
        index = meta_tmp.index
        adata = adata[:, adata.var.index.isin(index)]
        index = index[index.isin(adata.var.index)]
        mm = np.transpose(
            np.array([index.isin(self.successors[a] + [a]) for a in agg_to])
        ).astype(int)

        was_sparse = sparse.issparse(adata.X)
        unique_cols = list(set(adata.var.columns) - set(meta_tmp.columns))

        new_var = meta_tmp.loc[mask, :].join(adata.var.loc[:, unique_cols])
        counts = np.sum(mm, axis=0)
        if was_sparse:
            matrix = adata.X @ sparse.csr_array(mm)
        else:
            matrix = np.matmul(adata.X, mm)
        if summarize_method == "mean":
            matrix = matrix / counts
        return ad.AnnData(X=matrix, var=new_var, obs=adata.obs)

    def _check_parent(self, to_check: str, ancestors: list):
        """Make sure the parent terms do not contain each other"""
        return all([to_check not in self.successors[e] for e in ancestors])

    def get_predefined(self, path: str, domain: str) -> dict:
        with open(path, "r") as r:
            custom = json.load(r)
        id2group = {}
        reference = self.metadata.filter(pl.col("domain") == domain)
        for group, members in custom.items():
            for member in members:
                if member in reference["accession"]:
                    id2group[member] = group
        return id2group


# * Recoder


class RecodeGO:
    def __init__(
        self,
        id_col: str = "GENEID",
        level: int | None = None,
        agg_method: str = "sum",
        level_agg: str = "sum",
    ) -> None:
        self.adata: ad.AnnData | None = None
        self.id_col: str = id_col
        self.level: int | None = level
        self.agg: str = agg_method
        self.level_agg: str = level_agg
        self.go_map = get_go_data()

    def fit(self, adata: ad.AnnData):
        self.adata = adata.copy()

    def _group_genes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Helper for joining adata.var with the GO metadata
        Returns a tuple of [joined adata.var, adata.var grouped by GO]
        """
        mask = self.adata.var[self.id_col].isin(self.go_map["Gene stable ID"])
        self.adata = self.adata[:, mask]
        with_gos = (
            self.adata.var.reset_index(drop=True)
            .reset_index(names="index")
            .merge(self.go_map, left_on=self.id_col, right_on="Gene stable ID")
        )
        name_map = {
            "GO domain": "domain",
            "GO term name": "term",
            "GO term evidence code": "evidence_code",
            "evidence rating": "evidence_rating",
        }
        with_gos_grouped: pd.DataFrame = (
            with_gos.groupby("GO term accession")[list(name_map.keys())]
            .first()
            .rename(name_map, axis=1)
        )
        with_gos_grouped["accession"] = with_gos_grouped.index
        return with_gos, with_gos_grouped

    def fit_transform(self, adata: ad.AnnData, **kwargs) -> ad.AnnData:
        self.fit(adata)
        return self.transform(**kwargs)

    def transform(self) -> ad.AnnData:
        """Collapse gene expression data into GO

        Parameters
        ----------
        param : adata
        param : summarize_method how to aggregate the gene expression values for a given
            GO term. Options are mean or sum


        Returns
        -------
        An adata object where variables are GO terms
        """
        was_sparse: bool = sparse.issparse(self.adata.X)
        with_gos, with_gos_grouped = self._group_genes()
        go_mm: np.ndarray = (
            with_gos.loc[:, [self.id_col, "GO term accession"]]
            .assign(value=1)
            .pivot(index=self.id_col, columns="GO term accession", values="value")
            .values
        )
        go_mm[np.isnan(go_mm)] = 0
        if was_sparse:
            go_matrix = self.adata.X @ sparse.csr_array(go_mm)
        else:
            go_matrix = np.matmul(self.adata.X, go_mm)

        counts = np.sum(go_mm, axis=0)
        with_gos_grouped.loc[:, "counts"] = counts
        if self.agg == "mean":
            go_matrix = go_matrix / counts
        recoded = ad.AnnData(X=go_matrix, var=with_gos_grouped, obs=self.adata.obs)

        if self.level is not None:
            sgo: SubsetGO = SubsetGO(subset=list(recoded.var.index))
            recoded = sgo.aggregate_to_level(
                self.level, recoded, summarize_method=self.level_agg
            )
        return recoded


# * Helper functions


def relation_path(paths: list[tuple], relation: str) -> list:
    """Find the path from a list of paths (which are edge lists)
    that consists only of `relation`
    """
    result = []
    for path in paths:
        if all(map(lambda x: x[2] == relation, path)):
            for p in path:
                result.append((*p[:2][::-1], p[2]))
        # Note: This changes the direction of the obonet GO graph so that successors are children and predecessors are ancestors
        break
    return result


def join_dict(df: pl.DataFrame, dct: dict, by: str, colname: str):
    if isinstance(dct, ChainMap):
        dct = dict(dct)
    temp_df = pl.DataFrame(dct).melt().rename({"variable": by, "value": colname})
    return df.join(temp_df, on=by)


def all_successors(G: nx.DiGraph) -> dict:
    """Return a dictionary mapping the nodes of G to ALL of their
    children/successors (unlike an adjacency list, which
    only returns the immediate) children
    """
    successors = {}
    for node in G.nodes:
        tree = nx.bfs_tree(G, node)  # Creating a bfs tree for each
        # node restricts the view of G to `node` and everything below it
        children = list(tree.nodes)
        children.remove(node)
        successors[node] = children
    return successors


def get_level_map(G: nx.DiGraph, root=None) -> dict:
    """Return a dictionary mapping the nodes of G to their levels in G"""
    level_map: dict = {}
    if not (root := G.graph.get("root", root)):
        raise ValueError("Root must be specified!")
    for i, layer in enumerate(nx.bfs_layers(G, root)):
        for node in layer:
            level_map[node] = i
    return level_map


# * Misc utilities


def go_from_ebi(terms: list[str]) -> dict:
    base_url = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/"
    encoded = requests.utils.quote(",".join(terms))
    response = requests.get(f"{base_url}{encoded}")
    response.raise_for_status()
    return response.json()


def fill_missing(path: Path) -> None:
    """Fill in `term` and `domain` for GO entries missing it in metadata_path"""
    df = pl.read_csv(path)
    missing = df.filter(pl.col("domain").is_null())
    complete = df.filter(pl.col("domain").is_not_null())
    dmap = {
        "molecular_function": "MF",
        "cellular_component": "CC",
        "biological_process": "BP",
    }
    data: dict = go_from_ebi(list(missing["accession"]))
    tmp = {"accession": [], "term": [], "domain": []}
    for item in data["results"]:
        tmp["accession"].append(item.get("id"))
        tmp["term"].append(item.get("name"))
        domain = dmap.get(item.get("aspect"))
        tmp["domain"].append(domain)
    found = pl.DataFrame(tmp)
    still_missing_path = path.parent.joinpath(f"{path.stem}-MISSING{path.suffix}")
    filtered = missing.filter(~pl.col("accession").is_in(found["accession"]))
    if not filtered.is_empty():
        filtered.write_csv(still_missing_path)
    pl.concat([complete, found]).write_csv(path)


def get_go_data() -> pd.DataFrame:
    go_evidence_codes = {
        "EXP": ("Inferred from Experiment", 1),
        "IDA": ("Inferred from Direct Assay", 1),
        "IPI": ("Inferred from Physical Interaction", 1),
        "IMP": ("Inferred from Mutant Phenotype", 1),
        "IGI": ("Inferred from Genetic Interaction", 1),
        "IEP": ("Inferred from Expression Pattern", 1),
        "HTP": ("Inferred from High Throughput Experiment", 1),
        "HDA": ("Inferred from High Throughput Direct Assay", 1),
        "HMP": ("Inferred from High Throughput Mutant Phenotype", 1),
        "HGI": ("Inferred from High Throughput Genetic Interaction", 1),
        "HEP": ("Inferred from High Throughput Expression Pattern", 1),
        "ISS": ("Inferred from Sequence or Structural Similarity", 3),
        "ISO": ("Inferred from Sequence Orthology", 3),
        "ISA": ("Inferred from Sequence Alignment", 3),
        "ISM": ("Inferred from Sequence Model", 3),
        "IGC": ("Inferred from Genomic Context", 3),
        "IBA": ("Inferred from Biological aspect of Ancestor", 3),
        "IBD": ("Inferred from Biological aspect of Descendant", 3),
        "IKR": ("Inferred from Key Residues", 3),
        "IRD": ("Inferred from Rapid Divergence", 3),
        "RCA": ("Inferred from Reviewed Computational Analysis", 2),
        "TAS": ("Traceable Author Statement", 2),
        "NAS": ("Non-traceable Author Statement", 3),
        "IC": ("Inferred by Curator", 2),
        "ND": ("No biological Data available", 4),
        "IEA": ("Inferred from Electronic Annotation", 3),
    }
    go_evidence_df = pd.DataFrame(
        {
            "GO term evidence code": list(go_evidence_codes.keys()),
            "evidence rating": [i[1] for i in go_evidence_codes.values()],
        }
    )
    go_map = pd.read_csv(ut.get_data("ensembl_go_map_2025-3-20.tsv"), sep="\t")
    go_map = go_map.loc[~go_map["GO term accession"].isna(), :]
    go_map = (
        go_map.merge(go_evidence_df, on="GO term evidence code", how="left")
        .sort_values("evidence rating")
        .groupby(["Gene stable ID", "GO term accession"])
        .first()
        .reset_index()
    )
    return go_map
