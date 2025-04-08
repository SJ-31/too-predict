#!/usr/bin/env ipython

#!/usr/bin/env python
import json
import sys
from collections import ChainMap
from pathlib import Path

import networkx as nx
import obonet
import pandas as pd
import polars as pl
import requests

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
        metadata = pl.read_csv(mpath)
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
        self.metadata = self.__get_node_data().join(
            metadata.select("accession", "term", "domain"), on="accession"
        )

    def __get_node_data(self):
        self.successors: dict = {}  # Map of GO_IDs to list of child terms
        level_map: dict = {}
        from_sample: dict = {}
        # GO_IDs found in the sample are True, others (used to link GO terms back to their roots) are False
        # Produces a data frame that maps GO terms to the number of children they have
        for node in self.G.nodes():
            if node in self.sample_gos:
                from_sample[node] = True
            else:
                from_sample[node] = False
        nx.set_node_attributes(self.G, {"from_sample": from_sample})
        for root in self.roots.values():
            current = nx.bfs_tree(self.G, root)
            current.graph["root"] = root
            self.successors = ChainMap(self.successors, all_successors(current))
            level_map = ChainMap(level_map, level_map(current))

        return (
            (
                pl.DataFrame(from_sample)
                .melt()
                .rename({"variable": "accession", "value": "in_sample"})
            )
            .with_columns(
                n_children=pl.col("accession").map_elements(
                    lambda x: len(self.successors[x]), return_dtype=pl.Int16
                ),
                level=pl.col("accession").map_elements(
                    lambda x: level_map[x], return_dtype=pl.Int16
                ),
            )
            .sort("n_children", descending=True)
        )

    def __check_parent(self, to_check: str, ancestors: list):
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

    def get_parents(self, domain: str, n: int = 18, show=True, min_depth=2, pre=""):
        """Select higher-level parent GO terms from the specified domain that partition the domain into `n` bins.
        Goal is to map all child terms to some higher-level term to make for more concise summarization
        higher-level terms are selected by the number of children they have, as well as the specified depth. They can also be pre-specified with the `pre` argument

        :return: A dictionary of the following
        `map`: Map of GO terms in the specified sub-ontology to their assigned higher-level terms
        `parents`: Map of chosen parents to their terms
        `unassigned`: GO terms in the sub-ontology graph that are not children of any of the chosen parents. Happens when chosen parents have few children
        `pre`: path to a json file containing pre-defined groups (mapping a GO id or group name to specific terms) that will override other mappings
        """
        children_per_cat = self.metadata.shape[0] / n
        filtered = self.metadata.filter(
            (
                (pl.col("domain") == domain)
                & (pl.col("n_children") <= children_per_cat)
                & (pl.col("level") >= min_depth)
            )
        ).sort("n_children", descending=True)
        parents: list = []
        for go_id, *_ in filtered.iter_rows():
            if self.__check_parent(go_id, parents):
                parents.append(go_id)
            if len(parents) == n:
                break
        parent_df = filtered.filter(pl.col("accession").is_in(parents))
        if show:
            print(parent_df)
        child_to_parent: dict = self.get_predefined(pre, domain) if pre else {}
        unassigned: set = set(filtered["accession"])
        for parent in parents:
            for child in self.successors[parent]:
                if child not in child_to_parent:
                    child_to_parent[child] = parent
            child_to_parent[parent] = parent
        unassigned = unassigned - child_to_parent.keys()
        return {
            "map": child_to_parent,
            "parents": dict(zip(parent_df["accession"], parent_df["term"])),
            "unassigned": unassigned,
        }


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


def level_map(G: nx.DiGraph, root=None) -> dict:
    """Return a dictionary mapping the nodes of G to their levels in G"""
    level_map: dict = {}
    if not (root := G.graph.get("root", root)):
        raise ValueError("Root must be specified!")
    for i, layer in enumerate(nx.bfs_layers(G, root)):
        for node in layer:
            level_map[node] = i
    return level_map


def to_json(
    go_path: str,
    sample_path: str,
    go_info_path: str,
    outdir: str,
    n_groups: int,
    predefined: str = None,
):
    data = {"CC": None, "BP": None, "MF": None}
    S = SubsetGO(go_path, sample_path, go_info_path)
    for sub in data.keys():
        rep = S.get_parents(sub, n=n_groups, pre=predefined)
        data[sub] = {
            "map": rep["map"],
            "unassigned": list(rep["unassigned"]),
        }
    with open(f"{outdir}/go_parents.json", "w") as j:
        json.dump(data, j)


def into_domain(gos: list, mapping: dict) -> dict:
    partitioned: dict = {"CC": [], "BP": [], "MF": []}
    for go in gos:
        lookup: str = mapping[go]
        if lookup != "NA":  # Check if term is obsolete
            partitioned[lookup].append(go)
    return partitioned


def get_parents(
    source: str,
    go_terms: list,
    rep_map: dict,
    term_map: dict,
    domain_map: dict,
    priority: list = None,
) -> tuple[dict, dict]:
    """Obtain higher level parent GO terms from a term list (i.e. terms of a protein)

    Args:
        source (string): where do these ids come from?
        go_terms (list): list of GO ids to get higher-level terms from
        rep_map (dict): mapping of GO ids to their representative parent terms
        domain_map (dict): map of GO ids to their sub-ontology
        domain_map (dict): map of GO ids to their English terms
    """
    grouped = into_domain(go_terms, domain_map)
    terms = pl.DataFrame()
    missing = {"CC": 0, "BP": 0, "MF": 0}
    found_special = False
    for o in missing.keys():
        cur_group = grouped[o]
        reduced = {}
        cur_map = rep_map[o]
        for id in cur_group:
            found = cur_map["map"].get(id)
            if found and priority and found in priority:
                reduced[found] = sys.maxsize
                found_special = True
            elif found:
                reduced[found] = reduced.get(found, 0) + 1
            else:
                reduced[id] = 1
        if not reduced:
            continue
        temp = (
            pl.DataFrame(reduced)
            .melt()
            .rename({"variable": "accession", "value": "count"})
            .with_columns(domain=pl.lit(o))
            .sample(n=len(reduced), shuffle=True)
            .sort("count", descending=True)
        )
        # Shuffling (using sample) is just a precaution in case all the entries have the same "count"
        terms = pl.concat([terms, temp], how="vertical")
    if terms.is_empty():
        # print(f"Warning: obsolete terms {list(go_terms)}")
        return {"CC": "unknown", "BP": "unknown", "MF": "unknown"}, missing
    aggregated = terms.group_by("domain", maintain_order=True).agg(
        pl.col("accession").first()
    )
    result = dict(zip(aggregated["domain"], aggregated["accession"]))
    if found_special:
        print(result)
    for k in missing.keys():
        v = result.get(k)
        if not v:
            result[k] = "unknown"
            missing[k] = 1
            continue
        cur_map = rep_map[k]
        if (v not in cur_map["map"] or v in cur_map["unassigned"]) and (
            not priority or v not in priority
        ):
            result[k] = "other"
        else:
            if ":" in v:  # Check that the key is actually a GO id, and
                # not one of the user-defined groups
                result[k] = term_map[v]
            else:
                print(f"from end loop {v}")
                result[k] = v
    return result, missing


def find_go_parents(
    combined_results: str, go_info_path: str, parents_path: str, priority_path: str = ""
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find higher-level GO terms of identified proteins in a results file

    Args:
        combined_results (str): path to results file
        go_info_path (str): path to file containing GO term metadata
        parents_path (str): path to json file mapping GO ids to their assigned parent terms
        output (str): output file name
    """
    info: pl.DataFrame = pl.read_csv(go_info_path, separator="\t")
    id2domain = dict(zip(info["accession"], info["domain"]))
    id2term = dict(zip(info["accession"], info["term"]))
    data = pl.read_csv(combined_results, separator="\t", null_values="NA")
    shape_before: tuple = data.shape
    has_go = data.filter(pl.col("accession").is_not_null())
    no_go = data.filter(pl.col("accession").is_null())
    if priority_path:
        with open(priority_path, "r") as j:
            priority = list(json.load(j).keys())
    else:
        priority = []
    with open(parents_path, "r") as j:
        rmap: dict = json.load(j)
    id_go: pl.DataFrame = (
        has_go.select("ProteinId", "accession", "GO_counts")
        .with_columns(
            accession=pl.col("accession").map_elements(
                lambda x: x.split(";"), return_dtype=pl.List(pl.Utf8)
            )
        )
        .sort("GO_counts", descending=True)
    )
    cc, bp, mf = [], [], []
    missing_counts = {"CC": 0, "BP": 0, "MF": 0}
    for id, gos in zip(id_go["ProteinId"], id_go["accession"]):
        parents, missing = get_parents(
            id, gos, rmap, id2term, id2domain, priority=priority
        )
        cc.append(parents["CC"])
        bp.append(parents["BP"])
        mf.append(parents["MF"])
        for k, v in missing.items():
            missing_counts[k] += v
    added = has_go.with_columns(
        GO_category_CC=pl.Series(cc),
        GO_category_MF=pl.Series(mf),
        GO_category_BP=pl.Series(bp),
    )
    no_go = no_go.with_columns(
        GO_category_CC=pl.lit("unknown"),
        GO_category_MF=pl.lit("unknown"),
        GO_category_BP=pl.lit("unknown"),
    )
    result = pl.concat([added, no_go], how="vertical")
    missing_df = (
        pl.DataFrame(missing_counts)
        .melt()
        .rename({"variable": "domain", "value": "n_missing"})
    )
    if not result.shape[0] == shape_before[0]:
        raise ValueError(
            f"""
                        nrows before and after do not match!\n
                        rows before: {shape_before[0]}\n
                        rows current: {result.shape[0]}\n
                         """
        )
    return result.to_pandas(), missing_df.to_pandas()


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
