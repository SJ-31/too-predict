#!/usr/bin/env ipython
import json
import sys

import pandas as pd
import polars as pl

from too_predict.go_utils import SubsetGO


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


def into_domain(gos: list, mapping: dict) -> dict:
    partitioned: dict = {"CC": [], "BP": [], "MF": []}
    for go in gos:
        lookup: str = mapping[go]
        if lookup != "NA":  # Check if term is obsolete
            partitioned[lookup].append(go)
    return partitioned


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


# Is a method of SubsetGo
def ss_get_parents(self, domain: str, n: int = 18, show=True, min_depth=2, pre=""):
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
        if self._check_parent(go_id, parents):
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


# Different backends for range getter
# # TODO: haven't implemented id2labels for the others
# def _get_ranges_nx(
#     self,
#     id: str,
#     vals: np.ndarray,
#     labels: pd.Series,
#     use_unique: bool = True,
#     n_bins: int = 30,
#     report_n: int = 3,
#     cutoff=0.5,
# ) -> tuple:
#     if use_unique:
#         nodes = np.unique(vals)
#     else:
#         nodes = np.linspace(start=min(vals), stop=max(vals), num=n_bins)
#     expr = pd.Series(vals, index=labels)
#     nodes = sorted(nodes)
#     G: nx.Graph = nx.Graph()
#     for pair in itertools.combinations(nodes, 2):
#         begin = min(pair)
#         end = pair[0] if begin == pair[1] else pair[1]
#         narrowed = expr[(begin <= expr) & (expr <= end)]
#         counts = narrowed.index.value_counts()
#         gini = self.gini_impurity(
#             counts=counts, size=len(narrowed), report_n=report_n
#         )
#         if gini < cutoff:
#             G.add_edge(begin, end, within=counts)
#     ranges = []
#     range2contents = {}
#     for cmp in nx.connected_components(G):
#         s = G.subgraph(cmp)
#         rge = (min(s.nodes), max(s.nodes))
#         cur_counts = reduce(
#             lambda x, y: x if all(x.values >= y.values) else y,
#             nx.get_edge_attributes(s, "within").values(),
#         ).sort_values(ascending=False)
#         top_count, top_label = cur_counts[0], cur_counts.index[0]
#         if self._check_label_p(top_label, top_count):
#             ranges.append(rge)
#             range2contents[rge] = cur_counts
#             for lab in cur_counts.index[:report_n]:
#                 self.label_tracker[lab] = self.label_tracker.get(lab, 0) + 1
#     return ranges, range2contents

# def _get_ranges_it(
#     self,
#     id: str,
#     vals: np.ndarray,
#     labels: pd.Series,
#     use_unique: bool = True,
#     n_bins: int = 30,
#     report_n: int = 3,
#     cutoff=0.5,
# ) -> tuple:
#     if use_unique:
#         nodes = np.unique(vals)
#     else:
#         nodes = np.linspace(start=min(vals), stop=max(vals), num=n_bins)
#     expr = pd.Series(vals, index=labels)
#     It: IntervalTree = IntervalTree()
#     for pair in itertools.combinations(nodes, 2):
#         begin = min(pair)
#         end = pair[0] if begin == pair[1] else pair[1]
#         narrowed = expr[(begin <= expr) & (expr <= end)]
#         counts = narrowed.index.value_counts()
#         gini = self.gini_impurity(
#             counts=counts, size=len(narrowed), report_n=report_n
#         )
#         if gini < cutoff:
#             It.add(Interval(begin, end, data=counts))
#     ranges = []
#     range2contents = {}
#     It.merge_overlaps(
#         data_reducer=lambda x, y: x if all(x.values >= y.values) else y
#     )
#     for it in It.items():
#         rge = (it.begin, it.end)
#         sorted = it.data.sort_values(ascending=False)
#         top_count, top_label = sorted[0], sorted.index[0]
#         if self._check_label_p(top_label, top_count):
#             ranges.append(rge)
#             range2contents[rge] = it.data
#             for lab in it.data.sort_values().index[:report_n]:
#                 self.label_tracker[lab] = self.label_tracker.get(lab, 0) + 1
#     return ranges, range2contents
