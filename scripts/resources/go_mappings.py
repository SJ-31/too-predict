#!/usr/bin/env ipython

# Get map of go terms to parent terms at a specific level
import pandas as pd
import too_predict.go_utils as gt
import too_predict.utils as ut

go_terms = list(pd.read_csv(ut.get_data("go_meta_2025-4-8.csv"))["accession"])
sg = gt.SubsetGO(subset=go_terms)

top_level = 3
tmp_dict = {
    g: sg.successors[g]
    for g in sg.metadata[sg.metadata["level"] == top_level]["accession"]
}
top_df: pd.DataFrame = (
    pd.DataFrame({"top_level": tmp_dict.keys(), "accession": tmp_dict.values()})
    .explode("accession")
    .merge(
        sg.metadata.loc[:, ["term"]],
        how="inner",
        left_on="top_level",
        right_index=True,
    )
).rename({"term": "top_level_name"}, axis=1)

top_df.to_csv(
    ut.get_data(f"mappings/go_term2to_level_{top_level}.csv", must_exist=False),
    index=False,
)
