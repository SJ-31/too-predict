#!/usr/bin/env ipython

from functools import reduce
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import too_predict._train_utils as tt
import too_predict.evaluation as te
import too_predict.utils as ut
from pyhere import here

# #  --- CODE BLOCK ---

PROJECTS_TO_CHECK = [
    "GSE235548",
    "GSE185335",
    "GSE276387",
    "GSE212014",
    "GSE278302",
    "GSE233532",
    "GSE198697",
    "GSE277147",
    "GSE280749",
    "GSE218385",
    "GSESE247359",
    "GSE218114",
    "GSE214295",
    "GSE230383",
    "GSE233468",
    "GSE243649",
    "GSE247380",
    "GSE249670",
    "GSE253558",
    "GSE261012",
]

PREFIXES = ["TARGET", "CPTAC", "CGCI"]


def fitness(model, adata) -> float:
    print(f"Computing fitness for adata of shape {adata.shape}...")
    split_fn = tt.ADDITIONAL_SPLITS["CHULA"]
    accs = []
    for _ in range(5):  # avg for stability
        results = te.holdout(
            model,
            adata=adata,
            split_fns={"CHULA": split_fn},
            transformer=None,  # No need
            # because CLR is sample-independent
        )
        accs.append(results["misc"].iloc[0]["balanced_acc"])
    return float(np.mean(accs))


def simple_selection(
    model, adata, mode: Literal["case", "prefix"] = "case"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple sequential selection to choose subset of samples to exclude from the data
    Solution is a boolean mask on the additional subset
    TODO: removing individual samples might not have much of an effect. Try including projects as well
    TODO: could turn this into a more complex GA that tests different combinations of
    samples
    """
    if mode == "case":
        to_verify = adata[adata.obs["Project_ID"].isin(PROJECTS_TO_CHECK), :]
        kept = adata[~adata.obs["Project_ID"].isin(PROJECTS_TO_CHECK), :]
    else:
        prefix_mask = reduce(
            lambda x, y: x | y,
            [adata.obs["Project_ID"].str.startswith(p) for p in PREFIXES],
        )
        kept = adata[~prefix_mask, :]
        to_verify = adata[~prefix_mask, :]
    solution = np.array([False] * to_verify.shape[0])
    result_tracker: dict = {"score": [], "iter": [], "keep": []}

    prev_fitness = fitness(model, kept)
    result_tracker["score"].append(prev_fitness)
    result_tracker["iter"].append(0)
    result_tracker["keep"].append(True)

    def updater(value, soln) -> np.ndarray:
        try_soln = soln.copy()
        if mode == "case":
            try_soln[i] = True
        elif mode == "prefix":
            indices = to_verify.obs["Project_ID"].str.startswith(value)
            try_soln[indices] = True

        masked = to_verify[try_soln, :]
        cur_fitness: float = fitness(
            model,
            ad.concat([kept, masked], join="inner", axis="obs", merge="first"),
        )
        result_tracker["score"].append(cur_fitness)
        result_tracker["iter"].append(i + 1)
        if did_keep := cur_fitness > prev_fitness:
            return try_soln
        result_tracker["keep"].append(did_keep)
        return soln

    if mode == "case":
        for i in range(kept.shape[0]):  # Verify each sample individually
            print(f"Current samples added: {solution.sum()}")
            solution = updater(i, solution)
    elif mode == "prefix":
        for p in PREFIXES:  # Verify groups of samples for large projects where
            # the above is infeasible
            print(f"Current samples added: {solution.sum()}")
            solution = updater(p, solution)

    return kept.obs[solution, :], pd.DataFrame(result_tracker)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    args = vars(parser.parse_args())  # convert to dict
    return args


if __name__ == "__main__":
    args = parse_args()
    chosen = tt.MODELS["clr_xgb1_edger"]
    filter, model, transform, _, _, _ = tt.read_model_spec(chosen)
    if str(Path.home()) != "/home/shannc":
        adata = ut.training_data_internal()
    else:
        adata = ut.training_data_internal_test()
    outdir: Path = here("data", "output", "dataset_curation")
    adata = filter.fit_transform(adata)
    adata = transform.fit_transform(adata)
    case_df, case_tracker = simple_selection(model, adata, "case")
    # Do case-level selection on GSE, not too many
    case_df.reset_index().to_csv(
        outdir.joinpath("case_level_selection.csv"), index=False
    )
    case_tracker.to_csv(outdir.joinpath("case_selection_scores.csv"), index=False)
    proj_df, proj_tracker = simple_selection(model, adata, "prefix")
    proj_df.reset_index().to_csv(
        outdir.joinpath("project_level_selection.csv"), index=False
    )
    proj_tracker.to_csv(outdir.joinpath("case_tracker.csv"), index=False)
