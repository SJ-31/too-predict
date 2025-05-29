#!/usr/bin/env ipython

from collections.abc import Iterable
from pathlib import Path

import anndata as ad
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp as pu
import scanpy as sc
import scipy.optimize as opt
import seaborn as sns
import sklearn.feature_selection as fs
import too_predict._rust_helpers as rh
import too_predict.evaluation as te
import too_predict.go_utils as gu
import too_predict.utils as ut
from joblib import Parallel, delayed
from pyhere import here
from scipy import sparse
from scipy.stats import mode
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from too_predict._train_utils import ADDITIONAL_SPLITS, MODELS, read_model_spec
from too_predict.evaluation import sklearn_cv_coefs, write_cross_val
from too_predict.filter import Filter, get_redundant_features
from too_predict.imputer import Imputer
from too_predict.model import PredBase, RandomForestPred, XGBEstimator
from too_predict.range_finder import RangeFinder
from too_predict.transformer import Transformer
from too_predict.utils import (
    ref_feature_lists_internal,
    training_data_internal,
    training_data_internal_test,
)

OUTDIR = here("data", "output", "feature_selection")
adata: ad.AnnData

RECODE_GO: bool = False


# * Identify stable features for ALR

# ** Variance-based


def variance_threshold(f):
    vt = fs.VarianceThreshold(threshold=4)
    _ = vt.fit_transform(n_counts)
    variance = adata.var.loc[~vt.get_support(), :]
    variance.loc[:, "variance"] = np.nanvar(n_counts, axis=0)[~vt.get_support()]
    variance = variance.sort_values("variance")
    sns.histplot(variance["variance"])
    if RECODE_GO:
        name = "sklearn_variance_go.png"
        variance = variance.reset_index()
    else:
        name = "sklearn_variance.png"
    plt.savefig(here(OUTDIR, name))
    variance.to_csv(f, index=False)


# ** Highest proportionality
def get_proportionality(f):
    pairwise = rh.rho_matrix(n_counts, True)
    pair_df = pd.DataFrame(
        pairwise, columns=adata.var["GENEID"], index=adata.var["GENEID"]
    )
    central_prop = pair_df.median().sort_values(ascending=False)
    flattened_pairs = [
        (f"{i} - {j}", pair_df.loc[i, j])
        for i in pair_df.columns
        for j in pair_df.index
        if i != j
    ]
    best_pairs = sorted(flattened_pairs, key=lambda x: x[1], reverse=True)
    pair_df.to_csv(f, index=False)


# * Identify useful features


# <2025-02-21 Fri> but why identify features with a different model than the one
# you are going to eventually train on?
def sfm(f):
    forest = RandomForestPred("clr", "plus_one")
    forest.fit(adata, y="primary_site")
    model = fs.SelectFromModel(forest, prefit=True)
    selected = model.get_support()
    kept_features = forest.var.loc[selected, :]
    kept_features.loc[:, "importance"] = forest.feature_importances_[selected]
    kept_features.to_csv()


# ** Information
# Try out information gain method
# gene expression data is continuous though
def entropy(y, base=2) -> float:
    p = pd.Series(y).value_counts() / len(y)
    entropy = -np.sum(p * np.emath.logn(base, p))
    return entropy


def mutual_info(f):
    info = fs.mutual_info_classif(n_counts, labels)
    info_df = pd.DataFrame({"feature": adata.var["GENEID"], "mutual_info": info})
    info_df.sort_values("mutual_info", ascending=False, inplace=True)
    info_df.to_csv(f, index=False)
    # The higher the better here


# ** Checking feature importances
# Will do this with Rfs and the xgbestimator
def tree_importance(adata, classifier: PredBase, outdir):
    _, feat = ref_feature_lists_internal()
    first_filter = Filter(
        feature_col="GENEID",
        features=feat["edgeR_median_lfc_feature_list_3000"],
        inplace=False,
    )
    transformer = Transformer("clr", impute_fn=Imputer("plus_one"), inplace=False)
    x: ad.AnnData = first_filter.fit_transform(adata)
    x = transformer.fit_transform(x)
    classifier.fit(x, y="tumor_type")
    x.var["importance"] = classifier.model.feature_importances_
    cv_results_1 = cross_validate(classifier, x, label_col="tumor_type")
    write_cross_val(cv_results_1, outdir, "cv_before", "_cv_before")

    nonzero = x.var.loc[x.var["importance"] != 0, :]

    nonzero_features = list(nonzero["GENEID"])
    nonzero.to_csv(here(outdir, "nonzero_features.csv"))

    sec_filter = Filter(feature_col="GENEID", features=nonzero_features)
    x2: ad.AnnData = sec_filter.fit_transform(adata)
    x2 = transformer.fit_transform(x2)
    cv_results_2 = cross_validate(classifier, x2, label_col="tumor_type")
    write_cross_val(cv_results_2, outdir, "cv_after", "_cv_after")


# ** Filtering organoid-vs-primary DE genes
#
def ovp_filter(adata):
    ovp_top_tags = pd.read_csv(
        here(
            "data",
            "output",
            "chula_organoid_comparison",
            "de_enrichment",
            "sample_type_top_tags.tsv",
        ),
        sep="\t",
    )
    chosen_model = "clr_xgb3_1000_edger"
    F, model_fn, T, B, R, C = read_model_spec(MODELS[chosen_model])
    M = model_fn()
    blacklist = ovp_top_tags.query("PValue <= 0.05")["GENEID"].to_list()
    filtered = adata[:, ~adata.var["GENEID"].isin(blacklist)]
    filtered = T.fit_transform(filtered)
    outdir = here(OUTDIR, "ovp_filter")
    outdir.mkdir(exist_ok=True)

    # With rfecv
    rfecv = fs.RFECV(estimator=M, step=1, cv=StratifiedKFold(5))
    counts = filtered.X.toarray()
    labels = filtered.obs["tumor_type"]
    rfecv.fit(counts, labels)
    x_train, x_test, y_train, y_test = train_test_split(counts, labels)
    cv_score = cross_val_score(M, x_train, y_train)
    print(cv_score)

    df = pd.DataFrame({"GENEID": filtered.var["GENEID"], "ranking": rfecv.ranking_})
    df.to_csv(outdir.joinpath("rfecv_importances.csv"), index=False)
    score_df = pd.DataFrame(rfecv.cv_results_)
    score_df.to_csv(outdir.joinpath("rfecv_cv_results.csv"), index=False)

    # Linear model
    scaler = StandardScaler()  # Required by L2 penalty
    counts = filtered.X.toarray()
    counts = scaler.fit_transform(counts)

    lm = LogisticRegressionCV(
        solver="saga",
        penalty="elasticnet",
        cv=5,
        l1_ratios=[0, 0.5, 1],  # Try Ridge, balanced, Lasso
    )

    cross_val = cross_validate(lm, counts, labels)
    avg = cross_val["test_score"].mean()
    print(f"logistic regression {avg=} acc")
    print(cross_val["test_score"])

    coef_stdev = te.agg_lr_coefs(cross_val, filtered.var["GENEID"]).reset_index()

    first = cross_val["estimator"][0]
    coeffs = pd.DataFrame(
        np.transpose(first.coefs_),
        columns=first.classes_,
        index=filtered.var["GENEID"],
    ).reset_index()
    coeffs.to_csv(outdir.joinpath("lm_coefs_final.csv"), index=False)
    coef_stdev.to_csv(outdir.joinpath("lm_coef_std.csv"), index=False)


# ** Removing redundant features


def remove_redundant(adata):
    _, features = ref_feature_lists_internal()
    original = features["edgeR_median_lfc_feature_list_3000"]
    r_outdir: Path = OUTDIR.joinpath("redundant_features")
    r_outdir.mkdir(exist_ok=True, parents=True)
    hrange = [0.4, 0.6, 0.8]
    method_spec = ["correlation", "rho_prop"]
    filter = Filter(original, feature_col="GENEID")
    adata = filter.fit_transform(adata)
    clr = Transformer("clr", Imputer("plus_one"), inplace=False)
    model = PredBase(XGBEstimator())
    for method in method_spec:
        cur_outdir = r_outdir.joinpath(method)
        cur_outdir.mkdir(exist_ok=True, parents=True)
        if method == "correlation":
            tmp = clr.fit_transform(adata)
        else:
            tmp = adata.copy()
        for height in hrange:
            kept, removed, var = get_redundant_features(tmp, height, method)
            if len(kept) == len(adata.var.index):
                print(f"WARNING: {method} at height {height} did not cluster!")
                continue
            new_filter = Filter(kept, feature_col="GENEID")
            tmp = new_filter.fit_transform(tmp)
            results = model.cross_validate(tmp, n_splits=3, record_dir=cur_outdir)
            write_cross_val(results, cur_outdir, prefix=f"height_{height}")


# ** Testing range finder
def range_finder(adata):
    trans = Transformer(method="clr", impute_fn=Imputer("plus_one"), inplace=False)
    transformed = trans.fit_transform(adata)
    rfinder = RangeFinder()
    rfinder.fit(transformed)
    outdir = OUTDIR.joinpath("range_finder")
    outdir.mkdir(exist_ok=True)
    ut.write_pickle(rfinder, outdir.joinpath("range_finder.pkl"))
    rfinder.label_metrics.to_csv(outdir.joinpath("rf_label_metrics.csv"), index=False)
    rfinder.id_metrics.to_csv(outdir.joinpath("rf_id_metrics.csv"), index=False)


# ** With optimization

# *** Fns


def milp_optimize(batch_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
    """Use scipy's milp function to minimize the ratio

    batch_vals, y_vals are a 2d array of shape n_labels x n_features
    """
    c = np.nanmean(batch_vals, axis=0) / np.nanmean(y_vals, axis=0)
    c[np.isnan(c)] = 0
    c[np.isinf(c)] = 0

    solve = opt.milp(c, integrality=1, bounds=opt.Bounds(lb=0, ub=1))
    return solve.x


def dist_optimize(
    x: ad.AnnData,
    y: ad.AnnData,
    rng: np.random.Generator | None = None,
    n: int = 1000,
    n_iter: int = 1000,
    parallel: bool = False,
    n_jobs: int = 8,
) -> list[str]:
    """
    Find the optimal set of n features that
    minimizes the euclidean distance between the
    samples in x and those in y with random sampling
    """
    if rng is None:
        rng = np.random.default_rng()
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same number of features!")
    x_x: np.ndarray = x.X.toarray() if sparse.issparse(x.X) else x.X
    y_x: np.ndarray = y.X.toarray() if sparse.issparse(y.X) else y.X

    def helper() -> list:
        x_draw, y_draw = rng.integers(0, x.shape[0]), rng.integers(0, y.shape[0])
        x_vals, y_vals = x_x[x_draw, :], y_x[y_draw, :]
        comps = np.abs(x_vals - y_vals) ** 2
        prob = pu.LpProblem("distance", pu.LpMinimize)
        fvars = [pu.LpVariable(str(g), cat="Binary") for g in range(len(comps))]
        prob += pu.lpSum([f * comps[i] for i, f in enumerate(fvars)])
        # No square root, but effect is the same
        prob += pu.lpSum(fvars) == n
        prob.solve()
        result = [
            (int(v.name), v.varValue) for v in prob.variables() if "_" not in v.name
        ]
        result = sorted(result, key=lambda x: x[0])
        return [r[1] for r in result]

    if not parallel:
        iterations = np.array([helper() for _ in range(n_iter)])
    else:
        iterations = np.array(
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(helper)() for _ in range(n_iter)
            )
        )
    vote = mode(iterations, axis=0)
    kept = [f for i, f in enumerate(x.var.index) if vote.mode[i] == 1]
    return kept


def lfc_optimize(
    features: pd.Series | pd.Index,
    batch_vals: np.ndarray,
    y_vals: np.ndarray,
    n: int = 1000,
) -> list[str]:
    prob = pu.LpProblem("LFC", pu.LpMinimize)
    c = np.nanmean(batch_vals, axis=0) / np.nanmean(y_vals, axis=0)
    to_discard = np.isnan(c) | np.isinf(c)
    if to_discard.sum() > 0:
        print(f"lfc_optimize: {to_discard.sum()} inf or nan features!")
    features = features[~to_discard]
    c = c[~to_discard]
    fvars = [pu.LpVariable(g, cat="Binary") for g in features]

    prob += pu.lpSum([f * c[i] for i, f in enumerate(fvars)])
    prob += pu.lpSum(fvars) == n
    prob.solve()
    print(pu.LpStatus[prob.status])
    return [v.name for v in prob.variables() if v.varValue == 1]


# *** Use functions


def optimization_scanpy(adata):
    spc = MODELS["clr_random_forest_edger"]

    F, M, T, B, E = read_model_spec(spc)

    transformed: ad.AnnData = T.fit_transform(adata)
    split_fn = ADDITIONAL_SPLITS["CHULA"]
    sc.tl.rank_genes_groups(transformed, groupby="Sample_Type")
    ri = ut.RankInterpreter(transformed)
    batch_df = ri.feature_stat("logfoldchanges")
    sc.tl.rank_genes_groups(transformed, groupby="tumor_type")
    ri = ut.RankInterpreter(transformed)
    y_df = ri.feature_stat("logfoldchanges")

    batch_df, y_df = batch_df.align(y_df, join="inner", axis=0)
    batch_vals, y_vals = np.transpose(batch_df.values), np.transpose(y_df.values)
    solution = lfc_optimize(transformed.var.index, batch_vals, y_vals, n=3000)
    OUTDIR.joinpath("feature_lists").joinpath(
        "pulp_scanpy_minimized_lfc_ratio.txt"
    ).write_text("\n".join(solution))

    filtered = F.fit_transform(adata)
    transformed2 = T.fit_transform(filtered)
    train, test = split_fn(transformed2)

    dist_solution = dist_optimize(train, test, n=1500, n_iter=3000, parallel=True)
    dist_file: Path = OUTDIR.joinpath("feature_lists").joinpath(
        "pulp_euclidean_edgeR_3000_subset.txt"
    )
    dist_file.write_text("\n".join(dist_solution))


# * Run


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--memory", default="30")
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-c", "--cores", default=8)
    parser.add_argument(
        "-g", "--recode_go", default=False, help="", action="store_true"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        print("Using test subset")
        adata = training_data_internal_test()
        OUTDIR = OUTDIR.joinpath("test")
        OUTDIR.mkdir(exist_ok=True, parents=True)
    else:
        adata = training_data_internal()

    RECODE_GO = args.recode_go
    suffix = ""
    if RECODE_GO:
        rc = gu.RecodeGO(level=4)
        print("Recoding adata to GO...")
        adata = rc.fit_transform(adata)
        suffix = "GO"

    normalized = Transformer("clr", Imputer("plus_one"), inplace=False).fit_transform(
        adata
    )
    n_counts = normalized.X.toarray()
    labels = adata.obs["primary_site"]
    # Need to normalize first to move out of simplex

    variance_file = here(OUTDIR, f"sklearn_variance_{suffix}.csv")
    proportionality_file = here(OUTDIR, f"proportionality_matrix_{suffix}.csv")
    mutual_info_file = here(OUTDIR, f"mutual_info_{suffix}.csv")

    with joblib.parallel_backend("loky", n_jobs=args.cores):
        # remove_redundant(adata)
        # variance = read_existing(variance_file, variance_threshold, pd.read_csv)
        # mutual_info = read_existing(mutual_info_file, mutual_info, pd.read_csv)
        # prop = read_existing(
        #     proportionality_file, get_proportionality, pd.read_csv
        # )  # <2025-02-28 Fri> This fails, too much data?
        # tree_importance(
        #     adata, PredBase(model=XGBEstimator(importance_type="gain")), outdir
        # )
        # importance score with gain are the average gain across all trees
        # optimization_scanpy(adata)
        # ovp_filter(adata)
        range_finder(adata)
