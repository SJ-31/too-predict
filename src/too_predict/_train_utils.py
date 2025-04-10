#!/usr/bin/env ipython
from typing import Callable

from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import too_predict.model as tm
from too_predict.filter import Filter
from too_predict.go_utils import RecodeGO
from too_predict.imbalance import Balancer
from too_predict.imputer import Imputer
from too_predict.model import PredBase, RandomForestClassifier, XGBEstimator
from too_predict.transformer import Transformer
from too_predict.utils import (
    RNG,
    get_blacklist_internal,
    ref_feature_lists_internal,
)

REF_LISTS, FEATURE_LISTS = ref_feature_lists_internal()

# * Models
# Key:
# m : model
# i : imputation
# t : transformation
# f : feature set
# r : reference set
# e : encoding (only "GO" and None) are supported [2025-04-01 Tue]
# b : balancer
# k : kwargs to transformer
# l : feature blacklist

MODELS: dict = {
    "clr_random_forest_minfo": {
        "m": tm.RandomForestPred(),
        "t": "clr",
        "i": "plus_one",
        "f": "mutual_info_feature_list_3000",
        "s": True,
    },
    "clr_xgboost_edger_low_variance_ref": {
        "m": tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "r": "variance_feature_list_lowest_20",
        "s": True,  # [2025-04-08 Tue]
    },
    "clr_xgboost_edger_GO": {
        "m": tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_go_feature_list_40",
        "e": "GO",
        "s": True,
    },
    "clr_xgboost_variance_GO": {
        "m": tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "e": "GO",
        "f": "variance_go_feature_list_1500",
        "s": True,
    },
    "clr_xgboost_edger_smote": {
        "m": tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "b": Balancer("SMOTE"),
        "s": True,
    },
    "qsmooth_xgboost_edger_1000": {
        "m": tm.PredBase(model=tm.XGBEstimator()),
        "t": "qsmooth",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "s": True,
    },
    "clr_lr_edger_3000": {
        "m": tm.PredBase(model=LogisticRegression(solver="saga")),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,  # [2025-04-08 Tue]
    },
    "clr_xgboost_edger_1000_undersample": {
        "m": tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "b": Balancer(method="RandomUnderSampler", sampling_strategy="not minority"),
        "s": True,  # [2025-04-08 Tue]
    },
    "clr_xgb3_edger": {
        "m": tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,  # [2025-04-08 Tue] Surprisingly good
    },
    "clr_xgboost_edger_1000_organoid_edger_blacklist": {
        "m": tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "l": "edgeR_median_lfc_feature_list_3000-high_organoid_lfc.txt",
    },
    "clr_xgboost_edger": {
        "m": tm.PredBase(model=tm.XGBEstimator(), make_dense=True),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
    },
    "batch_xgboost_lg_edger": {  # [2025-03-27 Thu] TODO: choose a fast m for `outer`
        "m": tm.BatchBase(
            inner=tm.XGBEstimator(max_depth=3),
            outer_y="Sample_Type",
            categorical_support=False,
            outer=tm.PredBase(RandomForestClassifier()),
        ),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
    },
    "clr_ebm_edger": {
        "m": tm.PredBase(model=ExplainableBoostingClassifier(n_jobs=-2)),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,  # [2025-03-27 Thu] Want to try this out badly, but it's so slow
    },
    "clr_dt_edger": {  # A surrogate model
        "model": tm.PredBase(model=DecisionTreeClassifier()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "s": True,
    },
    "clr_random_forest_edger": {
        "m": tm.PredBase(model=RandomForestClassifier()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,
    },
    "alr_xgboost_low_variance": {
        "m": tm.AlrBase(
            tm.XGBEstimator(),
            references=REF_LISTS["variance_feature_list_lowest_20"],
        ),
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "r": "variance_feature_list_lowest_20",
        "s": True,
    },
    "alr_xgboost_low_variance_1000": {
        "m": tm.AlrBase(
            tm.XGBEstimator(),
            references=REF_LISTS["variance_feature_list_lowest_20"],
        ),
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "r": "variance_feature_list_lowest_20",
        "s": True,
    },
    "alr_xgboost_edger_lowest_1000": {
        "m": tm.AlrBase(
            tm.XGBEstimator(),
            references=REF_LISTS["edgeR_median_lfc_feature_list_lowest_20"],
        ),
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "r": "edgeR_median_lfc_feature_list_lowest_20",
    },
    "alr_xgboost_edger_lowest_1000_only_5": {
        "m": tm.AlrBase(
            tm.XGBEstimator(),
            references=REF_LISTS["edgeR_median_lfc_feature_list_lowest_20"],
            n_refs=5,
        ),
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "r": "edgeR_median_lfc_feature_list_lowest_20",
    },
    "clr_xgboost_go_level_4_sum": {
        "m": PredBase(XGBEstimator(max_depth=3)),
        "i": "plus_one",
        "e": RecodeGO(id_col="GENEID", level=4),
        "s": True,  # [2025-04-09 Wed]
    },
    "clr_xgboost_go_level_3_sum": {
        "m": PredBase(XGBEstimator(max_depth=3)),
        "i": "plus_one",
        "e": RecodeGO(id_col="GENEID", level=3),
        "s": True,  #  [2025-04-09 Wed]
    },
    "clr_xgboost_go_level_2_sum": {
        "m": PredBase(XGBEstimator(max_depth=3)),
        "i": "plus_one",
        "e": RecodeGO(id_col="GENEID", level=2),
    },
    "alr_random_forest_low_variance": {
        "m": tm.AlrBase(
            RandomForestClassifier(random_state=RNG),
            references=REF_LISTS["variance_feature_list_lowest_20"],
        ),
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "r": "variance_feature_list_lowest_20",
        "s": True,
    },
    "alr_random_forest_edger_lfc": {
        "m": tm.AlrBase(
            RandomForestClassifier(random_state=RNG),
            references=REF_LISTS["edgeR_median_lfc_feature_list_lowest_20"],
        ),
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "r": "edgeR_median_lfc_feature_list_lowest_20",
        "s": True,
    },
    "tmm_random_forest_edger": {
        "m": tm.RandomForestPred(),
        "t": "tmm",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,
    },
    "tpm_random_forest_edger": {
        "m": tm.RandomForestPred(),
        "t": "tpm",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,
    },
    "fpkm_random_forest_edger": {
        "m": tm.RandomForestPred(),
        "t": "fpkm",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,
    },
    "dirichlet_random_forest_edger": {
        "m": tm.SimPred(RandomForestClassifier(random_state=RNG), method="dirichlet"),
        "t": "none",
        "i": "none",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,
    },
    "clr_xgb3_pulp_lfc": {
        "m": tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "i": "plus_one",
        "f": "pulp_scanpy_minimized_lfc_ratio",
    },
    "clr_xgb3_pulp_euclidean": {
        "m": tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "i": "plus_one",
        "f": "pulp_euclidean_edgeR_3000_subset",
    },
}

# * Additional splits
# [2025-03-11 Tue]
# Want to see how well models handle Chula organoids and datasets from other projects
# Ideally this should be done with StratifiedGroupKFold but
# grouping is problematic because some groups are confounded
# with whatever you are labeling on
# This means that some instances won't be seen at all in the test data
# gc is the group variable excluded during the cv folds e.g. Sample_Type
ADDITIONAL_SPLITS: dict = {
    "CHULA": lambda x: (
        x[~x.obs["Project_ID"].str.contains("CHULA"), :],
        x[x.obs["Project_ID"].str.contains("CHULA"), :],
    ),
    "CGCI": lambda x: (
        x[~x.obs["Project_ID"].str.contains("CGCI"), :],
        x[x.obs["Project_ID"].str.contains("CGCI"), :],
    ),
    "CPTAC": lambda x: (
        x[~x.obs["Project_ID"].str.contains("CPTAC"), :],
        x[x.obs["Project_ID"].str.contains("CPTAC"), :],
    ),
    "GEO": lambda x: (
        x[~x.obs["Project_ID"].str.contains("GSE"), :],
        x[x.obs["Project_ID"].str.contains("GSE"), :],
    ),
}


# * Helper functions
def read_model_spec(
    spec: dict,
) -> tuple[Filter | None, PredBase, Transformer, Balancer | None, RecodeGO | None]:
    M: PredBase = spec.get("m")
    references = spec.get("r")
    features = spec.get("f")
    encoding = spec.get("e")
    blacklist = spec.get("l")

    if encoding is not None:
        fcol = "accession"
    else:
        fcol = "GENEID"
    b_list = None
    if blacklist is not None:
        b_list = get_blacklist_internal(blacklist)
    f_list = FEATURE_LISTS.get(features)
    r_list = REF_LISTS.get(references)
    if f_list:
        F = Filter(
            f_list if references is None else f_list + r_list,
            feature_col=fcol,
            blacklist=b_list,
        )
    else:
        F = None
    B: Balancer = spec.get("b")
    transformation_name = spec.get("t")
    kwargs: dict = spec.get("k", {})
    if transformation_name == "clr" and references is not None:
        kwargs.update({"features": r_list, "feature_col": "GENEID"})
    T = Transformer(
        transformation_name, impute_fn=Imputer(spec.get("i")), inplace=False, **kwargs
    )
    return F, M, T, B, encoding
