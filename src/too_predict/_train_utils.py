#!/usr/bin/env ipython
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from interpret.glassbox import ExplainableBoostingClassifier
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier

import too_predict.deep.logistic as d_log
import too_predict.evaluation as te
import too_predict.model as tm
import too_predict.recoder as rt
import too_predict.transformer as trf
from too_predict.corrector import Corrector
from too_predict.deep.callbacks import AverageBest
from too_predict.deep.nns import Disyak
from too_predict.filter import Filter
from too_predict.imbalance import Balancer
from too_predict.imputer import Imputer
from too_predict.model import Pipeline, PredBase, RandomForestClassifier, XGBEstimator
from too_predict.range_finder import RangeFinderPred
from too_predict.transformer import Transformer
from too_predict.utils import (
    RANDOM_STATE,
    cell_markers_internal,
    get_blacklist_internal,
    ref_feature_lists_internal,
)

REF_LISTS, FEATURE_LISTS = ref_feature_lists_internal()

marker_file = cell_markers_internal(file_only=True)
marker_meta = cell_markers_internal(meta=True)


def get_common():
    m: dict[str, list] = cell_markers_internal()
    return {k: v for k, v in m.items() if k.startswith("common-")}


# * Models
# Key:
# m : model
# i : imputation
# t : transformation
# f : feature set
# r : reference set
# e : encoding (only "GO" and None) are supported [2025-04-01 Tue]
# b : balancer
# c : correction to counts
# k : kwargs to transformer
# l : feature blacklist
# p : post_processing for transformer

# model, filter, transformation values must be functions of no arguments that return
# the object

MODELS: dict = {
    # ** Qsmooth
    "qsmooth_xgboost_edger_1000": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "qsmooth",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "s": True,
    },
    # ** CLR models
    "clr_xgb3_pulp_lfc": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "i": "plus_one",
        "f": "pulp_scanpy_minimized_lfc_ratio",
        "s": True,
    },
    "clr_xgb3_pulp_euclidean": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "i": "plus_one",
        "f": "pulp_euclidean_edgeR_3000_subset",
        "s": True,
    },
    "clr_random_forest_minfo": {
        "m": lambda: tm.RandomForestPred(),
        "t": "clr",
        "i": "plus_one",
        "f": "mutual_info_feature_list_3000",
        "s": True,
    },
    "clr_xgboost_edger_per_type": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_70_per_type_",
        "s": True,
    },
    "xgboost_edger_per_type": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": None,
        "i": None,
        "f": "edgeR_70_per_type",
        "s": True,
    },
    "clr_xgboost_edger_per_type_ovp": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_70_per_type_ovp_",
        "s": True,
        "w": ("additional"),
    },
    "clr_xgboost_edger_half": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "half_organoid_primary_feature_list_3000",
        "s": False,
        "w": ("additional"),
    },
    "clr_xgboost_edger_half_per_type": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_half_50_per_type",
        "s": False,
        "w": ("additional"),
    },
    "xgboost_edger_per_type_ovp": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": None,
        "i": None,
        "f": "edgeR_70_per_type_ovp_",
        "s": True,
        "w": ("additional"),
    },
    "clr_xgboost_edger_per_type_ovp_t_enriched": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_70_per_type_ovp_tissue_enriched",
        "s": True,
        "w": ("additional"),
    },
    "clr_xgboost_auroc_per_type_ovp": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "auroc_70_per_type_blacklist",
        "s": True,
        "w": ("additional"),
        # [2025-05-26 Mon] This has the good results for CGCI and CPTAC, but
        # it looks like fold changes are still superior for prediction purposes
    },
    "clr_xgboost_edger_per_type_ovp_ratio_only": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_70_per_type_ovp_ratio_only",
        "s": True,
    },
    "clr_xgboost_edger_tissue_markers": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "",
        "s": True,
    },
    "clr_xgboost_edger_low_variance_ref": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "r": "variance_feature_list_lowest_20",
        "s": True,  # [2025-04-08 Tue]
    },
    "clr_xgboost_edger_smote": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "b": Balancer("SMOTE"),
        "s": True,
    },
    "clr_lr_edger_3000": {
        "m": lambda: tm.PredBase(model=LogisticRegression(solver="saga")),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,  # [2025-04-08 Tue]
    },
    "clr_xgboost_edger_1000_undersample": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "b": Balancer(method="RandomUnderSampler", sampling_strategy="not minority"),
        "s": True,  # [2025-04-08 Tue]
    },
    "clr_xgb3_edger": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,  # [2025-04-08 Tue] Surprisingly good
    },
    "clr_xgb1_edger": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator(max_depth=1)),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,  # Good, use for fast training
    },
    "clr_xgb3_edger_rfecv": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "i": "plus_one",
        "f": "clr_xgb3_1000_edger_rfecv_feature_list",
        "s": True,  # [2025-05-07 Wed] Only 783 features, not bad
    },
    "clr_xgb3_1000_edger": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "s": True,
    },
    "clr_xgboost_edger_1000_organoid_edger_blacklist": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "l": "edgeR_median_lfc_feature_list_3000-high_organoid_lfc.txt",
        "s": True,
    },
    "clr_xgboost_edger_3000_organoid_edger_blacklist_v2": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "l": "organoid_vs_primary_lfc-1000.txt",
        "s": True,
    },
    "clr_xgboost_edger": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator(), make_dense=True),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,
    },
    "batch_xgboost_lg_edger": {  # [2025-03-27 Thu] TODO: choose a fast m for `outer`
        "m": lambda: tm.BatchBase(
            inner=tm.XGBEstimator(max_depth=3),
            outer_y="Sample_Type",
            categorical_support=False,
            outer=tm.PredBase(RandomForestClassifier()),
        ),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "s": True,
    },
    # "clr_ebm_edger": {
    #     "m": lambda: tm.PredBase(model=ExplainableBoostingClassifier(n_jobs=-2)),
    #     "t": "clr",
    #     "i": "plus_one",
    #     "f": "edgeR_median_lfc_feature_list_3000",
    #     "s": True,  # [2025-03-27 Thu] Want to try this out badly, but it's so slow
    # },
    "clr_dt_edger": {  # A surrogate model
        "m": lambda: tm.PredBase(model=DecisionTreeClassifier()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "s": True,
    },
    "clr_random_forest_edger": {
        "m": lambda: tm.PredBase(model=RandomForestClassifier()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,
    },
    # ** ALR models
    "alr_xgboost_low_variance_1000": {
        "m": lambda: tm.AlrBase(
            tm.XGBEstimator(),
            references=REF_LISTS["variance_feature_list_lowest_20"],
        ),
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "r": "variance_feature_list_lowest_20",
        "s": True,
    },
    "alr_xgboost_edger_lowest_1000": {
        "m": lambda: tm.AlrBase(
            tm.XGBEstimator(),
            references=REF_LISTS["edgeR_median_lfc_feature_list_lowest_20"],
        ),
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "r": "edgeR_median_lfc_feature_list_lowest_20",
        "s": True,  # BUG: this one has value errors for some reason
    },
    "alr_xgboost_edger_lowest_1000_only_5": {
        "m": lambda: tm.AlrBase(
            tm.XGBEstimator(),
            references=REF_LISTS["edgeR_median_lfc_feature_list_lowest_20"],
            n_refs=5,
        ),
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "r": "edgeR_median_lfc_feature_list_lowest_20",
    },
    "alr_random_forest_low_variance": {
        "m": lambda: tm.AlrBase(
            RandomForestClassifier(random_state=RANDOM_STATE),
            references=REF_LISTS["variance_feature_list_lowest_20"],
        ),
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "r": "variance_feature_list_lowest_20",
        "s": True,
    },
    "alr_random_forest_edger_lfc": {
        "m": lambda: tm.AlrBase(
            RandomForestClassifier(random_state=RANDOM_STATE),
            references=REF_LISTS["edgeR_median_lfc_feature_list_lowest_20"],
        ),
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_1000",
        "r": "edgeR_median_lfc_feature_list_lowest_20",
        "s": True,
    },
    "tmm_random_foret_edger": {
        "m": lambda: tm.RandomForestPred(),
        "t": "tmm",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,
    },
    "tpm_random_forest_edger": {
        "m": lambda: tm.RandomForestPred(),
        "t": "tpm",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,
    },
    "tpm_xgb3_per_type": {
        "m": lambda: tm.PredBase(tm.XGBEstimator()),
        "t": "tpm",
        "i": "plus_one",
        "f": "edgeR_70_per_type_",
        "s": True,
    },
    "fpkm_xgb3_per_type": {
        "m": lambda: tm.PredBase(tm.XGBEstimator()),
        "t": "fpkm",
        "i": "plus_one",
        "f": "edgeR_70_per_type_",
        "s": True,
    },
    "fpkm_random_forest_edger": {
        "m": lambda: tm.PredBase(RandomForestClassifier(random_state=RANDOM_STATE)),
        "t": "fpkm",
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,
    },
    "dirichlet_random_forest_edger": {
        "m": lambda: tm.SimPred(
            RandomForestClassifier(random_state=RANDOM_STATE), method="dirichlet"
        ),
        "t": "none",
        "i": "none",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,
    },
    # ** With correction
    "clr_xgb3_edger_pycombat_seq": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "c": {
            "method": "pycombat_seq",
            "batch": "is_organoid",
            "covar_mod": "tumor_type",
        },
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,  # [2025-04-25 Fri] Way too slow
    },
    "clr_xgb3_edger_combat_seq": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "c": {
            "method": "combat_seq",
            "batch": "is_organoid",
            "group": "tumor_type",
        },
        "i": "plus_one",
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,  # [2025-05-01 Thu] Higher accuracy but not as good for
        # organoids as combat ref
    },
    "clr_xgb3_edger_rbe": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "i": "plus_one",
        "c": {
            "method": "removeBatchEffect",
            "batch": "is_organoid",
            "group": "tumor_type",
        },
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,
    },
    "clr_xgb3_edger_deseq2": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "i": "plus_one",
        "c": {
            "method": "deseq2",
            "batch": "is_organoid",
            "group": "tumor_type",
        },
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,
    },
    "clr_xgb3_edger_combat_ref_no_group": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "i": "plus_one",
        "c": {
            "method": "combat_ref",
            "batch": "is_organoid",
        },
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,
    },
    "clr_xgb3_edger_combat_ref": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "i": "plus_one",
        "c": {
            "method": "combat_ref",
            "batch": "is_organoid",
            "group": "tumor_type",
        },
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,  # [2025-04-28 Mon] Now this works well, let's see if it'll work
        # if you don't correct beforehand
        # [2025-04-29 Tue] Okay looks like this only works if the batch correction is
        # able to use information from the organoid samples, which constitutes
        # data leakage
    },
    "clr_xgb3_edger_combat_ref_org_rbatch": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "i": "plus_one",
        "c": {
            "method": "combat_ref",
            "batch": "is_organoid",
            "group": "tumor_type",
            "reference_batch": "True",
        },
        "f": "edgeR_median_lfc_feature_list_3000",
        "s": True,
    },
    "clr_xgb3_edger_combat_ref_rfecv": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator(max_depth=3)),
        "t": "clr",
        "i": "plus_one",
        "c": {
            "method": "combat_ref",
            "batch": "is_organoid",
            "group": "tumor_type",
        },
        "f": "clr_xgb3_1000_edger_rfecv_feature_list",
        "s": True,  # [2025-05-09 Fri] Hmmn, worse than the 3000 features
        # suggests that your rfecv list is not good for the organoid task
    },
    # ** Recodings
    "clr_xgboost_edger_GO": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_go_feature_list_40",
        "e": "GO",
        "s": True,
    },
    "clr_xgboost_variance_GO": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "e": "GO",
        "f": "variance_go_feature_list_1500",
        "s": True,
    },
    "clr_xgboost_go_level_4_sum": {
        "m": lambda: PredBase(XGBEstimator(max_depth=3)),
        "i": "plus_one",
        "e": lambda: rt.Recoder("go", id_col="GENEID", level=4),
        "s": True,  # [2025-04-09 Wed]
    },
    "clr_xgboost_go_level_3_sum": {
        "m": lambda: PredBase(XGBEstimator(max_depth=3)),
        "i": "plus_one",
        "e": lambda: rt.Recoder("go", id_col="GENEID", level=3),
        "s": True,  #  [2025-04-09 Wed]
    },
    "clr_xgboost_go_level_2_sum": {
        "m": lambda: PredBase(XGBEstimator(max_depth=3)),
        "i": "plus_one",
        "e": lambda: rt.Recoder("go", id_col="GENEID", level=2),
        "s": True,
    },
    # *** Marker-only
    "clr_xgboost_plage": {
        "m": lambda: PredBase(XGBEstimator(max_depth=3)),
        "i": "plus_one",
        "e": lambda: rt.Recoder("plage", reference=marker_file, metadata=marker_meta),
        "s": True,
    },
    "clr_xgboost_gsva": {
        "m": lambda: PredBase(XGBEstimator(max_depth=3)),
        "i": "plus_one",
        "e": lambda: rt.Recoder("gsva", reference=marker_file, metadata=marker_meta),
        "s": True,  # [2025-04-29 Tue] Out of memory
    },
    "clr_xgboost_plage_common": {
        "m": lambda: PredBase(XGBEstimator(max_depth=3)),
        "i": "plus_one",
        "e": lambda: rt.Recoder("plage", reference=get_common(), metadata=marker_meta),
        "s": True,
    },
    "clr_xgboost_bisquemarker": {
        "m": lambda: PredBase(XGBEstimator(max_depth=3)),
        "i": "plus_one",
        "e": lambda: rt.Recoder("bisque_marker", markers=marker_file),
        "s": True,
    },
    "clr_xgboost_bisquemarker_common": {
        "m": lambda: PredBase(XGBEstimator(max_depth=3)),
        "i": "plus_one",
        "e": lambda: rt.Recoder("bisque_marker", markers=get_common()),
        "s": True,
    },
    # ** Less quantitative
    # *** Ranking
    "clr_ranks_mean_xgb_edger_per_type_ovp": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_70_per_type_ovp_",
        "s": True,
        "p": trf.into_ranks,
    },
    "clr_ranks-std_mean_xgb_edger_per_type_ovp": {
        "m": lambda: tm.PredBase(model=tm.XGBEstimator()),
        "t": "clr",
        "i": "plus_one",
        "f": "edgeR_70_per_type_ovp_",
        "s": True,
        "p": lambda x: trf.into_ranks(x, True),
    },
    # *** Range finder
    # [2025-06-04 Wed] Not yet
    # "clr_rf_mean_xgb_edger_per_type_ovp_t_enriched": {
    #     "m": lambda: RangeFinderPred(
    #         model=tm.PredBase(model=tm.XGBEstimator()),
    #         features_per_label=70,
    #         purity_cutoff=0.3,
    #         transformer=Transformer("clr", impute_fn=Imputer("plus_one")),
    #     ),
    #     "t": None,
    #     "i": None,
    #     "f": "edgeR_70_per_type_ovp_tissue_enriched",
    #     "s": True,
    # },
    # "clr_rf_bin_xgb_edger_per_type_ovp_t_enriched": {
    #     "m": lambda: RangeFinderPred(
    #         model=tm.PredBase(model=tm.XGBEstimator()),
    #         features_per_label=70,
    #         mask_method="binary",
    #         purity_cutoff=0.3,
    #         transformer=Transformer("clr", impute_fn=Imputer("plus_one")),
    #     ),
    #     "t": None,
    #     "i": None,
    #     "f": "edgeR_70_per_type_ovp_tissue_enriched",
    #     "s": True,
    # },
    # "clr_rf_median_xgb_edger_per_type_ovp_t_enriched": {
    #     "m": lambda: RangeFinderPred(
    #         model=tm.PredBase(model=tm.XGBEstimator()),
    #         features_per_label=70,
    #         mask_method="median",
    #         purity_cutoff=0.3,
    #         transformer=Transformer("clr", impute_fn=Imputer("plus_one")),
    #     ),
    #     "t": None,
    #     "i": None,
    #     "f": "edgeR_70_per_type_ovp_tissue_enriched",
    #     "s": True,
    # },
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
    "CHULA_NO_CPTAC": lambda x: (
        x[
            ~(
                x.obs["Project_ID"].str.contains("CHULA")
                | x.obs["Project_ID"].str.contains("CPTAC")
            )
        ],
        x[x.obs["Project_ID"].str.contains("CHULA"), :],
    ),
    "NO_ORGANOID": lambda x: (
        x[x.obs["Sample_Type"] != "organoid", :],
        x[x.obs["Sample_Type"] == "organoid", :],
    ),
    "CHULA_NO_ORGANOID": lambda x: (
        x[x.obs["Sample_Type"] != "organoid", :],
        x[x.obs["Project_ID"].str.contains("CHULA"), :],
    ),
}


# * Helper functions
#
def organoid_test_task(
    adata: ad.AnnData,
    model_spec: dict,
    outdir: Path | None = None,
    correction_mode: Literal["before_split", "on_train", "on_train_test"] = "on_train",
    organoid_col: str = "is_organoid",
    label_col: str = "tumor_type",
    with_randoms: bool = True,
    shuffle_kwargs: dict | None = None,
    prefix: str = "",
    save_split_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Test model's ability to generalize to organoid samples

    For a given tumor type `A`, which has both primary and organoid samples available,
        train the model on a train set that excludes the organoid samples of `A`, then
        test on the organoid samples of `A`
    We want the model to learn to separate tumor types in primary samples, AND to
        distinguish between organoid and primary samples such that if it finds
        an organoid sample, it alters the tumor type separation criteria

    Parameters
    ----------
    organoid_col : Boolean column that is True if a sample is an organoid sample
    label_col : Factor/string column containing the tumor types
    with_randoms : True if the test set should include some random other samples, and
        not just the organoid samples for the given tumor type
    """
    filter, model, transformer, balancer, encoder, corrector = read_model_spec(
        model_spec
    )
    if shuffle_kwargs is None:
        shuffle_kwargs = {}
    if encoder is not None:
        adata = encoder.fit_transform(adata)
    if filter is not None:
        adata = filter.fit_transform(adata)
    if correction_mode == "before_split":
        adata = corrector.fit_transform(adata)
    crosses = pd.crosstab(adata.obs[label_col], adata.obs[organoid_col])
    n: int = adata.shape[0]
    filtered = crosses.loc[crosses[True] > 0, :]
    split_masks: dict = {}
    for ttype in filtered.index.tolist():
        mask = (adata.obs[label_col] == ttype) & adata.obs[organoid_col]
        if with_randoms:
            splitter = ShuffleSplit(n_splits=1, **shuffle_kwargs)
            tmp = adata[~mask, :]
            train, test = next(splitter.split(np.zeros(tmp.shape)))
            train_mask, test_mask = np.array(
                [
                    list(map(lambda x: x in t, range(adata.shape[0])))
                    for t in [train, test]
                ]
            )
            train_mask = np.logical_xor(train_mask, mask)  # Ensure that the random
            # train indices don't include organoids we want to test
            split_masks[f"{ttype}_excluded"] = (
                np.logical_or(train_mask.copy(), (~mask).copy()),
                np.logical_or(
                    mask.copy(), test_mask
                ),  # Test set contains organoids and
                # the random splits
            )
        else:
            split_masks[f"{ttype}_excluded"] = ((~mask).copy(), mask.copy())
        p_train = (~mask).sum() / n
        p_test = mask.sum() / n
        split_prop = (p_train, p_test, p_train + p_test)
        print(f"{ttype} {split_prop=}")
    result: dict = te.holdout(
        model,
        adata,
        split_masks=split_masks,
        label_col=label_col,
        transformer=transformer,
        balancer=balancer,
        corrector=corrector
        if correction_mode in {"on_train", "on_train_test"}
        else None,
        apply_correction_to="train"
        if correction_mode in {"on_train", "on_train_test"}
        else "both",
        save_split_path=save_split_path,
        verbose=verbose,
    )
    if outdir is not None:
        te.write_cross_val(result, outdir=outdir, prefix=prefix)
    return result


def read_model_spec(
    spec: dict, pipeline: bool = False
) -> (
    tuple[
        Filter | None,
        PredBase,
        Transformer,
        Balancer | None,
        rt.Recoder | None,
        Corrector | None,
    ]
    | Pipeline
):
    try:
        M: PredBase = spec.get("m")()
    except TypeError:
        raise ValueError("The model value must be a callable producing the model")
    references = spec.get("r")
    features = spec.get("f")
    encoding = spec.get("e")
    ckwargs = spec.get("c")
    post_process = spec.get("p")
    blacklist = spec.get("l")

    if ckwargs:
        C = Corrector(**ckwargs)
    else:
        C = None

    if encoding is not None and isinstance(encoding, str):
        encoding = encoding()

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
    B: Balancer | None = spec.get("b")
    transformation_name = spec.get("t")
    kwargs: dict = spec.get("k", {})
    if transformation_name == "clr" and references is not None:
        kwargs.update({"features": r_list, "feature_col": "GENEID"})
    T = Transformer(
        transformation_name,
        impute_fn=Imputer(spec.get("i")),
        inplace=False,
        post_process=post_process,
        **kwargs,
    )
    if not pipeline:
        return F, M, T, B, encoding, C
    steps = []
    if B is not None:
        steps.append(B)
    if C is not None:
        steps.append(C)
    if F is not None:
        steps.append(F)
    if encoding is not None:
        steps.append(encoding)
    steps.append(T)
    steps.append(M)

    return Pipeline(steps)


def default_filter_transform(smk_config: dict) -> tuple[Filter, Transformer]:
    dct = smk_config["defaults"]["shallow"]
    t = Transformer(
        dct["transform"], impute_fn=Imputer(dct["imputation"]), inplace=False
    )
    f = Filter(
        features=FEATURE_LISTS[dct["feature_set"]],
        feature_col=dct["feature_col"],
        inplace=False,
    )
    return f, t


def get_model_fn(name: str, config: dict | None = None) -> Callable:
    if name == "MultiLevel":
        model = d_log.MultiLevel
    elif name == "MtcLR":
        model = d_log.MtcLr
    elif name in {"Disyak", "Disyak_All"}:
        model = Disyak

    if config is None:
        config = {}

    def make_model(**kwargs):
        return model(**kwargs, **config)

    return make_model


def get_callback_fn(name: str, config: dict | None = None) -> Callable:
    if name == "early_stop":
        cb = EarlyStopping
    elif name == "average_best":
        cb = AverageBest

    if config is None:
        config = {}

    def make(**kwargs):
        return cb(**kwargs, **config)

    return make


# * Snakemake


def smk_callbacks(config: dict) -> list[Callable]:
    """Retrieve enabled callbacks from config dictionary"""
    callbacks = []
    for callback, dct in config["callbacks"]:
        if dct.get("enabled"):
            callbacks.append(get_callback_fn(callback, dct["params"]))
    return callbacks


class DummySnake:
    def __init__(
        self,
        rule: str,
        input: dict | None = None,
        output: dict | None = None,
        configfile: Path | str | None = None,
        params: dict | None = None,
        log: dict | None = None,
        threads: int | None = None,
        wildcards: dict | None = None,
        scriptdir: str | None = None,
        resources: dict | None = None,
    ) -> None:
        self.rule: str = rule
        if configfile is not None:
            with open(configfile, "r") as f:
                self.config: dict = yaml.safe_load(f)
        else:
            self.config = {}
        self.params: dict = params if params is not None else {}
        self.input: dict = input if input is not None else {}
        self.output: dict = output if output is not None else {}
        self.wildcards: dict = wildcards if wildcards is not None else {}
        self.scriptdir: str | None = None
        self.resources: dict = resources if resources is not None else {}
