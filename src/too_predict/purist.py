#!/usr/bin/env ipython
#
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit


# Purist subtyping for pancreatic cancer
# Reproduced from
# Rashid NU, Peng XL, Jin C, Moffitt RA, Volmar KE, Belt BA, Panni RZ, Nywening TM, Herrera SG, Moore KJ, Hennessey SG, Morrison AB, Kawalerski R, Nayyar A, Chang AE, Schmidt B, Kim HJ, Linehan DC, Yeh JJ. Purity Independent Subtyping of Tumors (PurIST), A Clinically Robust, Single-sample Classifier for Tumor Subtyping in Pancreatic Cancer. Clin Cancer Res. 2020 Jan 1;26(1):82-92. doi: 10.1158/1078-0432.CCR-19-1467. Epub 2019 Nov 21. PMID: 31754050; PMCID: PMC6942634.
# Coefficients taken from supplementary table 5
def purist(adata: ad.AnnData, name_col: str = "GENENAME") -> pd.DataFrame:
    adata = adata.copy()
    counts: np.ndarray = adata.X if not sparse.issparse(adata.X) else adata.X.toarray()
    intercept = -6.815
    tsp2coeff = {
        ("GPR87", "REG4"): 1.994,
        ("KRT6A", "ANXA10"): 2.031,
        ("BCAR3", "GATA6"): 1.618,
        ("PTGES", "CLDN18"): 0.922,
        ("ITGA3", "LGALS4"): 1.059,
        ("C16orf74", "DDC"): 0.929,
        ("S100A2", "SLC40A1"): 2.505,
        ("KRT5", "CLRN3"): 0.485,
    }
    n_samples = adata.shape[0]
    score: np.ndarray = np.zeros(shape=(9, n_samples))
    score[8, :] = intercept
    for i, (k, v) in enumerate(tsp2coeff.items()):
        a_index = np.where(adata.var[name_col] == k[0])[0][0]
        b_index = np.where(adata.var[name_col] == k[1])[0][0]
        ranked = counts[:, a_index] > counts[:, b_index]
        score[i, :] = ranked * v

    print(score)
    tsp_score = np.sum(score, axis=0)
    proba = expit(tsp_score)
    # convert to probability with inverse logit

    classification = (
        pd.Series(proba > 0.5)
        .replace({True: "basal-like", False: "classical"})
        .astype(str)
    )
    adata.obs.loc[:, "purist_tsp_score"] = tsp_score
    adata.obs.loc[:, "purist_probability"] = proba
    adata.obs.loc[:, "purist_classification"] = classification.values
    return adata.obs


def test_purist():
    genes = [
        "GPR87",
        "REG4",
        "KRT6A",
        "ANXA10",
        "BCAR3",
        "GATA6",
        "PTGES",
        "CLDN18",
        "ITGA3",
        "LGALS4",
        "C16orf74",
        "DDC",
        "S100A2",
        "SLC40A1",
        "KRT5",
        "CLRN3",
    ]
    X = np.array(
        [
            [
                37.7,
                2.36,
                136.8,
                15.78,
                15.1,
                1.07,
                35.67,
                241.68,
                393.2,
                201.5,
                2.93,
                31.76,
                1.76,
                0.8,
                7.17,
                39.07,
            ]
        ]
    )
    adata = ad.AnnData(
        X=X,
        var=pd.DataFrame({"GENENAME": genes}, index=genes),
        obs=pd.DataFrame(index=["sample1"]),
    )
    print(purist(adata))
