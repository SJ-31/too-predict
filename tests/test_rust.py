#!/usr/bin/env ipython

import anndata as ad

# Data retrieval
import numpy as np
import pandas as pd
import pooch

# Core scverse libraries
import scanpy as sc
import sklearn as sk
import sklearn.feature_selection as fs
import too_predict._rust_helpers as rh
from numpy import float64
from sklearn.ensemble import RandomForestClassifier
from too_predict.model import AlrBase, AlrEstimator, PredBase, SimEstimator
from too_predict.normalizer import IMPLEMENTED_NORMALIZATION, Normalizer
from too_predict.simulation import Simulator

sk.set_config(enable_metadata_routing=True)

# #  --- CODE BLOCK ---
#
EXAMPLE_DATA = pooch.create(
    path=pooch.os_cache("scverse_tutorials"),
    base_url="doi:10.6084/m9.figshare.22716739.v1/",
)
EXAMPLE_DATA.load_registry_from_doi()

samples = {
    "s1d1": "s1d1_filtered_feature_bc_matrix.h5",
    "s1d3": "s1d3_filtered_feature_bc_matrix.h5",
}
adatas = {}

for sample_id, filename in samples.items():
    path = EXAMPLE_DATA.fetch(filename)
    sample_adata = sc.read_10x_h5(path)
    sample_adata.var_names_make_unique()
    adatas[sample_id] = sample_adata

adata = adatas["s1d1"]
adata.obs_names_make_unique()
# adata

rng = np.random.default_rng(seed=92)


arr = adata.X.toarray().astype(float64)
random_mat = np.random.rand(10, 1000)

# result = rh.phi_matrix(arr, True)
# #  --- CODE BLOCK ---
adata = adata[:50, :100]
random_labels = rng.choice(["foo", "bar", "baz", "bat"], adata.shape[0])
adata.obs["tumor_type"] = random_labels

tree = RandomForestClassifier()

myest = PredBase("clr", "plus_one", tree)


# #  --- CODE BLOCK ---
myalr = AlrBase(
    "plus_one",
    tree,
    references=[
        "ENSG00000237613",
        "ENSG00000186092",
        "ENSG00000238009",
        "ENSG00000239945",
    ],
    feature_col="gene_ids",
)

counts = np.random.randint(1, 1000, adata.shape)
adata.X = counts
myalr.fit(adata, var_col="gene_ids")


res = myalr.predict(adata, var_col="gene_ids")
# alrest = AlrEstimator(
#     tree,
# )
# for_alr = pd.DataFrame(counts, columns=adata.var["gene_ids"], index=None)

# alrest.fit(for_alr, random_labels)
# alrest.predict_proba(for_alr)

# #  --- CODE BLOCK ---
sim = Simulator(adata.X, "dirichlet")
result = sim.run()

simest = SimEstimator("dirichlet", tree)
simest.fit(counts, adata.obs["tumor_type"])
simest.predict(counts)
