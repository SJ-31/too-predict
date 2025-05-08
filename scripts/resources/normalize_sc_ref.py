#!/usr/bin/env ipython

from pathlib import Path

import anndata as ad
import pandas as pd
import scib
import too_predict.utils as ut
from pyhere import here
from too_predict.r_utils import pooled_normalization

test = str(Path.home()) == "/home/shannc"
storage_dir = (
    here("remote", "public_data") if not test else here("data", "tests", "scr_ref")
)
combined = ad.read_h5ad(here(storage_dir, "sc_ref_all.h5ad"))
if test:
    combined = combined[:, :1000]
    batch_size = 2
else:
    batch_size = 100


pooled_normalization(combined)

method = "scanorama"
corrected = ut.scanorama_correct(
    combined, batch_key="source", batch_size=batch_size, hvg=4000
)

ut.pca_to_leiden(combined)
ut.pca_to_leiden(corrected)

scores = scib.metrics.metrics_fast(
    combined, corrected, batch_key="source", label_key="cell_type"
)
# metrics_fast only computes
# - hvg_overlap
#   1 is best
# - cell type ASW (ASW_label), silhouette() function
#   1 is best
# - isolated_label_silhouette/isolated_labels()
#   This is the same as cell type ASW, but considering only "isolated" labels, which
#   are the cell types found in the fewest batches i.e. highly batch-specific cell types
#   1 is best
# - silhouette_batch() (ASW_label/batch)
#   1 is best
# - pcr_comparison() (PCR_batch)
#   1 is best (greater difference between batches)
# - graph_connectivity() (graph_conn)
#   1 is best (all cells with same identity connected)
scores = pd.DataFrame({"metric": scores.index, "value": scores.iloc[:, 0]}).reset_index(
    drop=True
)
scores.to_csv(here("data", "output", f"sc_ref_{method}_metrics.csv"), index=False)
corrected.write_h5ad(here(storage_dir, "sc_ref_all_corrected.h5ad"))
