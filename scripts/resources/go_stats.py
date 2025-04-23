#!/usr/bin/env ipython

# Quick helper script to get statistics of training data when recoding to GO terms

import too_predict.go_utils as gu
import too_predict.utils as ut
from pyhere import here

adata = ut.training_data_internal()

outdir = here("data", "output", "GO_meta")
outdir.mkdir(parents=True, exist_ok=True)
levels = range(3, 7)
coder = gu.RecodeGO()
recoded = coder.fit_transform(adata)
sgo = gu.SubsetGO(subset=recoded.var.index)
sgo.metadata["level"].value_counts().to_csv(here(outdir, "level_counts.csv"))
sgo.metadata.to_csv(here(outdir, "go_recoded_metadata.csv"), index=False)
