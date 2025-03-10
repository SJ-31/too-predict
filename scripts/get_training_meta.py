#!/usr/bin/env ipython

from pyhere import here
from too_predict.utils import training_data_internal

adata = training_data_internal()
adata.obs.to_csv(here("data", "training_data_obs.csv"), index=False)
adata.var.to_csv(here("data", "training_data_var.csv"), index=False)
