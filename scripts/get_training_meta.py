#!/usr/bin/env ipython

import pandas as pd
from pyhere import here
from too_predict.utils import training_data_internal

adata = training_data_internal()
adata.obs.to_csv(here("data", "training_data_obs.csv"), index=False)
crosses = pd.crosstab(adata.obs["primary_site"], adata.obs["tumor_type"])
crosses.melt(ignore_index=False).query("value > 0").sort_index().reset_index().to_csv(
    here("data", "training_data_tissue-site_assoc.csv"), index=False
)
adata.var.to_csv(here("data", "training_data_var.csv"), index=False)

adata_p = training_data_internal(label="primary_site")
primary_sites = list(set(adata_p.obs["primary_site"]))

here("data", "primary_sites.txt").write_text("\n".join(primary_sites))
