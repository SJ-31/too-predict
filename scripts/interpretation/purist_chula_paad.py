#!/usr/bin/env ipython

import pandas as pd
import too_predict.utils as ut
from pyhere import here
from too_predict.purist import purist

adata = ut.training_data_internal()
adata = adata[adata.obs["Project_ID"] == "CHULA-PAAD"]
adata.obs = adata.obs.loc[:, ["Case_ID"]]
result: pd.DataFrame = purist(adata)
result.to_csv(here("data", "output", "purist_pdac_results.csv"), index=False)
