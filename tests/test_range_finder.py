#!/usr/bin/env ipython

import too_predict.range_finder as tr
import too_predict.utils as ut

adata = ut.training_data_internal_test()

Rf = tr.RangeFinder(n_bins=50, purity_cutoff=0.3, max_features=5)
id = "ENSG00000000003"
test_ad = adata[:, :1000]
quant = Rf.fit_transform(test_ad)
Rf.get_range(id)

tplot = Rf.range_stripplot(id, hue="Sample_Type")
tplot.show()
