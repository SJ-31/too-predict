#!/usr/bin/env ipython

import numpy as np
import rpy2.robjects as ro
from scipy import stats

from too_predict.utils import r_cleanup

IMPLEMENTED_SIMULATION = {"dirichlet", "dirichlet_scale"}


class Simulator:
    def __init__(
        self,
        method: str,
        data: np.ndarray,
        n_instances: int = 5,
        prefix: str = "mc_",
        **kwargs,
    ) -> None:
        """Class for simulating count data

        Parameters
        ----------
        n : the number of simulations to carry out. Exact use on the method
        prefix : prefix for the layers in the adata object to hold the simulated counts


        Returns
        -------
        if `inplace` == False, an adata object where simulated counts are stored in the
        layers, prefixed with `prefix`

        Otherwise, the given adata is modified inplace
        """
        self.method = method
        self.counts = data
        self.prefix = prefix
        self.n = n_instances
        self.simulated = False
        self._instances = np.zeros([n_instances, *self.counts.shape])
        self.kwargs = kwargs

    @property
    def instances(self) -> np.ndarray:
        if not self.simulated:
            raise ValueError("Class hasn't been called yet!")
        return self._instances

    @instances.setter
    def instances(self, value):
        self._instances = value

    @instances.deleter
    def instances(self):
        del self._instances

    @r_cleanup
    def dirichlet_scale(self, gamma, condition_col) -> None:
        """Generate dirichlet instances with ALDEx2's scale simulation

        Parameters
        ----------
        gamma : uncertainty
        group : group

        Returns
        -------


        Notes
        -----

        """
        ro.r("library(ALDEx2)")
        with (ro.default_converter + ro.numpy2ri.converter).context():
            ro.globalenv["counts"] = np.transpose(self.counts)
        ro.globalenv["names"] = ro.StrVector(self.ad.var["gene_id"])
        ro.r("rownames(counts) <- names")
        gamma = 1e-3
        ro.globalenv["cond"] = ro.StrVector(self.ad.obs[condition_col])
        ro.r("mat <- model.matrix(~ cond)")
        ro.r(f"""
        clr <- aldex.clr(counts, mat, gamma = {gamma}, mc.samples = {int(self.n)},
        verbose = TRUE)
        """)

        kept_features = list(ro.r("getFeatureNames(clr)"))
        if not self.counts_only:
            self.adata = self.adata[:, self.adata.var["gene_id"].isin(kept_features)]

        with (ro.default_converter + ro.numpy2ri.converter).context():
            for i in range(self.n):
                inst = np.transpose(ro.r(f"getDirichletSample(clr, {i + 1})"))
                self._instances[i] = inst

    def __call__(self):
        match self.method:
            case "dirichlet":
                self.dirichlet(**self.kwargs)
            case "dirichlet_scale":
                self.dirichlet_scale(**self.kwargs)
            case _:
                raise ValueError("Method not implemented!")
        self.simulated = True
        return self.instances

    def dirichlet(self, prior: float = 0.5) -> None:
        """
        Obtain random dirichlet instances from read counts (based on the ALDEx2 R package)
        For each sample, use the vector of read counts as the concentration parameter

        TODO: how to define the prior?
        TODO: this isn't very performant, look into Rust (rand_distr) or parallel
            n must be < 10

        Parameters:
        -----------
        prefix : prefix for the layer in the resulting adata object containing each instance

        Notes
        -----
        Don't run this with self.make_sparse, it will take too long
        """
        arr = self.counts + prior
        instances = np.apply_along_axis(
            lambda x: stats.dirichlet.rvs(x, self.n), 1, arr
        )
        for i in range(instances.shape[1]):
            self._instances[i] = instances[:, i, :]
