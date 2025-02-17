#!/usr/bin/env ipython

from typing import Callable

import anndata as ad
import numpy as np
from scipy import sparse, stats

from too_predict.normalizer import Normalizer

IMPLEMENTED_SIMULATION = {"dirichlet"}


class Simulator(Normalizer):
    def __init__(
        self,
        adata: ad.AnnData,
        method: str,
        n: int = 5,
        prefix: str = "mc_",
        impute_fn: Callable | None = None,
        inplace=True,
        make_sparse=True,
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
        super().__init__(
            adata, method, impute_fn, inplace, make_sparse, IMPLEMENTED_SIMULATION
        )
        self.prefix = prefix
        self.n = n

    def dirichlet(self, prior: float = 0.5):
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
            inst = instances[:, i, :]
            self.ad.layers[f"{self.prefix}{i}"] = (
                inst if not self.make_sparse else sparse.csr_matrix(inst)
            )

    def run(self, **kwargs) -> ad.AnnData | None:
        match self.method:
            case "dirichlet":
                self.dirichlet(**kwargs)
        if not self.inplace:
            return self.ad
