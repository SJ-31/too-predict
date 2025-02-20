#!/usr/bin/env ipython
import numpy as np
from skbio.stats.composition import multi_replace

IMPLEMENTED_IMPUTATION = {"plus_one", "replace_one", "multi_replace", None}


class Imputer:
    def __init__(self, method: str) -> None:
        if method and method.lower() not in IMPLEMENTED_IMPUTATION:
            raise ValueError(f"Imputation method {method} not implemented!")
        self.method: str | None = method

    @staticmethod
    def replace_one(mat: np.ndarray):
        copy = mat.copy()
        copy[copy == 0] = 1
        return copy

    def run(self, counts: np.ndarray):
        match self.method:
            case "plus_one":
                return counts + 1
            case "replace_one":
                return self.replace_one(counts)
            case "multi_replace":
                return multi_replace(counts)
            case None:
                return counts
