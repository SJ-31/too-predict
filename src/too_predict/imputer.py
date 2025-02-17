#!/usr/bin/env ipython
import numpy as np

IMPLEMENTED_IMPUTATION = {"plus_one", "replace_one", None}


class Imputer:
    def __init__(self, method: str) -> None:
        if method and method.lower() not in IMPLEMENTED_IMPUTATION:
            raise ValueError(f"Imputation method {method} not implemented!")
        self.method: str | None = method

    @staticmethod
    def plus_one(mat: np.ndarray):
        return mat + 1

    @staticmethod
    def replace_one(mat: np.ndarray):
        copy = mat.copy()
        copy[copy == 0] = 1
        return copy

    def run(self, counts: np.ndarray):
        match self.method:
            case "plus_one":
                return self.plus_one(counts)
            case "replace_one":
                return self.replace_one(counts)
            case None:
                return counts
