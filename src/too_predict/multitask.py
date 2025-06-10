#!/usr/bin/env ipython


import numpy as np
import sklearn.preprocessing as sp


class MultiEncoder:
    """Helper class for encoding multiple string labels in a 2D array of
    shape n_samples x n_label_types
    """

    encoders: list[sp.LabelEncoder] | None

    def __init__(self, to_type=int) -> None:
        self.encoders = None
        self.to_type: type = to_type

    def _validate_arr(self, arr: np.ndarray) -> None:
        if len(arr.shape) != 2:
            raise ValueError("Given array must be 2D!")

    def fit(self, arr: np.ndarray):
        self.encoders = []
        self._validate_arr(arr)
        for i in range(arr.shape[1]):
            encoder = sp.LabelEncoder()
            encoder.fit(arr[:, i])
            self.encoders.append(encoder)

    def _t_helper(self, arr: np.ndarray, inverse: bool = False):
        self._validate_arr(arr)
        new = np.zeros_like(arr)
        if len(self.encoders) != arr.shape[1]:
            raise ValueError(
                "The array to transform has different number of label classes to those seen in fit!"
            )
        for i, encoder in enumerate(self.encoders):
            if inverse:
                new[:, i] = encoder.inverse_transform(arr[:, i])
            else:
                new[:, i] = encoder.transform(arr[:, i])
        return new.astype(self.to_type)

    def fit_transform(self, arr: np.ndarray) -> np.ndarray:
        self.fit(arr)
        return self.transform(arr)

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return self._t_helper(arr, False)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return self._t_helper(arr, True)
