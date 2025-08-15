#!/usr/bin/env ipython

from collections.abc import Sequence
from typing import override

import lightning as L
import too_predict.deep.torch_utils as d_ut
import torch
import torch.nn as nn
from torch import Tensor

"""
[1] Polepalli, Vinil. A Novel cVAE-Augmented Deep Learning Framework for Pan-Cancer RNA-Seq Classification. 2025, https://arxiv.org/abs/2508.02743.
"""


class VAE(d_ut.MultiModule):
    def __init__(
        self,
        in_features: int,
        n_classes_per_task: list[int],
        n_units_per_layer: Sequence[int],
        n_latent: int,
        prior,
        conf: d_ut.ModuleConfig | None = None,
        task_names: Sequence | None = None,
    ) -> None:
        """(conditonal) variational autoencoder

        Parameters
        ----------
        n_units_per_layer : Number of output hidden units per fully-connected layer
        """
        super().__init__(in_features, n_classes_per_task, conf, task_names)
        encoder = [
            nn.Linear(in_features=in_features, out_features=n_units_per_layer[0]),
            nn.Relu(),
        ]
        decoder = [nn.LazyLinear(out_features=in_features)]
        n_units_per_layer = n_units_per_layer[1:]
        for i, n in enumerate(n_units_per_layer):
            encoder.append(nn.LazyLinear(out_features=n))
            encoder.append(nn.ReLU())
            decoder.append(nn.LazyLinear(out_features=n_units_per_layer[-i]))
            decoder.append(nn.ReLU())
        decoder = decoder[::-1]

        self.encoder: nn.Sequential = nn.Sequential(*encoder)
        self.decoder: nn.Sequential = nn.Sequential(*decoder)
        self.mu: nn.LazyLinear = nn.LazyLinear(out_features=n_latent)
        self.logvar: nn.LazyLinear = nn.LazyLinear(out_features=n_latent)

    @override
    def forward(self, X):
        encoded = self.encoder(X)

        # Gaussian prior and reparamterization
        # Get parameters of latent variable distribution
        mu, logvar = self.mu(encoded), self.logvar(encoded)
        z = mu + logvar * self.prior(logvar.shape)
        # TODO: how to make this class-conditional?

        decoded = self.decoder(z)
        return encoded, decoded

    def decode(self, X: Tensor | None = None):
        "Sample from latent variable distribution and send to decoder"
        ...

    @override
    def criterion(self, y_pred, y_true, context: str | None = None):
        return super().criterion(y_pred, y_true, context)
        # TODO: should update training step to pass batch to criterion


class Polepalli(VAE):
    """cVAE proposed by [1]"""

    def __init__(
        self,
        in_features: int,
        n_classes_per_task: list[int],
        conf: d_ut.ModuleConfig | None = None,
        task_names: Sequence | None = None,
    ) -> None:
        super().__init__(
            in_features,
            n_classes_per_task,
            conf,
            task_names,
            n_units_per_layer=(256, 128),
            n_latent=10,
        )
