#!/usr/bin/env ipython

from collections.abc import Sequence
from typing import Any, override

import lightning as L
import too_predict.deep.torch_utils as d_ut
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

"""
[1] Polepalli, Vinil. A Novel cVAE-Augmented Deep Learning Framework for Pan-Cancer RNA-Seq Classification. 2025, https://arxiv.org/abs/2508.02743.
"""


# TODO: if the VAE default probability distribution doesn't work out, could you use
# it to estimate parameters for a general NB model?
#
class DistModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    def forward(self, X) -> tuple[Tensor, Distribution]:
        raise NotImplementedError()


class GaussianApprox(DistModule):
    """Produces latent variable with a Gaussian form"""

    def __init__(self, n_out: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mu: nn.LazyLinear = nn.LazyLinear(out_features=n_out)
        self.var: nn.LazyLinear = nn.LazyLinear(out_features=n_out)

    @override
    def forward(self, X):
        mu_hat, var_hat = self.mu(X), self.var(X)
        dist = torch.distributions.MultivariateNormal(
            mu_hat, scale_tril=torch.diag_embed(var_hat)
        )
        z = dist.rsample()
        return z, dist


class BaseVAE(d_ut.BaseNN):
    def __init__(
        self,
        in_features: int,
        n_classes_per_task: list[int],
        n_latent: int,
        approx_posterior: DistModule,
        prior: Distribution | None = None,
        activation: nn.Module = nn.ReLU,
        conf: d_ut.ModuleConfig | None = None,
    ) -> None:
        """(conditonal) variational autoencoder

        Parameters
        ----------
        n_units_per_layer : Number of output hidden units per fully-connected layer
        """
        super().__init__(in_features, n_classes_per_task, conf)

        self.approx_posterior: nn.Module = approx_posterior
        self.prior: Distribution | None = prior

        self.cache: dict[str, tuple[bool, list]] = {
            "train_loss": (False, []),
            "train_acc": (False, []),  # TODO: what metrics could you report instead?
            "val_acc": (False, []),
            "val_loss": (False, []),
            "test_loss": (False, []),
            "test_acc": (False, []),
        }
        if isinstance(self.conf.cache, str):
            self.set_cache(self.conf.cache)
        elif self.conf.cache is not None:
            for c in self.conf.cache:
                self.set_cache(c)

    @override
    def training_step(self, batch, batch_idx):
        x, y = batch
        z, dist = self(x)  # Latent variable and posterior distribution
        decoded = self.decode(z, y)
        return self.criterion(
            x_pred=decoded, x_true=x, y_true=y, dist=dist, context="train"
        )

    def criterion(
        self,
        x_pred: Tensor | None = None,
        x_true: Tensor | None = None,
        y_pred: Tensor | None = None,
        y_true: Tensor | None = None,
        context: str | None = None,
        dist: Distribution | None = None,
    ):
        kl_div = torch.distributions.kl.kl_divergence(dist, self.prior)
        recon_loss = nn.functional.mse_loss(x_pred, x_true)
        return kl_div + recon_loss

    @override
    def forward(self, X) -> tuple[Tensor, Distribution]:
        """Return point from latent space, and posterior distribution
        with parameters estimated by encoder
        """
        encoded = self.encoder(X)
        z: Tensor
        dist: Distribution
        z, dist = self.approx_posterior(encoded)
        return z, dist

    def decode(self, X: Tensor | None = None, y: Tensor | None = None):
        "Map from latent distribution to approximate inputs, optionally with class labels"
        raise NotImplementedError()

    @override
    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        return super().predict_step(*args, **kwargs)

    @override
    def validation_step(self, *args: Any, **kwargs: Any):
        return super().validation_step(*args, **kwargs)

    @override
    def test_step(self, *args: Any, **kwargs: Any):
        return super().test_step(*args, **kwargs)

    def sample(self, **kwargs):
        return self.decode(self.prior.rsample())


class cVAE(BaseVAE):
    """cVAE proposed by [1]"""

    def __init__(
        self,
        in_features: int,
        n_classes_per_task: list[int],
        conf: d_ut.ModuleConfig | None = None,
        n_units_per_layer=(256, 128),
        n_latent=10,
    ) -> None:
        super().__init__(
            in_features=in_features + sum(n_classes_per_task),
            n_latent=n_latent,
            n_classes_per_task=n_classes_per_task,
            conf=conf,
            approx_posterior=GaussianApprox(n_out=n_latent),
        )
        self.prior: Distribution | None = (
            torch.distributions.MultivariateNormal(
                torch.zeros(n_latent), torch.eye(n_latent)
            ),
        )
        stacked_in_features = in_features + sum(
            n_classes_per_task
        )  # Takes the genes AND a one-hot encoding of class labels
        encoder = [
            nn.Linear(
                in_features=stacked_in_features, out_features=n_units_per_layer[0]
            ),
            nn.ReLU(),
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

    @override
    def training_step(self, batch, batch_idx):
        x, y = batch
        one_hots = [nn.functional.one_hot(y_i) for y_i in d_ut.iter_cols(y)]
        input = torch.cat([x, *one_hots], dim=1)
        z, dist = self(input)
        decoded = self.decode(z, y)
        return self.criterion(
            x_pred=decoded, x_true=x, y_true=y, dist=dist, context="train"
        )

    @override
    def decode(self, X: Tensor | None = None, y: Tensor | None = None):
        one_hots = [nn.functional.one_hot(y_i) for y_i in d_ut.iter_cols(y)]
        decoded = self.decoder(torch.cat([X, *one_hots], dim=1))
        return decoded


# TODO: you could use an NB prior
