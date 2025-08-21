#!/usr/bin/env ipython

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, override

import anndata as ad
import lightning as L
import pandas as pd
import too_predict.deep.torch_utils as d_ut
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, TensorDataset

"""
[1] Polepalli, Vinil. A Novel cVAE-Augmented Deep Learning Framework for Pan-Cancer RNA-Seq Classification. 2025, https://arxiv.org/abs/2508.02743.
"""


class Assumptions(nn.Module):
    def __init__(self, prior: Distribution) -> None:
        super().__init__()
        self.prior: Distribution = prior

    @abstractmethod
    def reparameterize(
        self, X, with_dkl: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Encode tensor X into the latent space
        Optionally, return the KL divergence between the approximate posterior and prior
        """
        ...


class GaussianAssumptions(Assumptions):
    def __init__(self, n_out: int) -> None:
        super().__init__(prior=MultivariateNormal(torch.zeros(n_out), torch.eye(n_out)))
        self.mu: nn.LazyLinear = nn.LazyLinear(out_features=n_out)
        self.var: nn.LazyLinear = nn.LazyLinear(out_features=n_out)

    @override
    def reparameterize(self, X, with_dkl=False) -> Tensor | tuple[Tensor, Tensor]:
        mu_hat, var_hat = self.mu(X), self.var(X)
        scale = nn.functional.softplus(var_hat) + 1e-8
        dist = torch.distributions.MultivariateNormal(
            mu_hat, scale_tril=torch.diag_embed(scale)
        )
        z = dist.rsample()
        if not with_dkl:
            return z
        return z, torch.distributions.kl.kl_divergence(dist, self.prior).mean()


class GaussianAssumptionsClosed(Assumptions):
    def __init__(self, n_out: int) -> None:
        super().__init__(prior=MultivariateNormal(torch.zeros(n_out), torch.eye(n_out)))
        self.n_out: int = n_out
        self.mu: nn.LazyLinear = nn.LazyLinear(out_features=n_out)
        self.var: nn.LazyLinear = nn.LazyLinear(out_features=n_out)

    @override
    def reparameterize(self, X, with_dkl=False) -> Tensor | tuple[Tensor, Tensor]:
        mu, log_var = self.mu(X), self.var(X)
        std = torch.exp(0.5 * log_var)
        z = mu + log_var * torch.randn_like(std)
        if not with_dkl:
            return z
        dkl = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        return z, dkl


# [2025-08-21 Thu] The closed form seems to do better for some reason

# TODO: if the VAE default probability distribution doesn't work out, could you use
# it to estimate parameters for a general NB model?
#


class BaseVAE(d_ut.BaseNN):
    def __init__(
        self,
        in_features: int,
        n_classes_per_task: list[int],
        n_latent: int,
        assumptions: Assumptions,
        activation: nn.Module = nn.ReLU,
        conf: d_ut.ModuleConfig | None = None,
    ) -> None:
        """(conditonal) variational autoencoder

        Parameters
        ----------
        n_units_per_layer : Number of output hidden units per fully-connected layer
        """
        super().__init__(in_features, n_classes_per_task, conf)
        self.assumptions: Assumptions = assumptions

        self.cache: dict[str, tuple[bool, list]] = {}
        for context in ("train", "test", "val"):
            self.cache[f"{context}_loss"] = (False, [])
            self.cache[f"{context}_recon_loss"] = (False, [])
            self.cache[f"{context}_dkl"] = (False, [])
        if isinstance(self.conf.cache, str):
            self.set_cache(self.conf.cache)
        elif self.conf.cache is not None:
            for c in self.conf.cache:
                self.set_cache(c)

    @override
    def training_step(self, batch, batch_idx):
        x, y = batch
        z, dist = self(x)  # Latent variable and posterior distribution
        decoded = self.decode((z, y))
        return self.recon_loss(
            x_pred=decoded, x_true=x, y_true=y, dist=dist, context="train"
        )

    def recon_loss(
        self,
        x_pred: Tensor | None = None,
        x_true: Tensor | None = None,
        y_pred: Tensor | None = None,
        y_true: Tensor | None = None,
        context: str | None = None,
    ):
        recon_loss = nn.functional.mse_loss(x_pred, x_true)
        self.log(f"{context}_recon_loss", recon_loss)
        return recon_loss

    @override
    def forward(self, X) -> Tensor | tuple[Tensor, ...]:
        """Return point from latent space, and posterior distribution
        with parameters estimated by encoder
        """
        encoded = self.encoder(X)
        z: Tensor
        if self.training:
            z, dkl = self.assumptions.reparameterize(encoded, with_dkl=True)
            return z, dkl
        return self.assumptions.reparameterize(encoded, with_dkl=False)

    def decode(self, batch: tuple[Tensor, ...]):
        "Map from latent distribution to approximate inputs, optionally with class labels"
        raise NotImplementedError()

    @override
    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        return super().predict_step(*args, **kwargs)

    @override
    def validation_step(self, *args: Any, **kwargs: Any):
        return super().validation_step(*args, **kwargs)

    # @override
    # def test_step(self, batch, batch_idx):
    #     return super().test_step(*args, **kwargs)

    def from_prior(self, n: int) -> Tensor:
        "Randomly draw `n` samples from prior"
        return self.assumptions.prior.rsample((n,))

    def make_one_hot(self, labels: Tensor) -> tuple[Tensor, ...]:
        """Create a one-hot vector from labels, accounting for class counts"""
        encoded = [
            one_hot(labs, count)
            for labs, count in zip(d_ut.iter_cols(labels), self.n_classes)
        ]
        return tuple(encoded)


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
            assumptions=GaussianAssumptionsClosed(n_out=n_latent),
        )
        self.prior: Distribution | None = torch.distributions.MultivariateNormal(
            torch.zeros(n_latent), torch.eye(n_latent)
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
        one_hots = self.make_one_hot(y)
        input = torch.cat([x, *one_hots], dim=1)
        z, dkl = self(input)
        decoded = self.decode((z, y))
        r_loss = self.recon_loss(x_pred=decoded, x_true=x, y_true=y, context="train")
        loss = dkl + r_loss
        self.log("train_loss", loss)
        return loss

    @override
    def decode(self, batch):
        X, y = batch
        one_hots = self.make_one_hot(y)
        decoded = self.decoder(torch.cat([X, *one_hots], dim=1))
        return decoded

    def sample(self, labels: Tensor):
        x = self.from_prior(n=labels.shape[0])
        decoded: Tensor = self.decode((x, labels))
        return decoded.detach()

    def sample_as(
        self,
        labels: Tensor,
        type: Literal["anndata", "dataset"] = "dataset",
        label_names: Sequence | None = None,
    ) -> ad.AnnData | Dataset:
        x = self.sample(labels)
        if type == "dataset":
            return TensorDataset(x, labels)
        elif type == "anndata":
            if label_names is None:
                label_names = [f"label_{i}" for i in range(labels.shape[1])]
            obs = pd.DataFrame(
                {name: labs for name, labs in zip(label_names, d_ut.iter_cols(labels))}
            )
            return ad.AnnData(X=x.numpy(), obs=obs)


# TODO: you could use an NB prior
