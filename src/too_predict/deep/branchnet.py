#!/usr/bin/env ipython

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import override

import lightning as L
import numpy as np
import too_predict.deep.torch_utils as d_ut
import torch
import torch.nn as nn
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import OptimizerConfig
from sklearn import ensemble
from too_predict.deep.torch_utils import ModuleConfig, MultiModule, iter_cols
from too_predict.utils import load_pickle, write_pickle
from torch import Tensor
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

"""
References

[1] Rodríguez-Salas, D., & Riess, C. (2025). BranchNet: A Neuro-Symbolic Learning Framework for Structured Multi-Class Classification. arXiv preprint arXiv:2507.01781.

"""


class BranchNet(nn.Module):
    loss_p: float = 0.4

    def __init__(self, y_idx: int = 0, device="cpu") -> None:
        super().__init__()
        self.hidden_neurons: int = 0
        self.bn0: nn.Module
        self.w1: Tensor  # Input-to-hidden weights
        self.w2: Tensor  # Hidden-to-output weights
        self.m1: Tensor
        self.in_features: int
        self.out_features: int
        self.bn1: nn.Module
        self.bn2: nn.Module
        self.y_idx: int = 0  # For improved multi-class compatibility
        self.trees_fitted: bool = False
        if isinstance(device, str):
            self.device: torch.device = torch.device(device)
        else:
            self.device = device

    @override
    def forward(self, X: Tensor) -> Tensor:
        x = self.bn0(X)
        if self.training:
            x = F.linear(x, self.w1 * self.m1)
        else:
            x = F.linear(x, self.w1)
        x = self.bn1(x)
        x = F.sigmoid(x)
        x = self.bn2(x)
        x = F.linear(x, self.w2)
        return x

    def build_model_from_ensemble(self, tree_ensemble: ensemble.ExtraTreesClassifier):
        """
        gets all the necessary info from the tree ensemble
        """

        def bf_search(index: int, path: list, is_leaf):
            """
            Record class proportions for each path in a tree to a leaf, and
                features used for splitting
            """
            left_i = tree.children_left[index]
            right_i = tree.children_right[index]
            has_left_leaf = left_i != -1 and is_leaf[left_i]
            has_right_leaf = right_i != -1 and is_leaf[right_i]

            new_path = path[:]
            if tree.feature[index] >= 0:
                new_path.append(tree.feature[index])

            if has_left_leaf or has_right_leaf:
                n_samples: int = tree.n_node_samples[0]
                node_samples: int = tree.n_node_samples[index]
                parents.append(index)
                path_to_parent.append(new_path)
                factor: np.ndarray = node_samples / n_samples
                # Proportion of samples in each node
                dist: list = factor * tree.value[index][0]
                # Proportion of samples of each class for node at `index`
                class_proportion.append(dist)

            if not has_left_leaf:
                bf_search(left_i, new_path, is_leaf)
            if not has_right_leaf:
                bf_search(right_i, new_path, is_leaf)

        def get_w1(size):
            w1 = torch.zeros(size)
            i = 0
            for t, paths_in_tree in enumerate(all_path_to_parent):
                for path in paths_in_tree:
                    w1[i][path] = feature_importance[t][path]
                    i += 1
            w1 *= 1 / np.sqrt(self.in_features)
            return w1.to(self.device)

        def get_w2(size):
            w2 = torch.zeros(size)
            i = 0
            for t, proportions_in_tree in enumerate(all_proportions):
                for p, classes_involved_in_branch in enumerate(proportions_in_tree):
                    w2[:, i] = torch.from_numpy(classes_involved_in_branch)
                    i += 1
            w2 *= 1 / np.sqrt(self.in_features)
            return w2.to(self.device)

        self.in_features = tree_ensemble.n_features_in_
        self.out_features = tree_ensemble.n_classes_

        all_path_to_parent = []
        feature_importance = []  # list of lists where containing the feature indices used to split
        # the tree for a given branch of the tree
        # Computed over the ensemble
        all_proportions = []  # list of class proportions for each tree in the ensemble

        estimators = tree_ensemble.estimators_
        for estimator in estimators:
            tree = estimator.tree_
            is_leaf = (tree.children_left == -1) & (tree.children_right == -1)
            path_to_parent = []
            parents = []
            class_proportion = []
            bf_search(0, [], is_leaf)
            all_path_to_parent.append(path_to_parent)
            self.hidden_neurons += len(parents)
            all_proportions.append(class_proportion)
            importance = torch.zeros(self.in_features).float()
            for path in path_to_parent:
                for feat in path:
                    importance[feat] += 1
            feature_importance.append(importance / importance.max())

        self.bn0 = nn.BatchNorm1d(self.in_features, device=self.device)
        w1 = get_w1((self.hidden_neurons, self.in_features))
        self.m1 = (w1 != 0).to(self.device)
        self.w1 = nn.Parameter(w1).to(self.device)
        self.bn1 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        w2 = get_w2((self.out_features, self.hidden_neurons))
        self.bn2 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        self.w2 = nn.Parameter(w2, requires_grad=False).to(self.device)
        self.trees_fitted = True

    def build_from_dict(self, fn_dict):
        self.w1 = nn.Parameter(fn_dict["w1"])
        self.m1 = (self.w1 != 0).to(self.device)
        self.hidden_neurons = self.w1.shape[0]
        self.w2 = nn.Parameter(fn_dict["w2"], requires_grad=False)
        self.out_features = self.w2.shape[0]
        self.in_features = self.w1.shape[1]
        self.bn0 = nn.BatchNorm1d(self.in_features, device=self.device)
        self.bn1 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        self.bn2 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)

    @staticmethod
    def criterion(y_pred, y_true) -> Tensor:
        ce = nn.functional.cross_entropy(input=y_pred, target=y_true, reduction="none")
        pt = torch.exp(-ce)
        f = 0.5 * (1 - pt) ** 2.5 * ce
        loss = BranchNet.loss_p * f.mean() + (1 - BranchNet.loss_p) * ce.mean()
        return loss

    def _one_epoch(self, dataset: Dataset, optimizer: torch.optim.Optimizer) -> float:
        """Train the model for one epoch."""
        dataloader = DataLoader(
            dataset,
            batch_size=min(256, dataset.__len__()),
            shuffle=True,
            drop_last=True,
        )  # 630
        loss_sum = 0
        for x, y in dataloader:
            y = y.to(self.device)
            y = y[:, self.y_idx]
            y_pred = self.forward(x.to(self.device))
            ce = nn.functional.cross_entropy(input=y_pred, target=y, reduction="none")
            pt = torch.exp(-ce)  # pt = prob of correct class
            f = 0.5 * (1 - pt) ** 2.5 * ce
            loss = BranchNet.loss_p * f.mean() + (1 - BranchNet.loss_p) * ce.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
        return loss_sum / len(dataloader)  # (i+1)

    def val_step(self, dataset) -> float:
        """Validate the model for one epoch."""
        dataloader = DataLoader(
            dataset,
            batch_size=min(256, dataset.__len__()),
            shuffle=True,
            drop_last=True,
        )
        loss_sum = 0
        self.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(dataloader):
                y = y.to(self.device)
                y = y[:, self.y_idx]
                y_pred = self.forward(x.to(self.device))
                ce = nn.functional.cross_entropy(
                    input=y_pred, target=y, reduction="none"
                )
                pt = torch.exp(-ce)
                f = 0.5 * (1 - pt) ** 2.5 * ce
                loss = BranchNet.loss_p * f.mean() + (1 - BranchNet.loss_p) * ce.mean()
                loss_sum += loss.item()
        self.train()
        return loss_sum / len(dataloader)  # (i+1)

    @staticmethod
    def fit_trees(
        train: Dataset | None = None,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        save_to: Path | None = None,
        idx: int = 0,
    ) -> ensemble.ExtraTreesClassifier:
        if train is None and x is None and y is None:
            raise ValueError("Either `train` or `x` and `y` must be provided!")
        if train is not None:
            x, y = train[:]
            x = x.numpy()
            y = y.numpy()[:, idx]
        trees = ensemble.ExtraTreesClassifier()
        trees.fit(x, y)
        if save_to:
            write_pickle(trees, save_to)
        return trees

    def fit(
        self,
        train: Dataset,
        ensemble: ensemble.ExtraTreesClassifier | Path | None = None,
        val: Dataset | None = None,
        epochs=1500,
        learning_rate=0.01,
        show_progress: bool = True,
    ):
        if isinstance(ensemble, Path):
            ensemble = load_pickle(ensemble)
        if not self.trees_fitted:
            self.build_model_from_ensemble(ensemble)

        min_val_loss = 10000000
        patience = 0
        max_patience = 100
        progress_bar = tqdm(range(epochs))
        loss_history = []
        val_loss_history = []
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=180)
        for i, _ in enumerate(progress_bar):
            loss = self._one_epoch(train, optimizer)
            if show_progress:
                progress_bar.set_description(f"Loss: {loss:.6f}")
            loss_history.append(loss)
            if val is not None:
                val_loss = self.val_step(val)
                val_loss_history.append(val_loss)
                scheduler.step(val_loss)
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    torch.save(self.state_dict(), "temporal.pt")
                    patience = 0
                else:
                    patience += 1
                if patience == max_patience:
                    break
        if i < epochs - 1 and val is not None:
            self.load_state_dict(torch.load("temporal.pt", weights_only=True))
        del scheduler


class MultiBranch(MultiModule):
    def __init__(
        self,
        in_features: int,
        n_classes_per_task: list[int],
        conf: ModuleConfig | None = None,
        branchnets: Sequence[BranchNet] | None = None,
        fit_separately: bool = False,
        fit_separately_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        """Initialize MultiBranch - naive implementation of BranchNet for multitask
        setting

        Parameters
        ----------
        branchnets : Optional sequence of pre-fitted BranchNets with frozen weights
        fit_separately : bool
            If pre-fitted BranchNets are not provided, and this is True, this module's
            BranchNets are each fit independently on the training data and their weights
            are frozen before the shared layer is learned
            Otherwise, the BranchNet w1 layers are allowed to learn alongside the shared
            layer
        fit_separately_kwargs : Arguments passed to BranchNet.fit

        """
        super().__init__(
            in_features,
            n_classes_per_task,
            conf=conf,
            **kwargs,
        )
        self.bn_fitted: bool = False  # Whether the branchnets are pre-fitted
        self.trees_fitted: bool = False
        self.bn_fit_separately: bool = fit_separately
        self.bn_fit_kwargs: dict = (
            fit_separately_kwargs
            if fit_separately_kwargs is not None
            else {
                "epochs": 1500,
                "learning_rate": 0.01,
                "show_progress": False,
                "ensemble": None,
            }
        )
        if branchnets is not None:
            self.branchnets: nn.ModuleList = nn.ModuleList(branchnets)
            for net in self.branchnets:
                net.w1.requires_grad = False  # with this
            self.bn_fitted = True
            self.trees_fitted = True
        else:
            self.branchnets = nn.ModuleList(
                [
                    BranchNet(y_idx=i, device=self.conf.init_device)
                    for i, _ in enumerate(self.n_classes)
                ]
            )

        self.shared: nn.Parameter = nn.Parameter(torch.eye(self.in_features))
        self.shared_bias: nn.Parameter = nn.Parameter(torch.zeros(self.in_features))

    def fit_trees(
        self,
        train: Dataset | None = None,
        savedir: Path | str | None = None,
        from_path: bool = False,
    ):
        print("\nFitting ExtraTrees...")
        if savedir is None and from_path:
            raise ValueError("`from_path` was specified, but no path was given")
        if not from_path and train is not None:
            x, y = train[:]
            x = x.cpu().numpy()
            y = y.cpu().numpy()
            savedir = Path(savedir) if isinstance(savedir, str) else savedir
            for i, (name, y_true) in enumerate(zip(self.task_names, iter_cols(y))):
                outpath = savedir.joinpath(name) if savedir is not None else None
                trees = BranchNet.fit_trees(x=x, y=y_true, save_to=outpath)
                self.branchnets[i].build_model_from_ensemble(trees)
        else:
            for task in self.task_names:
                trees = load_pickle(savedir.joinpath(task))
                self.branchnets[i].build_model_from_ensemble(trees)
        self.trees_fitted = True
        print("\nFitting ExtraTrees complete")

    def fit_branchnets(
        self,
        train: Dataset,
        val: Dataset | None = None,
        freeze: bool = False,
        epochs: int = 1500,
        lr: float = 0.01,
        show_progress: bool = True,
    ):
        """Fit branchnets separately using authors' original implementation"""
        print("\nFitting BranchNets...")
        for bn in self.branchnets:
            bn: BranchNet
            bn.fit(
                train=train,
                val=val,
                epochs=epochs,
                learning_rate=lr,
                show_progress=show_progress,
            )
            if freeze:
                for param in bn.parameters():
                    param.requires_grad = False
        self.bn_fitted = True
        print("\nFitting BranchNets complete")

    @override
    def init_out_bias(self, targets) -> None:
        pass

    @override
    def forward(self, X):
        x = nn.functional.relu(F.linear(X, self.shared, bias=self.shared_bias))
        try:
            result = [bn(x) for bn in self.branchnets]
            return tuple(result)
        except AttributeError:
            print("WARNING: branchnets haven't been initialized!")
            return tuple([0, 0])

    @override
    def configure_optimizers(self) -> OptimizerConfig:
        adam = torch.optim.Adam(self.parameters(), lr=0.01)
        return {
            "optimizer": adam,
            "lr_scheduler": CosineAnnealingWarmRestarts(optimizer=adam, T_0=180),
            # Oriignal optimization routine
        }

    @override
    def criterion(self, y_pred, y_true, context: str | None = None):
        total_loss = 0
        for cur_pred, cur_truth in zip(y_pred, iter_cols(y_true)):
            total_loss += BranchNet.criterion(y_pred=cur_pred, y_true=cur_truth)
        return total_loss

    @override
    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        return BranchCallback()


class BranchCallback(Callback):
    def fit_underlying(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        if pl_module.bn_fitted:
            return
        train = trainer.train_dataloader.dataset
        if trainer.val_dataloaders:
            val: Dataset | None = (
                trainer.val_dataloaders.dataset
                if not isinstance(trainer.val_dataloaders, Sequence)
                else trainer.val_dataloaders[0].dataset
            )
        else:
            val = None
        kwargs = pl_module.bn_fit_kwargs.copy()
        kwargs["train"] = train
        kwargs["val"] = val
        if not pl_module.trees_fitted:
            pl_module.fit_trees(train)
        if pl_module.bn_fit_separately:
            pl_module.fit_branchnets(**kwargs)

    @override
    def on_train_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        self.fit_underlying(trainer, pl_module)
