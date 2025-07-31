#!/usr/bin/env ipython

import pickle
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import override

import lightning as L
import numpy as np
import too_predict.deep.torch_utils as d_ut
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import OptimizerConfig
from sklearn import ensemble
from too_predict.deep.torch_utils import MultiModule, iter_cols
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


class BranchNet(L.LightningModule):
    loss_p: float = 0.4

    def __init__(self, y_idx: int = 0) -> None:
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
        self.m1 = w1 != 0
        self.w1 = nn.Parameter(w1)
        self.bn1 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        w2 = get_w2((self.out_features, self.hidden_neurons))
        self.bn2 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        self.w2 = nn.Parameter(w2, requires_grad=False)

    def build_from_dict(self, fn_dict):
        self.w1 = nn.Parameter(fn_dict["w1"])
        self.m1 = self.w1 != 0
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
        ensemble: ensemble.ExtraTreesClassifier | Path,
        train: Dataset,
        val: Dataset | None = None,
        epochs=1500,
        learning_rate=0.01,
    ):
        if isinstance(ensemble, Path):
            ensemble = load_pickle(ensemble)
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


# TODO: wrap this up with MultiBranch
# Make two versions: one where the underlying branch nets learn alongside the
# shared weights
# and another where the branch nets are fixed and only the shared weights update
# The default implementation also uses early stopping by monitoring validation loss


class MultiBranch(MultiModule):
    def __init__(
        self,
        in_features: int,
        n_classes_per_task: list[int],
        branchnets: Sequence[BranchNet] | None = None,
        record_metrics: bool = True,
        task_names: Sequence[str] | None = None,
        task_weights: Tensor | Sequence | None = None,
        l1_pars: dict | None = None,
        l2_pars: dict | None = None,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
        scheduler_config: dict | None = None,
        cache: str | None | Sequence = None,
        log_norm: bool = False,
        scaler: d_ut.TorchScaler | None = None,
    ) -> None:
        super().__init__(
            in_features,
            n_classes_per_task,
            record_metrics,
            task_names,
            task_weights,
            l1_pars,
            l2_pars,
            optimizer_fn,
            scheduler_fn,
            scheduler_config,
            cache,
            log_norm,
            scaler,
        )
        self.bn_fitted: bool = False
        if branchnets is not None:
            self.branchnets: nn.ModuleList = nn.ModuleList(branchnets)
            for net in self.branchnets:
                net.w1.requires_grad = False  # with this
            self.bn_fitted = True
        else:
            self.branchnets = nn.ModuleList(
                [BranchNet(y_idx=i) for i, _ in enumerate(self._n_classes)]
            )

        self.shared: nn.Parameter = nn.Parameter(torch.eye(self._in_features))
        self.shared_bias: nn.Parameter = nn.Parameter(torch.zeros(self._in_features))

    def fit_trees(
        self,
        train: Dataset | None = None,
        savedir: Path | str | None = None,
        from_path: bool = False,
    ):
        if savedir is None and from_path:
            raise ValueError("`from_path` was specified, but no path was given")
        if not from_path and train is not None:
            x, y = train[:]
            x = x.numpy()
            y = y.numpy()
            savedir = Path(savedir) if isinstance(savedir, str) else savedir
            for i, (name, y_true) in enumerate(zip(self._task_names, iter_cols(y))):
                outpath = savedir.joinpath(name) if savedir is not None else None
                trees = BranchNet.fit_trees(x=x, y=y_true, save_to=outpath)
                self.branchnets[i].build_model_from_ensemble(trees)
        else:
            for task in self._task_names:
                trees = load_pickle(savedir.joinpath(task))
                self.branchnets[i].build_model_from_ensemble(trees)

    @override
    def forward(self, X):
        x = F.linear(X, self.shared, bias=self.shared_bias)
        result = [bn(x) for bn in self.branchnets]
        return tuple(result)

    @override
    def configure_optimizers(self) -> OptimizerConfig:
        adam = torch.optim.Adam(self.parameters(), lr=0.01)
        return {
            "optimizer": adam,
            "lr_scheduler": CosineAnnealingWarmRestarts(optimizer=adam, T_0=180),
        }

    @override
    def criterion(self, y_pred, y_true, context: str | None = None):
        total_loss = 0
        for cur_pred, cur_truth in zip(y_pred, iter_cols(y_true)):
            total_loss += BranchNet.criterion(y_pred=cur_pred, y_true=cur_truth)
        return total_loss
