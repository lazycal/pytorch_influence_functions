from typing import List, Union

import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Subset

from pytorch_influence_functions.test_influence_functions.utils.dummy_dataset import DummyDataset


class LogisticRegression(LightningModule):
    def __init__(
            self, n_classes=3, n_features=10, idx_to_remove=None
    ):
        super().__init__()
        self.training_set = DummyDataset(n_classes=n_classes, n_features=n_features)

        if idx_to_remove is not None:
            self.idx_to_remove = idx_to_remove
            all_indices = set(np.arange(self.training_set.data.shape[0]))
            indices_to_keep = all_indices - self.idx_to_remove

            self.training_set = Subset(self.training_set, indices_to_keep)

        self.test_set = DummyDataset(n_classes=n_classes, n_features=n_features, train=False)

        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        return x

    def loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def train_dataloader(self, batch_size=32) -> DataLoader:
        return DataLoader(self.training_set, batch_size=batch_size, shuffle=True, num_workers=4)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        loss = self.loss(logits, y)
        softmax = F.softmax(logits, dim=0)
        _, predicted = torch.max(softmax.data, 1)
        correct = (predicted == y).sum().double()
        accuracy = correct / x.size(0)

        return {"val_loss": loss, "val_acc": accuracy}

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_set, batch_size=32, num_workers=4)
