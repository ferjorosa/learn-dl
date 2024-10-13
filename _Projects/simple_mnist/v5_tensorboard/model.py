import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from torch import nn, optim
from metrics import MyAccuracy


class FFNNLightningModule(pl.LightningModule):
    def __init__(self, input_size, num_classes, learning_rate=0.001):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes
        )
        # Define our own metric, to show how to do it:
        self.my_accuracy = MyAccuracy()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        # This metric estimation is very slow, we should use a Profiler
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        my_accuracy = self.my_accuracy(scores, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
                "train_my_accuracy": my_accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def _common_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        y = batch["labels"]

        # x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
