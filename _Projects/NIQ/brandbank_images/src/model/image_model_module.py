import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AdamW

from src.model import image_model_catalog as catalog


class ImageModelModule(pl.LightningModule):
    def __init__(
        self,
        label2id,
        learning_rate,
        model_name,
        pretrained=True,
        pooling=True,
    ):
        super().__init__()

        self.model = catalog.get_model(
            model_name, len(label2id), pretrained, pooling
        )
        self.learning_rate = learning_rate

    def forward(self, pixel_values):
        logits = self.model(pixel_values=pixel_values)
        return logits

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct / pixel_values.shape[0]

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)

        self.log_dict(
            {"training_loss": loss, "training_accuracy": accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["pixel_values"].shape[0],
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)

        self.log_dict(
            {
                "validation_loss": loss,
                "validation_accuracy": accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["pixel_values"].shape[0],
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)

        self.log_dict(
            {
                "test_loss": loss,
                "test_accuracy": accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["pixel_values"].shape[0],
        )

        return loss

    def predict_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        logits = self(pixel_values)
        predictions = logits.argmax(-1)
        probabilities = torch.softmax(logits, dim=-1)
        confidences = torch.max(probabilities, dim=-1).values

        if "labels" in batch.keys():
            labels = batch["labels"]
            return {
                "predictions": predictions.cpu().numpy(),
                "confidences": confidences.cpu().numpy(),
                "labels": labels.cpu().numpy(),
            }
        else:
            return {
                "predictions": predictions.cpu().numpy(),
                "confidences": confidences.cpu().numpy(),
            }

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=self.learning_rate)
