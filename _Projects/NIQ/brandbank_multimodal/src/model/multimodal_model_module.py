import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AdamW

from src.model import multimodal_model_catalog as catalog


class MultimodalModelModule(pl.LightningModule):
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

    def forward(self, input_ids, attention_mask, pixel_values):
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        return logits

    def common_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        logits = self(input_ids, attention_mask, pixel_values)

        criterion = nn.CrossEntropyLoss(
            ignore_index=-100
        )  # Ensure the correct ignore index

        loss = criterion(logits, labels)

        # Calculating accuracy
        predictions = logits.argmax(-1)
        mask = labels != -100  # Mask for non-ignore indices
        masked_predictions = predictions[mask]
        masked_labels = labels[mask]

        correct = (masked_predictions == masked_labels).sum().item()
        non_ignore_count = len(masked_labels)

        # Prevent division by zero
        accuracy = correct / non_ignore_count if non_ignore_count > 0 else 0.0

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)

        self.log_dict(
            {"training_loss": loss, "training_accuracy": accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["input_ids"].shape[0],
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
            batch_size=batch["input_ids"].shape[0],
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
            batch_size=batch["input_ids"].shape[0],
        )

        return loss

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]
        logits = self(input_ids, attention_mask, pixel_values)
        predictions = logits.argmax(-1)
        probabilities = torch.softmax(logits, dim=-1)
        confidences = torch.max(probabilities, dim=-1).values

        if "labels" in batch.keys():
            labels = batch["labels"]

            # Mask for non-ignore indices
            mask = labels != -100
            masked_predictions = predictions[mask]
            masked_confidences = confidences[mask]
            masked_labels = labels[mask]

            return {
                "predictions": masked_predictions.cpu().numpy(),
                "confidences": masked_confidences.cpu().numpy(),
                "labels": masked_labels.cpu().numpy(),
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
