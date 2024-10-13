import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AdamW

from src.model import multimodal_model_catalog as catalog


class MultimodalModel3LossesModule(pl.LightningModule):
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
        text_logits, image_logits, multimodal_logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        return text_logits, image_logits, multimodal_logits

    def common_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        text_logits, image_logits, multimodal_logits = self(
            input_ids, attention_mask, pixel_values
        )

        criterion = nn.CrossEntropyLoss(
            ignore_index=-100
        )  # Ensure the correct ignore index

        mask = labels != -100  # Mask for non-ignore indices
        masked_labels = labels[mask]
        non_ignore_count = len(masked_labels)

        # Text
        text_loss = criterion(text_logits, labels)
        text_predictions = text_logits.argmax(-1)
        masked_text_predictions = text_predictions[mask]
        text_correct = (masked_text_predictions == masked_labels).sum().item()
        text_accuracy = (
            text_correct / non_ignore_count if non_ignore_count > 0 else 0.0
        )

        # Image
        image_loss = criterion(image_logits, labels)
        image_predictions = image_logits.argmax(-1)
        masked_image_predictions = image_predictions[mask]
        image_correct = (
            (masked_image_predictions == masked_labels).sum().item()
        )
        image_accuracy = (
            image_correct / non_ignore_count if non_ignore_count > 0 else 0.0
        )

        # Multimodal
        multimodal_loss = criterion(multimodal_logits, labels)
        multimodal_predictions = multimodal_logits.argmax(-1)
        masked_multimodal_predictions = multimodal_predictions[mask]
        multimodal_correct = (
            (masked_multimodal_predictions == masked_labels).sum().item()
        )
        multimodal_accuracy = (
            multimodal_correct / non_ignore_count
            if non_ignore_count > 0
            else 0.0
        )

        # Total loss
        total_loss = text_loss + image_loss + multimodal_loss

        return (
            text_loss,
            text_accuracy,
            image_loss,
            image_accuracy,
            multimodal_loss,
            multimodal_accuracy,
            total_loss,
        )

    def training_step(self, batch, batch_idx):
        (
            text_loss,
            text_accuracy,
            image_loss,
            image_accuracy,
            multimodal_loss,
            multimodal_accuracy,
            total_loss,
        ) = self.common_step(batch, batch_idx)

        self.log_dict(
            {
                "training_text_loss": text_loss,
                "training_text_accuracy": text_accuracy,
                "training_image_loss": image_loss,
                "training_image_accuracy": image_accuracy,
                "training_multimodal_loss": multimodal_loss,
                "training_multimodal_accuracy": multimodal_accuracy,
                "training_total_loss": total_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["input_ids"].shape[0],
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        (
            text_loss,
            text_accuracy,
            image_loss,
            image_accuracy,
            multimodal_loss,
            multimodal_accuracy,
            total_loss,
        ) = self.common_step(batch, batch_idx)

        self.log_dict(
            {
                "validation_text_loss": text_loss,
                "validation_text_accuracy": text_accuracy,
                "validation_image_loss": image_loss,
                "validation_image_accuracy": image_accuracy,
                "validation_multimodal_loss": multimodal_loss,
                "validation_multimodal_accuracy": multimodal_accuracy,
                "validation_total_loss": total_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["input_ids"].shape[0],
        )
        return total_loss

    def test_step(self, batch, batch_idx):
        (
            text_loss,
            text_accuracy,
            image_loss,
            image_accuracy,
            multimodal_loss,
            multimodal_accuracy,
            total_loss,
        ) = self.common_step(batch, batch_idx)

        self.log_dict(
            {
                "test_text_loss": text_loss,
                "test_text_accuracy": text_accuracy,
                "test_image_loss": image_loss,
                "test_image_accuracy": image_accuracy,
                "test_multimodal_loss": multimodal_loss,
                "test_multimodal_accuracy": multimodal_accuracy,
                "test_total_loss": total_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["input_ids"].shape[0],
        )

        return total_loss

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]

        text_logits, image_logits, multimodal_logits = self(
            input_ids, attention_mask, pixel_values
        )

        text_predictions = text_logits.argmax(-1)
        text_probabilities = torch.softmax(text_logits, dim=-1)
        text_confidences = torch.max(text_probabilities, dim=-1).values

        image_predictions = image_logits.argmax(-1)
        image_probabilities = torch.softmax(image_logits, dim=-1)
        image_confidences = torch.max(image_probabilities, dim=-1).values

        multimodal_predictions = multimodal_logits.argmax(-1)
        multimodal_probabilities = torch.softmax(multimodal_logits, dim=-1)
        multimodal_confidences = torch.max(
            multimodal_probabilities, dim=-1
        ).values

        if "labels" in batch.keys():
            labels = batch["labels"]
            mask = labels != -100  # Mask for non-ignore indices
            masked_labels = labels[mask]

            masked_text_predictions = text_predictions[mask]
            masked_text_confidences = text_confidences[mask]

            masked_image_predictions = image_predictions[mask]
            masked_image_confidences = image_confidences[mask]

            masked_multimodal_predictions = multimodal_predictions[mask]
            masked_multimodal_confidences = multimodal_confidences[mask]

            return {
                "text_predictions": masked_text_predictions.cpu().numpy(),
                "text_confidences": masked_text_confidences.cpu().numpy(),
                "image_predictions": masked_image_predictions.cpu().numpy(),
                "image_confidences": masked_image_confidences.cpu().numpy(),
                "multimodal_predictions": masked_multimodal_predictions.cpu().numpy(),
                "multimodal_confidences": masked_multimodal_confidences.cpu().numpy(),
                "labels": masked_labels.cpu().numpy(),
            }
        else:
            return {
                "text_predictions": text_predictions.cpu().numpy(),
                "text_confidences": text_confidences.cpu().numpy(),
                "image_predictions": image_predictions.cpu().numpy(),
                "image_confidences": image_confidences.cpu().numpy(),
                "multimodal_predictions": multimodal_predictions.cpu().numpy(),
                "multimodal_confidences": multimodal_confidences.cpu().numpy(),
            }

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=self.learning_rate)
