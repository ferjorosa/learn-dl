import torch.nn as nn
from transformers import (
    FlavaModel,
    FlavaConfig,
    AutoTokenizer,
    AutoImageProcessor,
)


class FlavaMultimodalEncoder3LossesModel(nn.Module):

    model_name = "facebook/flava-full"

    @classmethod
    def tokenizer(cls):
        return AutoTokenizer.from_pretrained(cls.model_name)

    @classmethod
    def image_processor(cls):
        return AutoImageProcessor.from_pretrained(cls.model_name)

    def __init__(self, num_labels, pretrained, pooling=True):
        super().__init__()

        if pretrained:
            self.base_model = FlavaModel.from_pretrained(self.model_name)
        else:
            config = FlavaConfig.from_pretrained(self.model_name)
            self.base_model = FlavaModel.from_config(config)

        self.pooling = pooling
        self.text_classifier = nn.Linear(
            self.base_model.config.hidden_size, num_labels
        )
        self.image_classifier = nn.Linear(
            self.base_model.config.hidden_size, num_labels
        )
        self.multimodal_classifier = nn.Linear(
            self.base_model.config.hidden_size, num_labels
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        # Apply the classifier layer
        if self.pooling:
            text_logits = self.text_classifier(
                outputs.text_output.pooler_output
            )
            image_logits = self.image_classifier(
                outputs.image_output.pooler_output
            )
            multimodal_logits = self.multimodal_classifier(
                outputs.multimodal_output.pooler_output
            )
        else:
            text_logits = self.text_classifier(
                outputs.text_output.last_hidden_state[:, 0]
            )
            image_logits = self.image_classifier(
                outputs.image_output.last_hidden_state[:, 0]
            )
            multimodal_logits = self.multimodal_classifier(
                outputs.multimodal_output.last_hidden_state[:, 0]
            )
        return text_logits, image_logits, multimodal_logits
