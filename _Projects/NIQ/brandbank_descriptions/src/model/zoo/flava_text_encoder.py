import torch.nn as nn
from transformers import FlavaTextModel, FlavaTextConfig, AutoTokenizer


class FlavaTextEncoderModel(nn.Module):

    model_name = "facebook/flava-full"

    @classmethod
    def tokenizer(cls):
        return AutoTokenizer.from_pretrained(cls.model_name)

    def __init__(self, num_labels, pretrained, pooling=True):
        super().__init__()

        if pretrained:
            self.base_model = FlavaTextModel.from_pretrained(self.model_name)
        else:
            config = FlavaTextConfig.from_pretrained(self.model_name)
            self.base_model = FlavaTextModel.from_config(config)

        self.pooling = pooling
        self.classifier = nn.Linear(
            self.base_model.config.hidden_size, num_labels
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Apply the classifier layer
        if self.pooling:
            logits = self.classifier(
                outputs.pooler_output
            )  # Average 2D pooling
        else:
            logits = self.classifier(
                outputs.last_hidden_state[:, 0]
            )  # Use the [CLS] token representation

        return logits
