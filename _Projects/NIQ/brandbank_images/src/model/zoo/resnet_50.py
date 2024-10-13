import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor, AutoConfig


class Resnet50Model(nn.Module):

    model_name = "microsoft/resnet-50"

    @classmethod
    def image_processor(cls):
        return AutoImageProcessor.from_pretrained(cls.model_name)

    def __init__(self, num_labels, pretrained, pooling=True):
        super().__init__()

        if pretrained:
            self.base_model = AutoModel.from_pretrained(self.model_name)
        else:
            config = AutoConfig.from_pretrained(self.model_name)
            self.base_model = AutoModel.from_config(config)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.base_model.config.hidden_sizes[-1], num_labels),
        )
        if pooling is False:
            raise ValueError(
                "There is no CLS token in Resnets, pooling is mandatory"
            )

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values=pixel_values)

        # Apply the classifier layer
        # (Flatten after average pooling)
        logits = self.classifier(outputs.pooler_output)

        return logits
