from src.model.zoo import (
    BertBaseModel,
    DistilbertBaseModel,
    FlavaTextEncoderModel,
)

catalog = {
    "BERT_BASE": BertBaseModel,
    "DISTILBERT_BASE": DistilbertBaseModel,
    "FLAVA_TEXT_ENCODER": FlavaTextEncoderModel,
}


def get_model(model_name, num_labels, pretrained, pooling):
    if model_name not in catalog:
        raise ValueError(f"{model_name} is not present in the catalog")
    # Call the appropriate function from the catalog
    return catalog[model_name](num_labels, pretrained, pooling)


def get_tokenizer(model_name):
    if model_name not in catalog:
        raise ValueError(f"{model_name} is not present in the catalog")
    # Call the appropriate function from the catalog
    return catalog[model_name].tokenizer()
