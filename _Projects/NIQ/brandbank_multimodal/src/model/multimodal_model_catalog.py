from src.model.zoo import (
    FlavaMultimodalEncoderModel,
    FlavaMultimodalEncoder3LossesModel,
)

catalog = {
    "FLAVA_MULTIMODAL_ENCODER": FlavaMultimodalEncoderModel,
    "FLAVA_MULTIMODAL_ENCODER_3_LOSSES": FlavaMultimodalEncoder3LossesModel,
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


def get_image_processor(model_name):
    if model_name not in catalog:
        raise ValueError(
            f"{model_name} is not present in the image processor catalog"
        )
    # Call the appropriate function from the catalog
    return catalog[model_name].image_processor()
