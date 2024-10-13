from src.model.zoo import (
    VisionTransformerModel,
    Resnet50Model,
    FlavaImageEncoderModel,
)

catalog = {
    "VISION_TRANSFORMER": VisionTransformerModel,
    "RESNET_50": Resnet50Model,
    "FLAVA_IMAGE_ENCODER": FlavaImageEncoderModel,
}


def get_model(model_name, num_labels, pretrained, pooling):
    if model_name not in catalog:
        raise ValueError(f"{model_name} is not present in the model catalog")
    # Call the appropriate function from the catalog
    return catalog[model_name](num_labels, pretrained, pooling)


def get_image_processor(model_name):
    if model_name not in catalog:
        raise ValueError(
            f"{model_name} is not present in the image processor catalog"
        )
    # Call the appropriate function from the catalog
    return catalog[model_name].image_processor()
