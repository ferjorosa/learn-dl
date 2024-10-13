The objective of this mini-project is to have a better understanding of Brandbank data and how we can improve downstream classification tasks such as PRODUCT_GROUP and GLOBAL_PACKAGKING prediction. For reference, We have done some experiments with FLAVA's image encoder for predicting PRODUCT_GROUP and the results were quite bad.

We want to see if we are able to accurately predict this kind of characteristics using only image information and compare the results with those of text/multimodal Transformers.

For these tasks, we are going to use the data files previously generated in Parquet + HDF5 format.

We are going to use PyTorch Lightning as the DL framework

#### Warning

`Could not find image processor class in the image processor config or the model config. Loading based on pattern matching with the model's feature extractor configuration.`

https://discuss.huggingface.co/t/error-finding-processors-image-class-loading-based-on-pattern-matching-with-feature-extractor/31890/9