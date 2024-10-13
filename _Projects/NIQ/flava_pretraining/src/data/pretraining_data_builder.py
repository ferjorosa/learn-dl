import numpy as np
import pandas as pd
import h5py
from datasets import (
    DatasetInfo,
    BuilderConfig,
    GeneratorBasedBuilder,
    SplitGenerator,
    Value,
    Features,
    Sequence,
    Array3D,
)
from datasets.download import DownloadManager
from typing import List, Dict, Any, Generator, Tuple
from pathlib import Path


class PretrainingDataBuilderConfig(BuilderConfig):
    def __init__(self, **kwargs):
        super(PretrainingDataBuilderConfig, self).__init__(**kwargs)


class PretrainingDataBuilder(GeneratorBasedBuilder):
    def _info(self) -> DatasetInfo:
        dict_features = {
            "PRODUCT_ID": Value("string"),
            "DESCRIPTIONS": Sequence(Value("string")),
            "NEGATIVE_DESCRIPTIONS": Sequence(Value("string")),
            "IMAGES": Sequence(Array3D(dtype=np.uint8, shape=(500, 500, 3))),
        }

        return DatasetInfo(
            description="Pretraining dataset",
            features=Features(dict_features),
            supervised_keys=None,
            homepage="https://example.com/dataset",  # Update with the actual dataset URL
        )

    def _split_generators(
        self, dl_manager: DownloadManager
    ) -> List[SplitGenerator]:
        eliza_dataset_name = Path(self.config.data_dir).parts[-2]
        # Create SplitGenerator instances for each data split
        split_generators = [
            SplitGenerator(
                name="train",
                gen_kwargs={
                    "filepath": Path(self.config.data_dir)
                    / f"{eliza_dataset_name}_train",
                    "split_name": "train",
                },
            ),
            SplitGenerator(
                name="val",
                gen_kwargs={
                    "filepath": Path(self.config.data_dir)
                    / f"{eliza_dataset_name}_val",
                    "split_name": "val",
                },
            ),
            SplitGenerator(
                name="test",
                gen_kwargs={
                    "filepath": Path(self.config.data_dir)
                    / f"{eliza_dataset_name}_test",
                    "split_name": "test",
                },
            ),
        ]

        return split_generators

    def _generate_examples(
        self, filepath: str, split_name: str
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        # Relevant files
        filepath = Path(filepath)
        descriptions_parquet = filepath / "descriptions.parquet"
        images_h5 = filepath / "images.hdf5"

        # Load DataFrames
        df_descriptions = pd.read_parquet(descriptions_parquet)

        # Set 'PRODUCT_ID' as the index in df_chars for faster iteration
        df_descriptions = df_descriptions.set_index("PRODUCT_ID")

        # Open the HDF5 image file
        with h5py.File(images_h5, "r") as hdf5_file:
            for product_id in df_descriptions.index:
                product_id_str = str(product_id)
                if product_id_str not in hdf5_file:
                    raise KeyError(
                        f"PRODUCT_ID {product_id} not found in HDF5 file."
                    )
                else:
                    image_group = hdf5_file[product_id_str]
                    images = []

                    for dataset_name in image_group.keys():
                        dataset = image_group[dataset_name]
                        images.append(
                            dataset[:]
                        )  # Append the dataset's data to the list

                        # If it is a single string, we wrap the string with a list,
                        # If there are multiple strings, we can call "tolist()"
                        descriptions = df_descriptions[
                            df_descriptions["PRODUCT_ID"] == product_id
                        ]["DESCRIPTION"].tolist()

                        descriptions = (
                            [descriptions]
                            if isinstance(descriptions, str)
                            else descriptions.tolist()
                        )

                        # Sample negative descriptions not aligned with the PRODUCT_ID
                        negative_descriptions = (
                            df_descriptions[
                                df_descriptions.index != product_id
                            ]
                            .sample(len(descriptions))["DESCRIPTION"]
                            .tolist()
                        )

                    # We sample a number of negative descriptions (i.e., not aligned with the PRODUCT_ID)
                    # equal to the number of descriptions currently associated to the PRODUCT_ID in question
                    negative_descriptions = (
                        df_descriptions[df_descriptions.index != product_id]
                        .sample(len(descriptions))["DESCRIPTION"]
                        .tolist()
                    )

                    yield product_id, {
                        "PRODUCT_ID": product_id,
                        "DESCRIPTIONS": descriptions,
                        "NEGATIVE_DESCRIPTIONS": negative_descriptions,
                        "IMAGES": images,
                    }
