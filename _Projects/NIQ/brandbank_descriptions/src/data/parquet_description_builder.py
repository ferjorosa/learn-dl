import pandas as pd
from datasets import (
    DatasetInfo,
    BuilderConfig,
    GeneratorBasedBuilder,
    SplitGenerator,
    Value,
    Features,
    Sequence,
)
from datasets.download import DownloadManager
from typing import List, Dict, Any, Generator, Tuple
from pathlib import Path


class ParquetDescriptionBuilderConfig(BuilderConfig):
    def __init__(self, **kwargs):
        super(ParquetDescriptionBuilderConfig, self).__init__(**kwargs)


class ParquetDescriptionBuilder(GeneratorBasedBuilder):
    def _info(self) -> DatasetInfo:
        dict_features = {
            "PRODUCT_ID": Value("string"),
            "PRODUCT_GROUP": Value("string"),
            "MODULE": Value("string"),
            "GLOBAL_PACKAGING": Value("string"),
            "DESCRIPTIONS": Sequence(Value("string")),
        }

        return DatasetInfo(
            description="Parquet description dataset",
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
        chars_parquet = filepath / "chars.parquet"
        descriptions_parquet = filepath / "descriptions.parquet"

        # Load DataFrames
        df_chars = pd.read_parquet(chars_parquet)
        df_descriptions = pd.read_parquet(descriptions_parquet)

        # Set 'PRODUCT_ID' as the index in df_chars for faster iteration
        df_chars = df_chars.set_index("PRODUCT_ID")

        for product_id in df_chars.index:
            char_info = df_chars.loc[product_id]
            descriptions = df_descriptions[
                df_descriptions["PRODUCT_ID"] == product_id
            ]["DESCRIPTION"].tolist()

            yield product_id, {
                "PRODUCT_ID": product_id,
                "PRODUCT_GROUP": char_info["PRODUCT_GROUP"],
                "MODULE": char_info["MODULE"],
                "GLOBAL_PACKAGING": char_info["GLOBAL_PACKAGING"],
                "DESCRIPTIONS": descriptions,
            }
