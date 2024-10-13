import yaml
from easydict import EasyDict
from datasets.utils.file_utils import DownloadConfig
from datasets import load_dataset


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return EasyDict(config)


if __name__ == "__main__":
    config_path = "./config/config_example.yaml"
    config = load_config(config_path)

    params = {}
    params["path"] = config.DATASET_BUILDER_PATH
    params["cache_dir"] = config.CACHE_DIR
    params["data_dir"] = config.DATA_DIR
    params["download_config"] = DownloadConfig(config.DOWNLOAD_CFG_PATH)
    params["name"] = config.DATA_NAME
    params["streaming"] = True

    ds = load_dataset(**params)
    train_stream = ds["train"]
    shuffled_train_stream = ds["train"].shuffle(seed=42)

    print(0)
