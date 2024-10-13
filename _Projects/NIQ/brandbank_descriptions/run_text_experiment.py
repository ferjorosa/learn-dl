import yaml
import pandas as pd
import pytorch_lightning as pl
import pyarrow.parquet as pq
from easydict import EasyDict
from pathlib import Path
from datasets.utils.file_utils import DownloadConfig
from pytorch_lightning import loggers

from src.data import DescriptionDataModule
from src.model import DescriptionModelModule

from src.callbacks import SaveConfigCallback, TrainingTimeCallback
from src.utils.metrics import calculate_metrics, top_10_confusions
from src.utils.utils import map_col


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return EasyDict(config)


def create_label2id_maps(config, label_name, ignore_labels):
    train_df_chars_file = (
        Path(config.DATA_DIR) / f"{config.DATA_NAME}_train" / "chars.parquet"
    )
    val_df_chars_file = (
        Path(config.DATA_DIR) / f"{config.DATA_NAME}_val" / "chars.parquet"
    )
    test_df_chars_file = (
        Path(config.DATA_DIR) / f"{config.DATA_NAME}_test" / "chars.parquet"
    )

    df_chars_train = pd.read_parquet(train_df_chars_file)
    df_chars_val = pd.read_parquet(val_df_chars_file)
    df_chars_test = pd.read_parquet(test_df_chars_file)

    unique_labels = (
        set(df_chars_train[label_name].unique())
        | set(df_chars_val[label_name].unique())
        | set(df_chars_test[label_name].unique())
    )

    # Manually setting IDs for ignored labels to -100
    for label in ignore_labels:
        unique_labels.discard(label)
        unique_labels.add(label)

    id2label = {id: label for id, label in enumerate(unique_labels)}
    label2id = {label: id for id, label in id2label.items()}

    # Assigning -100 to all ignored labels
    for label in ignore_labels:
        label_id = label2id.get(label)
        if label_id is not None:
            label2id[label] = -100
            id2label[-100] = label

    return label2id, id2label


def estimate_data_length(config):
    train_df_chars_file = (
        Path(config.DATA_DIR) / f"{config.DATA_NAME}_train" / "chars.parquet"
    )
    val_df_chars_file = (
        Path(config.DATA_DIR) / f"{config.DATA_NAME}_val" / "chars.parquet"
    )
    test_df_chars_file = (
        Path(config.DATA_DIR) / f"{config.DATA_NAME}_test" / "chars.parquet"
    )

    train_data_length = pq.ParquetFile(train_df_chars_file).metadata.num_rows
    val_data_length = pq.ParquetFile(val_df_chars_file).metadata.num_rows
    test_data_length = pq.ParquetFile(test_df_chars_file).metadata.num_rows

    return train_data_length, val_data_length, test_data_length


def flatten_predictions(predictions):
    flattened_predictions = {}
    for entry in predictions:
        for key, value in entry.items():
            if key not in flattened_predictions:
                flattened_predictions[key] = []
            flattened_predictions[key].extend(value)
    return flattened_predictions


def train_and_test_model(config_path):
    # Load YAML configuration
    config = load_config(config_path)

    # HuggingFace dataset parameters
    params = {}
    params["path"] = config.DATASET_BUILDER_PATH
    params["cache_dir"] = config.CACHE_DIR
    params["data_dir"] = config.DATA_DIR
    params["download_config"] = DownloadConfig(config.DOWNLOAD_CFG_PATH)
    params["name"] = config.DATA_NAME
    params["streaming"] = True

    # Estimate data length for the semi-iterable datasets
    (
        train_data_length,
        val_data_length,
        test_data_length,
    ) = estimate_data_length(config)

    # Create label2id mappings
    label_name = config.LABEL_NAME
    ignore_labels = config.IGNORE_LABELS
    label2id, id2label = create_label2id_maps(
        config, label_name, ignore_labels
    )

    # Logger
    logger = loggers.CSVLogger(
        config.LOGS_DIR,
        name=f"{config.DATA_NAME}/{config.LABEL_NAME}/{config.MODEL_NAME}",
    )

    # Lightning modules
    model_name = config.MODEL_NAME
    pretrained = True
    pooling = True

    data_module = DescriptionDataModule(
        params=params,
        model_name=model_name,
        label_colname=label_name,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        label2id=label2id,
        max_seq_length=config.MAX_SEQ_LENGTH,
        iterable=False,
        train_data_length=train_data_length,
        val_data_length=val_data_length,
        test_data_length=test_data_length,
        random_seed=42,
    )

    model_module = DescriptionModelModule(
        model_name=model_name,
        label2id=label2id,
        learning_rate=float(config.LEARNING_RATE),
        pretrained=pretrained,
        pooling=pooling,
    )

    trainer = pl.Trainer(
        devices=config.COMPUTE_DEVICES,
        accelerator=config.COMPUTE_ACCELERATOR,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        logger=logger,
        callbacks=[
            SaveConfigCallback(config),
            TrainingTimeCallback(),
        ],
    )

    # Model training and testing
    trainer.fit(model_module, data_module)
    # trainer.validate(model_module, data_module)
    trainer.test(model_module, data_module)

    # Generate predictions (test dataset)
    predictions = trainer.predict(model_module, data_module)
    predictions = flatten_predictions(predictions)
    df_predictions = pd.DataFrame(predictions)
    id2label_series = pd.Series(id2label)
    df_predictions["labels"] = map_col(
        df_predictions["labels"], id2label_series
    )
    df_predictions["predictions"] = map_col(
        df_predictions["predictions"], id2label_series
    )

    # Generate prediction metrics
    df_prediction_metrics = calculate_metrics(
        df=df_predictions,
        label_colname="labels",
        prediction_colname="predictions",
    )

    df_top_10_confusions = top_10_confusions(
        df=df_predictions,
        label_colname="labels",
        prediction_colname="predictions",
    )

    # Export predictions and its associated metrics
    checkpoint_dir = Path(trainer.checkpoint_callback.dirpath)
    save_dir = checkpoint_dir.parent
    df_predictions.to_csv(save_dir / "predictions.csv", index=False)
    df_prediction_metrics.to_csv(
        save_dir / "predictions_metrics.csv", index=False
    )
    df_top_10_confusions.to_csv(
        save_dir / "predictions_top_10_confusions.csv", index=False
    )


if __name__ == "__main__":
    config_path = "./config/config_example.yaml"
    train_and_test_model(config_path)
