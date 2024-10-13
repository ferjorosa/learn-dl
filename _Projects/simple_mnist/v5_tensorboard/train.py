import torch
import pytorch_lightning as pl

from model import FFNNLightningModule
from dataset import MnistDataModule
from callbacks import MyPrintingCallback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
import config

if __name__ == "__main__":

    logger = TensorBoardLogger(
        "simple_mnist/v5_tensorboard/tb_logs", name="mnist_model_v0"
    )

    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "simple_mnist/v5_tensorboard/tb_logs/profiler0"
        ),
        schedule=torch.profiler.schedule(
            skip_first=10, wait=1, warmup=1, active=20
        ),
    )

    data_module = MnistDataModule(
        config.IMAGE_SIZE,
        config.BATCH_SIZE,
        config.VAL_SPLIT_SIZE,
        config.NUM_CPUS,
    )

    model = FFNNLightningModule(
        input_size=config.INPUT_SIZE,
        num_classes=config.NUM_CLASSES,
        learning_rate=config.LEARNING_RATE,
    )

    trainer = pl.Trainer(
        devices=config.COMPUTE_DEVICES,
        # accelerator=config.COMPUTE_ACCELERATOR,
        min_epochs=1,
        max_epochs=1000,
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback()],
        logger=logger,
        profiler=profiler,
    )

    trainer.fit(model, data_module)
    # trainer.validate(model, data_module)
    # trainer.test(model, data_module)
