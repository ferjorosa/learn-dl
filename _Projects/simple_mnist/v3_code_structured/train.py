import pytorch_lightning as pl

from model import FFNNLightningModule
from dataset import MnistDataModule
import config

if __name__ == "__main__":

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
        accelerator=config.COMPUTE_ACCELERATOR,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
    )

    trainer.fit(model, data_module)
    trainer.validate(model, data_module)
    trainer.test(model, data_module)
