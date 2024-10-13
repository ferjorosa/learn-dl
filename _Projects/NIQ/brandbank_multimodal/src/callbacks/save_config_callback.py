import pytorch_lightning as pl
import yaml
from pathlib import Path


class SaveConfigCallback(pl.Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config_file_name = "config.yaml"

    def on_train_start(self, trainer, pl_module):
        # Get the checkpoint directory using pathlib
        checkpoint_dir = Path(trainer.checkpoint_callback.dirpath)

        save_dir = checkpoint_dir.parent

        # Construct the full path for the config file
        config_file_path = save_dir / self.config_file_name

        # Save the configuration to a YAML file
        with open(config_file_path, "w") as yaml_file:
            yaml.dump(vars(self.config), yaml_file)
