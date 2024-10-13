import torch
import pytorch_lightning as pl

from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class MnistDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_size,
        batch_size,
        val_split_size,
        num_workers,
        random_seed=42,
    ):
        super().__init__()
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_split_size = val_split_size
        self.num_workers = num_workers
        self.random_seed = random_seed

    def _train_transforms(self, examples):
        examples["pixel_values"] = [
            self.transformations(image) for image in examples["image"]
        ]
        return examples

    def _val_transforms(self, examples):
        examples["pixel_values"] = [
            self.transformations(image) for image in examples["image"]
        ]
        return examples

    def _collate_batch(self, examples):
        pixel_values = torch.stack(
            [example["pixel_values"].view(-1) for example in examples]
        )
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    # Multiple GPU
    # Since we are using HuggingFace, we only going to use the setup() method, because
    # it will download and split the data with a single call
    def setup(self, stage):
        # Load data
        train_ds, test_ds = load_dataset(
            "mnist", split=["train[:5000]", "test[:1000]"]
        )

        # Split training data into training and validation sets
        splits = train_ds.train_test_split(
            test_size=self.val_split_size, seed=self.random_seed
        )
        self.train_ds = splits["train"]
        self.val_ds = splits["test"]
        self.test_ds = test_ds

        # Transformations
        self.transformations = transforms.Compose(
            [
                transforms.Resize(
                    self.image_size
                ),  # not needed because all images already have the appropriate size
                transforms.ToTensor(),
            ]
        )

        self.train_ds.set_transform(self._train_transforms)
        self.val_ds.set_transform(self._val_transforms)
        self.test_ds.set_transform(self._val_transforms)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_batch,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_batch,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_batch,
            num_workers=self.num_workers,
        )
