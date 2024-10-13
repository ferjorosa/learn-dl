import torch
import pytorch_lightning as pl

from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomApply,
    RandomRotation,
    Resize,
    ToTensor,
)
from PIL import Image

from src.data.iterable_image_dataset import IterableImageDataset
from src.data.semi_iterable_image_dataset import SemiIterableImageDataset
from src.model import image_model_catalog as catalog


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        params,
        model_name,
        label_colname,
        batch_size,
        num_workers,
        label2id,
        iterable=True,
        train_data_length=None,
        val_data_length=None,
        test_data_length=None,
        random_seed=42,
    ):
        super().__init__()
        self.params = params
        self.model_name = model_name
        self.label_colname = label_colname
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label2id = label2id
        self.random_seed = random_seed
        self.iterable = iterable
        self.train_data_length = train_data_length
        self.val_data_length = val_data_length
        self.test_data_length = test_data_length

        if iterable is False and train_data_length is None:
            raise ValueError(
                "If iterable is False, train_data_length needs to be passed"
            )

        if iterable is False and val_data_length is None:
            raise ValueError(
                "If iterable is False, val_data_length needs to be passed"
            )

        if iterable is False and test_data_length is None:
            raise ValueError(
                "If iterable is False, test_data_length needs to be passed"
            )

        self.images_colname = "IMAGES"  # to-be-moved
        self.id_colname = "PRODUCT_ID"  # to-be-moved

    # Single GPU
    # Dataset downloading process should go here
    def prepare_data(self):
        pass

    # Multiple GPU
    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup
    def setup(self, stage: str):
        # Load HuggingFace dataset
        ds = load_dataset(**self.params)

        # Image transformations
        processor = catalog.get_image_processor(self.model_name)
        image_mean = processor.image_mean
        image_std = processor.image_std
        # This is a bit weird way of taking the first key in the "size" dicitonary.
        # We do it this way because different image processors have different things here.
        # For example, the processor for ViT has two keys (height, width), but the resnet50
        # has a single key called "shortest_edge". Since we are going to work with square
        # images, we can take the first key in either case
        size = processor.size[list(processor.size.keys())[0]]

        normalize = Normalize(mean=image_mean, std=image_std)
        self.train_transforms = Compose(
            [
                # RandomResizedCrop(size),
                # RandomHorizontalFlip(),
                RandomResizedCrop(size=size, scale=(0.4, 1.0)),
                RandomApply(
                    torch.nn.ModuleList([RandomRotation(degrees=(180, 180))]),
                    p=0.1,
                ),
                ToTensor(),
                normalize,
            ]
        )
        self.val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

        # Create datasets
        if self.iterable:
            self.train_ds = self._create_iterable_dataset(ds, "train")
            self.val_ds = self._create_iterable_dataset(ds, "val")
            self.test_ds = self._create_iterable_dataset(ds, "test")
            self.predict_ds = self._create_iterable_dataset(ds, "test")
        else:
            self.train_ds = self._create_semi_iterable_dataset(
                ds, "train", self.train_data_length
            )
            self.val_ds = self._create_semi_iterable_dataset(
                ds, "val", self.val_data_length
            )
            self.test_ds = self._create_semi_iterable_dataset(
                ds, "test", self.test_data_length
            )
            self.predict_ds = self._create_semi_iterable_dataset(
                ds, "test", self.test_data_length
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            collate_fn=lambda batch: self._collate_batch(
                batch, self.train_transforms
            ),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            collate_fn=lambda batch: self._collate_batch(
                batch, self.val_transforms
            ),
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            collate_fn=lambda batch: self._collate_batch(
                batch, self.val_transforms
            ),
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_ds,
            batch_size=self.batch_size,
            collate_fn=lambda batch: self._collate_batch(
                batch, self.val_transforms
            ),
            num_workers=self.num_workers,
        )

    def _create_iterable_dataset(self, ds, split):
        return IterableImageDataset(
            data=ds[split],
            images_colname=self.images_colname,
            label_colname=self.label_colname,
            id_colname=self.id_colname,
        )

    def _create_semi_iterable_dataset(self, ds, split, data_length):
        return SemiIterableImageDataset(
            data=ds[split],
            data_length=data_length,
            images_colname=self.images_colname,
            label_colname=self.label_colname,
            id_colname=self.id_colname,
        )

    def _collate_batch(self, batch, image_transforms):
        ids, labels, images = zip(*batch)

        # Apply transformations to the images and stack the resulting tensors
        transformed_images = [
            image_transforms(Image.fromarray(image)) for image in images
        ]
        images = torch.stack(transformed_images, dim=0)

        # Convert labels to a tensor of IDs
        label_ids = [self.label2id[label] for label in labels]
        label_ids = torch.tensor(label_ids)

        # Create a list of IDs
        id_list = list(ids)

        # Create a dictionary to store batched data
        batched_data = {
            "pixel_values": images,
            "PRODUCT_ID": id_list,
            "labels": label_ids,
        }

        return batched_data
