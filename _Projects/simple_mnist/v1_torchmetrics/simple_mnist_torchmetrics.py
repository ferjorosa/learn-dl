import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from datasets import load_dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import Metric


def train_transforms(examples):
    examples["pixel_values"] = [
        transformations(image) for image in examples["image"]
    ]
    return examples


def val_transforms(examples):
    examples["pixel_values"] = [
        transformations(image) for image in examples["image"]
    ]
    return examples


def collate_batch(examples):
    pixel_values = torch.stack(
        [example["pixel_values"].view(-1) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "correct", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()


class FFNNLightningModule(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes
        )
        # Define our own metric, to show how to do it:
        self.my_accuracy = MyAccuracy()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        # This metric estimation is very slow, we should use a Profiler
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        my_accuracy = self.my_accuracy(scores, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
                "train_my_accuracy": my_accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def _common_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        y = batch["labels"]

        # x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


image_size = (28, 28)
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 32
num_epochs = 3
num_cpus = 1  # multiprocessing.cpu_count() -1

# Load mnist from HuggingFace Hub (only a small portion for demonstration purposes)
train_ds, test_ds = load_dataset(
    "mnist", split=["train[:5000]", "test[:1000]"]
)

# Split training data into training and validation sets
splits = train_ds.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
val_ds = splits["test"]

# Transformations
transformations = transforms.Compose(
    [
        transforms.Resize(
            image_size
        ),  # not needed because all images already have the appropriate size
        transforms.ToTensor(),
    ]
)

train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch,
    num_workers=num_cpus,
)
val_loader = DataLoader(
    dataset=val_ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch,
    num_workers=num_cpus,
)
test_loader = DataLoader(
    dataset=test_ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch,
    num_workers=num_cpus,
)

model = FFNNLightningModule(input_size=input_size, num_classes=num_classes)

trainer = pl.Trainer(
    devices=1,
    accelerator="gpu",
    min_epochs=1,
    max_epochs=3,
    precision="16-mixed",
)

trainer.fit(model, train_loader, val_loader)
