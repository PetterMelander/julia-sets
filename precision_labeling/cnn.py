import random

import cv2
import lightning as L
import numpy as np
import torch
import torchmetrics
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import MLFlowLogger
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose

torch.set_float32_matmul_precision("high")


class SimpleCNN(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.normalize = False
        self.save_hyperparameters()

        # very big
        # self.layers = nn.Sequential(
        #     nn.Conv2d(1, 32, 5, 2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(256, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )

        # medium
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # big
        # self.layers = nn.Sequential(
        #     nn.Conv2d(1, 32, 5, 2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )

        # small
        # self.layers = nn.Sequential(
        #     nn.Conv2d(1, 32, 3, 2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout()
        self.head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy("binary")
        self.recall = torchmetrics.Recall("binary")
        self.specificity = torchmetrics.Specificity("binary")
        self.aucroc = torchmetrics.AUROC("binary")

        self.false_positives = []
        self.false_negatives = []

    def forward(self, input_img: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            with torch.no_grad():
                input_img = (
                    input_img / 2500.0
                )  # preprocess inside forward for easy onnx export
        x = self.layers(input_img)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, labels.unsqueeze(1).to(torch.float32))
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.unsqueeze(1)
        logits = self.forward(inputs)
        val_loss = self.loss_fn(logits, labels.to(torch.float32))
        self.log("val_loss", val_loss)

        self.accuracy(logits, labels)
        self.log("val_accuracy", self.accuracy, on_epoch=True)

        self.recall(logits, labels)
        self.log("val_sensitivity", self.recall, on_epoch=True)

        self.specificity(logits, labels)
        self.log("val_specificity", self.specificity, on_epoch=True)

        self.aucroc(logits, labels)
        self.log("val_aucroc", self.aucroc, on_epoch=True)

        preds = (torch.sigmoid(logits) > 0.5).long()
        false_positives = (preds == 1) & (labels == 0)
        false_negatives = (preds == 0) & (labels == 1)

        if false_negatives.any():
            for i in range(len(false_negatives)):
                if false_negatives[i] == 1:
                    self.false_negatives.append(inputs[i].detach().cpu().numpy())

        if false_positives.any():
            for i in range(len(false_positives)):
                if false_positives[i] == 1:
                    self.false_positives.append(inputs[i].detach().cpu().numpy())

    def on_validation_epoch_end(self):
        if self.false_negatives:
            example = random.choice(self.false_negatives)
            example = np.permute_dims(example, (1, 2, 0))
            example = (example - example.min()) / (example.max() - example.min())
            mlflow_client = self.logger.experiment
            run_id = self.logger.run_id
            caption = f"Epoch_{self.current_epoch}_false_negative"
            mlflow_client.log_image(
                run_id=run_id,
                image=example,
                artifact_file=f"misclassified/{caption}.png",
            )
            self.false_negatives.clear()

        if self.false_positives:
            example = random.choice(self.false_positives)
            example = np.permute_dims(example, (1, 2, 0))
            example = (example - example.min()) / (example.max() - example.min())
            mlflow_client = self.logger.experiment
            run_id = self.logger.run_id
            caption = f"Epoch_{self.current_epoch}_false_positive"
            mlflow_client.log_image(
                run_id=run_id,
                image=example,
                artifact_file=f"misclassified/{caption}.png",
            )
            self.false_positives.clear()

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, nesterov=True)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, pct_start=0.1, total_steps=500)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return [optimizer], [scheduler]


def pfm_loader(path: str) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.expand_dims(img, axis=2)
    return img.astype(np.float32)


class ScalingTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor]:
        return img / 2500


class CustomTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, transform: Compose) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.dataset[idx]
        x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.dataset)


def main() -> None:
    torch.manual_seed(41)
    model = SimpleCNN()

    train_ds = ImageFolder(
        "datasets/v8/train",
        loader=pfm_loader,
        is_valid_file=lambda x: x.endswith(".pfm"),
        transform=Compose(
            [transforms.ToTensor()],
        ),
    )

    val_ds = ImageFolder(
        "datasets/v8/val",
        loader=pfm_loader,
        is_valid_file=lambda x: x.endswith(".pfm"),
        transform=Compose(
            [transforms.ToTensor()],
        ),
    )
    # train_ds, val_ds = torch.utils.data.random_split(ds, [0.9, 0.1])
    train_transforms = Compose(
        [
            ScalingTransform(),
            # transforms.RandomResizedCrop(size=224, scale=(0.25, 1.0)),
            # transforms.RandomInvert(),
            # transforms.RandomSolarize(0.5),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomAffine(degrees=180, shear=(-22, 22, -22, 22))
            transforms.RandomRotation((0, 360)),
        ]
    )
    val_transforms = Compose([ScalingTransform()])
    train_ds = CustomTransformDataset(train_ds, train_transforms)
    val_ds = CustomTransformDataset(val_ds, val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=16,
        drop_last=True,
        prefetch_factor=16,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=128,
        num_workers=16,
        pin_memory=True,
    )

    mlf_logger = MLFlowLogger()

    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(monitor="val_loss", patience=500, mode="min")
    model_checkpoint = ModelCheckpoint(monitor="val_accuracy", save_top_k=5, mode="max")
    trainer = L.Trainer(
        # precision="bf16-mixed",
        benchmark=True,
        logger=mlf_logger,
        callbacks=[lr_monitor, early_stopping, model_checkpoint],
        max_epochs=5000,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
