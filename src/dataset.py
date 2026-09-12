import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = [
    "burn mark",
    "flash",
    "short shot",
    "sink mark"
]


def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class DefectDataset(Dataset):

    def __init__(
        self,
        img_dir,
        csv_file,
        transform=None
    ):
        self.img_dir = img_dir
        self.transform = transform

        self.df = pd.read_csv(
            csv_file
        )

        # Normalize all CSV headers.
        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.lower()
        )

        self.target_cols = CLASS_NAMES

        # Verify that all required columns exist.
        required_columns = (
            ["filename"]
            +
            self.target_cols
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing CSV columns: "
                + ", ".join(
                    missing_columns
                )
            )

    def __len__(self):
        return len(
            self.df
        )

    def __getitem__(
        self,
        idx
    ):
        row = self.df.iloc[
            idx
        ]

        img_name = str(
            row["filename"]
        ).strip()

        img_path = os.path.join(
            self.img_dir,
            img_name
        )

        image = Image.open(
            img_path
        ).convert(
            "RGB"
        )

        labels = row[
            self.target_cols
        ].to_numpy(
            dtype="float32"
        )

        labels = torch.tensor(
            labels,
            dtype=torch.float32
        )

        if self.transform:
            image = self.transform(
                image
            )

        return image, labels