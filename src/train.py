import os
import random
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import DefectDataset, get_transforms
from model import get_model, freeze_backbone, unfreeze_backbone


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "data/final_clean/train"
VALID_DIR = "data/final_clean/valid"

TRAIN_CSV = "data/final_clean/train/_classes.csv"
VALID_CSV = "data/final_clean/valid/_classes.csv"

MODEL_PATH = "models/efficientnet_defect_v2.pth"

LOG_PATH = "results/efficientnet_v2_training_log.csv"
PLOT_PATH = "results/efficientnet_v2_training_history.png"

BATCH_SIZE = 16

HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 10

HEAD_LR = 1e-3
FINETUNE_LR = 1e-4

RANDOM_SEED = 42

CLASS_NAMES = [
    "burn mark",
    "flash",
    "short shot",
    "sink mark"
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    val_loader,
    criterion,
    device
):
    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item()
                *
                images.size(0)
            )

    return (
        val_loss
        /
        len(
            val_loader.dataset
        )
    )


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

def calculate_pos_weights(
    dataset,
    device
):
    total_images = len(
        dataset
    )

    positive_counts = []

    for class_name in CLASS_NAMES:

        count = int(
            dataset.df[
                class_name
            ].sum()
        )

        positive_counts.append(
            count
        )

    weights = []

    for positive_count in positive_counts:

        negative_count = (
            total_images
            -
            positive_count
        )

        weight = (
            negative_count
            /
            positive_count
        )

        weights.append(
            weight
        )

    print(
        "\nTraining class distribution:"
    )

    for (
        class_name,
        positive_count,
        weight
    ) in zip(
        CLASS_NAMES,
        positive_counts,
        weights
    ):
        print(
            f"  {class_name:12s} | "
            f"Positive: {positive_count:3d} | "
            f"pos_weight: {weight:.4f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):
    model.train()

    train_loss = 0.0

    for images, labels in train_loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            *
            images.size(0)
        )

    return (
        train_loss
        /
        len(
            train_loader.dataset
        )
    )


# ============================================================
# SAVE TRAINING LOG
# ============================================================

def save_training_log(
    history
):
    os.makedirs(
        "results",
        exist_ok=True
    )

    with open(
        LOG_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "phase",
                "epoch",
                "train_loss",
                "val_loss"
            ]
        )

        writer.writeheader()

        writer.writerows(
            history
        )


# ============================================================
# SAVE TRAINING GRAPH
# ============================================================

def save_training_plot(
    history
):
    train_losses = [
        item["train_loss"]
        for item in history
    ]

    val_losses = [
        item["val_loss"]
        for item in history
    ]

    epochs = list(
        range(
            1,
            len(history) + 1
        )
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        train_losses,
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        marker="o",
        label="Validation Loss"
    )

    plt.axvline(
        x=HEAD_EPOCHS + 0.5,
        linestyle="--",
        label="Fine-Tuning Begins"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "BCE Loss"
    )

    plt.title(
        "EfficientNet-B0 V2 Training History"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN TRAINING
# ============================================================

def train():

    set_seed(
        RANDOM_SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 70
    )

    print(
        "EFFICIENTNET-B0 V2 TRAINING"
    )

    print(
        "=" * 70
    )

    print(
        f"\nExecuting training on device: "
        f"{device}"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    train_dataset = DefectDataset(
        img_dir=TRAIN_DIR,
        csv_file=TRAIN_CSV,
        transform=get_transforms(
            is_train=True
        )
    )

    val_dataset = DefectDataset(
        img_dir=VALID_DIR,
        csv_file=VALID_CSV,
        transform=get_transforms(
            is_train=False
        )
    )

    print(
        f"\nTraining images:   "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        RANDOM_SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        generator=train_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        )
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nLoading ImageNet-pretrained "
        "EfficientNet-B0..."
    )

    model = get_model(
        num_classes=4,
        pretrained=True
    ).to(
        device
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    pos_weight = (
        calculate_pos_weights(
            train_dataset,
            device
        )
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    best_val_loss = float(
        "inf"
    )

    history = []

    # ========================================================
    # PHASE 1
    # Train classifier head only
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PHASE 1: TRAINING CLASSIFIER HEAD"
    )

    print(
        "=" * 70
    )

    freeze_backbone(
        model
    )

    optimizer = torch.optim.Adam(
        filter(
            lambda parameter:
                parameter.requires_grad,
            model.parameters()
        ),
        lr=HEAD_LR
    )

    for epoch in range(
        1,
        HEAD_EPOCHS + 1
    ):

        train_loss = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        history.append({
            "phase":
                "classifier_head",

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "val_loss":
                val_loss
        })

        print(
            f"Epoch "
            f"{epoch}/{HEAD_EPOCHS} "
            f"- Train Loss: "
            f"{train_loss:.4f} "
            f"| Val Loss: "
            f"{val_loss:.4f}"
        )

        if (
            val_loss
            <
            best_val_loss
        ):
            best_val_loss = (
                val_loss
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                f"--> Saved best checkpoint "
                f"(Val Loss: "
                f"{best_val_loss:.4f})"
            )

    # ========================================================
    # PHASE 2
    # Fine-tune entire network
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PHASE 2: FINE-TUNING ENTIRE NETWORK"
    )

    print(
        "=" * 70
    )

    unfreeze_backbone(
        model
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FINETUNE_LR
    )

    for epoch in range(
        1,
        FINETUNE_EPOCHS + 1
    ):

        train_loss = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        history.append({
            "phase":
                "fine_tuning",

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "val_loss":
                val_loss
        })

        print(
            f"Epoch "
            f"{epoch}/{FINETUNE_EPOCHS} "
            f"- Train Loss: "
            f"{train_loss:.4f} "
            f"| Val Loss: "
            f"{val_loss:.4f}"
        )

        if (
            val_loss
            <
            best_val_loss
        ):
            best_val_loss = (
                val_loss
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                f"--> Saved best checkpoint "
                f"(Val Loss: "
                f"{best_val_loss:.4f})"
            )

    # --------------------------------------------------------
    # SAVE LOGS
    # --------------------------------------------------------

    save_training_log(
        history
    )

    save_training_plot(
        history
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBest validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        "\nBest model saved to:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nTraining log saved to:"
    )

    print(
        LOG_PATH
    )

    print(
        "\nTraining plot saved to:"
    )

    print(
        PLOT_PATH
    )


if __name__ == "__main__":
    train()