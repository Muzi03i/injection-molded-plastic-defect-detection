import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import DefectDataset, get_transforms
from model_resnet import (
    get_resnet_model,
    freeze_backbone,
    unfreeze_backbone
)


# Reproducibility
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def evaluate(model, val_loader, criterion, device):
    model.eval()

    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

    return val_loss / len(val_loader.dataset)


def train():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Executing ResNet18 training on device: {device}")

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Dataset setup

    train_dataset = DefectDataset(
        img_dir='data/raw/train',
        csv_file='data/raw/train/_classes.csv',
        transform=get_transforms(is_train=True)
    )

    val_dataset = DefectDataset(
        img_dir='data/raw/valid',
        csv_file='data/raw/valid/_classes.csv',
        transform=get_transforms(is_train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False
    )

    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    # Model

    model = get_resnet_model(
        num_classes=4,
        pretrained=True
    ).to(device)

    # Same class weighting as EfficientNet
    pos_weight = torch.tensor(
        [9.00, 2.62, 5.77, 3.47],
        dtype=torch.float32
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    logs = []

    best_val_loss = float('inf')

    global_epoch = 0

    # PHASE 1
    # Train classifier head only

    print("\n--- Phase 1: Training ResNet18 Classifier Head ---")

    freeze_backbone(model)

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=1e-3
    )

    for epoch in range(5):
        global_epoch += 1

        model.train()

        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()
            optimizer.step()

            train_loss += (
                loss.item() *
                images.size(0)
            )

        train_loss /= len(
            train_loader.dataset
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch + 1}/5 "
            f"- Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f}"
        )

        logs.append({
            'epoch': global_epoch,
            'phase': 'classifier_head',
            'learning_rate': 1e-3,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

    # PHASE 2
    # Fine-tune entire ResNet18

    print("\n--- Phase 2: Fine-Tuning Entire ResNet18 ---")

    unfreeze_backbone(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    for epoch in range(10):
        global_epoch += 1

        model.train()

        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()
            optimizer.step()

            train_loss += (
                loss.item() *
                images.size(0)
            )

        train_loss /= len(
            train_loader.dataset
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch + 1}/10 "
            f"- Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f}"
        )

        logs.append({
            'epoch': global_epoch,
            'phase': 'fine_tuning',
            'learning_rate': 1e-4,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        # Save best model based only on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                'models/resnet18_baseline.pth'
            )

            print(
                "--> Saved best ResNet18 checkpoint "
                f"(Val Loss: {best_val_loss:.4f})"
            )

    # Save training logs

    log_df = pd.DataFrame(logs)

    log_path = (
        'results/'
        'resnet18_training_log.csv'
    )

    log_df.to_csv(
        log_path,
        index=False
    )

    print(
        f"\nTraining log saved to '{log_path}'"
    )

    # Training graph

    plt.figure(figsize=(8, 5))

    plt.plot(
        log_df['epoch'],
        log_df['train_loss'],
        marker='o',
        label='Training Loss'
    )

    plt.plot(
        log_df['epoch'],
        log_df['val_loss'],
        marker='o',
        label='Validation Loss'
    )

    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title(
        'ResNet18 Baseline Training History'
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    graph_path = (
        'results/'
        'resnet18_training_history.png'
    )

    plt.savefig(
        graph_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(
        f"Training graph saved to '{graph_path}'"
    )

    print(
        "\nResNet18 baseline training complete."
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )


if __name__ == '__main__':
    train()