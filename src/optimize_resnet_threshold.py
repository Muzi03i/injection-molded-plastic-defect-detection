import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    hamming_loss
)

from dataset import DefectDataset, get_transforms
from model_resnet import get_resnet_model


def optimize_threshold():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    target_names = [
        'burn mark',
        'flash',
        'short shot',
        'sink mark'
    ]

    print(f"Using device: {device}")

    os.makedirs('results', exist_ok=True)

    # Load validation dataset
    val_dataset = DefectDataset(
        img_dir='data/raw/valid',
        csv_file='data/raw/valid/_classes.csv',
        transform=get_transforms(is_train=False)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False
    )

    print(f"Validation images: {len(val_dataset)}")

    # Load trained ResNet18 baseline
    model = get_resnet_model(
        num_classes=4,
        pretrained=False
    )

    model.load_state_dict(
        torch.load(
            'models/resnet18_baseline.pth',
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []

    # Get validation probabilities
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.append(
                probs.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    # Thresholds to test
    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    thresholds = np.round(
        thresholds,
        2
    )

    global_results = []

    print("\n--- ResNet18 Global Threshold Search ---")

    for threshold in thresholds:
        preds = (
            all_probs >= threshold
        ).astype(int)

        macro_f1 = f1_score(
            all_labels,
            preds,
            average='macro',
            zero_division=0
        )

        micro_f1 = f1_score(
            all_labels,
            preds,
            average='micro',
            zero_division=0
        )

        exact_acc = accuracy_score(
            all_labels,
            preds
        )

        hamming_acc = (
            1 - hamming_loss(
                all_labels,
                preds
            )
        )

        global_results.append({
            'threshold': threshold,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'exact_accuracy': exact_acc,
            'hamming_accuracy': hamming_acc
        })

        print(
            f"Threshold {threshold:.2f} | "
            f"Macro F1: {macro_f1:.4f} | "
            f"Micro F1: {micro_f1:.4f}"
        )

    results_df = pd.DataFrame(
        global_results
    )

    best_row = results_df.loc[
        results_df['macro_f1'].idxmax()
    ]

    best_threshold = float(
        best_row['threshold']
    )

    print("\nBest ResNet18 GLOBAL threshold:")
    print(f"{best_threshold:.2f}")

    print(
        f"Validation Macro F1: "
        f"{best_row['macro_f1']:.4f}"
    )

    print(
        f"Validation Micro F1: "
        f"{best_row['micro_f1']:.4f}"
    )

    # Save results
    results_df.to_csv(
        'results/resnet18_threshold_search.csv',
        index=False
    )

    with open(
        'results/resnet18_threshold_optimization.txt',
        'w'
    ) as f:

        f.write("ResNet18 Threshold Optimization\n")
        f.write("=" * 50 + "\n\n")

        f.write(
            f"Validation images: "
            f"{len(val_dataset)}\n\n"
        )

        f.write(
            f"Best global threshold: "
            f"{best_threshold:.2f}\n"
        )

        f.write(
            f"Validation Macro F1: "
            f"{best_row['macro_f1']:.4f}\n"
        )

        f.write(
            f"Validation Micro F1: "
            f"{best_row['micro_f1']:.4f}\n"
        )

    # Plot threshold vs Macro F1
    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df['threshold'],
        results_df['macro_f1'],
        marker='o'
    )

    plt.xlabel('Decision Threshold')
    plt.ylabel('Validation Macro F1')

    plt.title(
        'Threshold Optimization - ResNet18 Baseline'
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        'results/resnet18_threshold_optimization.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print("\nResults saved to:")
    print("results/resnet18_threshold_search.csv")
    print("results/resnet18_threshold_optimization.txt")
    print("results/resnet18_threshold_optimization.png")


if __name__ == '__main__':
    optimize_threshold()