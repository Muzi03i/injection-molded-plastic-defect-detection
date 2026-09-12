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
from model import get_model


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

    # Load VALIDATION data only

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

    # Load trained EfficientNet model

    model = get_model(
        num_classes=4,
        pretrained=False
    )

    model.load_state_dict(
        torch.load(
            'models/efficientnet_defect.pth',
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []

    # Get probabilities for validation images

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

    # Search for best SINGLE threshold

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

    print("\n--- Global Threshold Search ---")

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

    best_global_threshold = float(
        best_row['threshold']
    )

    print(
        "\nBest GLOBAL threshold:"
    )

    print(
        f"{best_global_threshold:.2f}"
    )

    print(
        f"Validation Macro F1: "
        f"{best_row['macro_f1']:.4f}"
    )

    print(
        f"Validation Micro F1: "
        f"{best_row['micro_f1']:.4f}"
    )

    # Optimize one threshold PER CLASS

    print(
        "\n--- Per-Class Threshold Search ---"
    )

    per_class_thresholds = []
    per_class_results = []

    for i, class_name in enumerate(
        target_names
    ):
        best_threshold = 0.50
        best_f1 = -1

        for threshold in thresholds:
            class_preds = (
                all_probs[:, i] >= threshold
            ).astype(int)

            class_f1 = f1_score(
                all_labels[:, i],
                class_preds,
                zero_division=0
            )

            if class_f1 > best_f1:
                best_f1 = class_f1
                best_threshold = threshold

        per_class_thresholds.append(
            best_threshold
        )

        per_class_results.append({
            'class': class_name,
            'threshold': best_threshold,
            'validation_f1': best_f1
        })

        print(
            f"{class_name:12s} -> "
            f"Threshold: {best_threshold:.2f} | "
            f"F1: {best_f1:.4f}"
        )

    per_class_thresholds = np.array(
        per_class_thresholds
    )

    # Evaluate per-class thresholds on VALIDATION set

    per_class_preds = (
        all_probs >= per_class_thresholds
    ).astype(int)

    per_class_macro_f1 = f1_score(
        all_labels,
        per_class_preds,
        average='macro',
        zero_division=0
    )

    per_class_micro_f1 = f1_score(
        all_labels,
        per_class_preds,
        average='micro',
        zero_division=0
    )

    per_class_exact_acc = accuracy_score(
        all_labels,
        per_class_preds
    )

    per_class_hamming_acc = (
        1 - hamming_loss(
            all_labels,
            per_class_preds
        )
    )

    print(
        "\n--- Per-Class Threshold Validation Results ---"
    )

    print(
        f"Macro F1:         "
        f"{per_class_macro_f1:.4f}"
    )

    print(
        f"Micro F1:         "
        f"{per_class_micro_f1:.4f}"
    )

    print(
        f"Exact Accuracy:    "
        f"{per_class_exact_acc:.4f}"
    )

    print(
        f"Hamming Accuracy:  "
        f"{per_class_hamming_acc:.4f}"
    )

    # Save results

    results_df.to_csv(
        'results/global_threshold_search.csv',
        index=False
    )

    per_class_df = pd.DataFrame(
        per_class_results
    )

    per_class_df.to_csv(
        'results/per_class_thresholds.csv',
        index=False
    )

    with open(
        'results/threshold_optimization.txt',
        'w'
    ) as f:

        f.write(
            "EfficientNet-B0 Threshold Optimization\n"
        )

        f.write(
            "=" * 50 + "\n\n"
        )

        f.write(
            f"Validation images: "
            f"{len(val_dataset)}\n\n"
        )

        f.write(
            "Best Global Threshold\n"
        )

        f.write(
            "-" * 30 + "\n"
        )

        f.write(
            f"Threshold: "
            f"{best_global_threshold:.2f}\n"
        )

        f.write(
            f"Macro F1: "
            f"{best_row['macro_f1']:.4f}\n"
        )

        f.write(
            f"Micro F1: "
            f"{best_row['micro_f1']:.4f}\n\n"
        )

        f.write(
            "Best Per-Class Thresholds\n"
        )

        f.write(
            "-" * 30 + "\n"
        )

        for result in per_class_results:
            f.write(
                f"{result['class']}: "
                f"{result['threshold']:.2f} "
                f"(F1 = "
                f"{result['validation_f1']:.4f})\n"
            )

        f.write(
            "\nPer-Class Threshold Results\n"
        )

        f.write(
            "-" * 30 + "\n"
        )

        f.write(
            f"Macro F1: "
            f"{per_class_macro_f1:.4f}\n"
        )

        f.write(
            f"Micro F1: "
            f"{per_class_micro_f1:.4f}\n"
        )

        f.write(
            f"Exact Accuracy: "
            f"{per_class_exact_acc:.4f}\n"
        )

        f.write(
            f"Hamming Accuracy: "
            f"{per_class_hamming_acc:.4f}\n"
        )

    #  Plot global threshold vs Macro F1

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        results_df['threshold'],
        results_df['macro_f1'],
        marker='o'
    )

    plt.xlabel(
        'Decision Threshold'
    )

    plt.ylabel(
        'Validation Macro F1'
    )

    plt.title(
        'Threshold Optimization - EfficientNet-B0'
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        'results/threshold_optimization.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(
        "\nResults saved to:"
    )

    print(
        "results/global_threshold_search.csv"
    )

    print(
        "results/per_class_thresholds.csv"
    )

    print(
        "results/threshold_optimization.txt"
    )

    print(
        "results/threshold_optimization.png"
    )


if __name__ == '__main__':
    optimize_threshold()