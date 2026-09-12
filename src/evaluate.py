import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
    accuracy_score,
    hamming_loss
)

from dataset import DefectDataset, get_transforms
from model import get_model


def run_evaluation():
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

    # Load test dataset

    test_dataset = DefectDataset(
        img_dir='data/raw/test',
        csv_file='data/raw/test/_classes.csv',
        transform=get_transforms(is_train=False)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )

    print(f"Test images: {len(test_dataset)}")

    # Load trained EfficientNet-B0 model

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

    all_preds = []
    all_labels = []

    # Final decision threshold.
    # Selected using the validation set by maximizing Macro F1.
    threshold = 0.60

    # Generate predictions

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            preds = (
                probs >= threshold
            ).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Classification metrics

    report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        zero_division=0
    )

    print(
        "\n--- EfficientNet-B0 Test Classification Report ---"
    )

    print(report)

    # Exact-match accuracy:
    # All four labels for an image must be correct.
    exact_accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    # Hamming accuracy:
    # Percentage of individual defect decisions that are correct.
    hamming_accuracy = (
        1 - hamming_loss(
            all_labels,
            all_preds
        )
    )

    print(
        f"Exact-match Accuracy: "
        f"{exact_accuracy:.4f} "
        f"({exact_accuracy * 100:.2f}%)"
    )

    print(
        f"Hamming Accuracy:     "
        f"{hamming_accuracy:.4f} "
        f"({hamming_accuracy * 100:.2f}%)"
    )

    print(
        f"Decision Threshold:   "
        f"{threshold:.2f}"
    )

    # Save numerical evaluation results

    results_file = (
        'results/'
        'efficientnet_final_evaluation.txt'
    )

    with open(results_file, 'w') as f:
        f.write(
            "EfficientNet-B0 Final Test Evaluation\n"
        )

        f.write(
            "=" * 50 + "\n\n"
        )

        f.write(
            f"Number of test images: "
            f"{len(test_dataset)}\n"
        )

        f.write(
            f"Decision threshold: "
            f"{threshold:.2f}\n"
        )

        f.write(
            "Threshold selection method: "
            "Maximum Macro F1 on validation set\n\n"
        )

        f.write(
            "Classification Report\n"
        )

        f.write(
            "-" * 50 + "\n"
        )

        f.write(report)

        f.write("\n")

        f.write(
            f"Exact-match Accuracy: "
            f"{exact_accuracy:.4f} "
            f"({exact_accuracy * 100:.2f}%)\n"
        )

        f.write(
            f"Hamming Accuracy: "
            f"{hamming_accuracy:.4f} "
            f"({hamming_accuracy * 100:.2f}%)\n"
        )

    print(
        f"\nEvaluation results saved to "
        f"'{results_file}'"
    )

    # Confusion matrices

    mcm = multilabel_confusion_matrix(
        all_labels,
        all_preds
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8)
    )

    axes = axes.ravel()

    for i, cls_name in enumerate(target_names):
        cm = mcm[i]

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=axes[i],
            xticklabels=[
                'Absent',
                'Present'
            ],
            yticklabels=[
                'Absent',
                'Present'
            ]
        )

        axes[i].set_title(
            f'Confusion Matrix: {cls_name}'
        )

        axes[i].set_xlabel(
            'Predicted'
        )

        axes[i].set_ylabel(
            'Actual'
        )

    plt.tight_layout()

    confusion_path = (
        'results/'
        'efficientnet_final_confusion_matrix.png'
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(
        f"Confusion matrix saved to "
        f"'{confusion_path}'"
    )

    plt.show()


if __name__ == '__main__':
    run_evaluation()