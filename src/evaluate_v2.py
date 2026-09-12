import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from dataset import DefectDataset, get_transforms
from model import get_model


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = "data/final_clean/test"
TEST_CSV = "data/final_clean/test/_classes.csv"

MODEL_PATH = "models/efficientnet_defect_v2.pth"

RESULTS_TXT = (
    "results/"
    "efficientnet_v2_final_evaluation.txt"
)

CONFUSION_PATH = (
    "results/"
    "efficientnet_v2_confusion_matrix.png"
)

BATCH_SIZE = 16

# Locked using validation Macro F1.
THRESHOLD = 0.40

CLASS_NAMES = [
    "Burn Mark",
    "Flash",
    "Short Shot",
    "Sink Mark"
]


# ============================================================
# PREDICTIONS
# ============================================================

def get_predictions(
    model,
    loader,
    device
):
    model.eval()

    all_probabilities = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            outputs = model(
                images
            )

            probabilities = torch.sigmoid(
                outputs
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

    probabilities = np.vstack(
        all_probabilities
    )

    labels = np.vstack(
        all_labels
    ).astype(
        int
    )

    return probabilities, labels


# ============================================================
# CONFUSION MATRICES
# ============================================================

def save_confusion_matrices(
    labels,
    predictions
):
    matrices = multilabel_confusion_matrix(
        labels,
        predictions
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8)
    )

    axes = axes.flatten()

    for index, (
        matrix,
        class_name
    ) in enumerate(
        zip(
            matrices,
            CLASS_NAMES
        )
    ):

        tn, fp, fn, tp = (
            matrix.ravel()
        )

        display_matrix = np.array([
            [tn, fp],
            [fn, tp]
        ])

        axis = axes[
            index
        ]

        image = axis.imshow(
            display_matrix,
            cmap="Blues"
        )

        axis.set_title(
            class_name
        )

        axis.set_xticks(
            [0, 1]
        )

        axis.set_yticks(
            [0, 1]
        )

        axis.set_xticklabels(
            ["Predicted 0", "Predicted 1"]
        )

        axis.set_yticklabels(
            ["Actual 0", "Actual 1"]
        )

        for row in range(2):
            for column in range(2):

                axis.text(
                    column,
                    row,
                    str(
                        display_matrix[
                            row,
                            column
                        ]
                    ),
                    ha="center",
                    va="center"
                )

    figure.suptitle(
        "EfficientNet-B0 V2 "
        "Test Confusion Matrices"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 72
    )

    print(
        "EFFICIENTNET-B0 V2 FINAL TEST EVALUATION"
    )

    print(
        "=" * 72
    )

    print(
        f"\nDevice: {device}"
    )

    print(
        f"Locked threshold: "
        f"{THRESHOLD:.2f}"
    )

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    test_dataset = DefectDataset(
        img_dir=TEST_DIR,
        csv_file=TEST_CSV,
        transform=get_transforms(
            is_train=False
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        f"Test images: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = get_model(
        num_classes=4,
        pretrained=False
    ).to(
        device
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    print(
        "\nLoaded model:"
    )

    print(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    probabilities, labels = (
        get_predictions(
            model,
            test_loader,
            device
        )
    )

    predictions = (
        probabilities
        >= THRESHOLD
    ).astype(
        int
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "CLASSIFICATION REPORT"
    )

    print(
        "=" * 72
    )

    print(
        report
    )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    macro_precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    micro_precision = precision_score(
        labels,
        predictions,
        average="micro",
        zero_division=0
    )

    micro_recall = recall_score(
        labels,
        predictions,
        average="micro",
        zero_division=0
    )

    micro_f1 = f1_score(
        labels,
        predictions,
        average="micro",
        zero_division=0
    )

    exact_match = accuracy_score(
        labels,
        predictions
    )

    hamming_accuracy = (
        1.0
        -
        np.mean(
            labels != predictions
        )
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "OVERALL TEST METRICS"
    )

    print(
        "=" * 72
    )

    print(
        f"Macro Precision: "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall:    "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1:        "
        f"{macro_f1:.4f}"
    )

    print()

    print(
        f"Micro Precision: "
        f"{micro_precision:.4f}"
    )

    print(
        f"Micro Recall:    "
        f"{micro_recall:.4f}"
    )

    print(
        f"Micro F1:        "
        f"{micro_f1:.4f}"
    )

    print()

    print(
        f"Exact Match:     "
        f"{exact_match:.4f} "
        f"({exact_match * 100:.2f}%)"
    )

    print(
        f"Hamming Accuracy:"
        f" {hamming_accuracy:.4f} "
        f"({hamming_accuracy * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    matrices = multilabel_confusion_matrix(
        labels,
        predictions
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PER-CLASS CONFUSION MATRICES"
    )

    print(
        "=" * 72
    )

    for class_name, matrix in zip(
        CLASS_NAMES,
        matrices
    ):

        tn, fp, fn, tp = (
            matrix.ravel()
        )

        print(
            f"\n{class_name}"
        )

        print(
            f"  TN: {tn}"
        )

        print(
            f"  FP: {fp}"
        )

        print(
            f"  FN: {fn}"
        )

        print(
            f"  TP: {tp}"
        )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    save_confusion_matrices(
        labels,
        predictions
    )

    with open(
        RESULTS_TXT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "EFFICIENTNET-B0 V2 "
            "FINAL TEST EVALUATION\n"
        )

        file.write(
            "=" * 60
            +
            "\n\n"
        )

        file.write(
            f"Model: {MODEL_PATH}\n"
        )

        file.write(
            f"Test images: "
            f"{len(test_dataset)}\n"
        )

        file.write(
            f"Threshold: "
            f"{THRESHOLD:.2f}\n"
        )

        file.write(
            "Threshold selected using "
            "validation Macro F1 only.\n\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n"
        )

        file.write(
            "-" * 60
            +
            "\n"
        )

        file.write(
            report
        )

        file.write(
            "\nOVERALL METRICS\n"
        )

        file.write(
            "-" * 60
            +
            "\n"
        )

        file.write(
            f"Macro Precision: "
            f"{macro_precision:.4f}\n"
        )

        file.write(
            f"Macro Recall: "
            f"{macro_recall:.4f}\n"
        )

        file.write(
            f"Macro F1: "
            f"{macro_f1:.4f}\n"
        )

        file.write(
            f"Micro Precision: "
            f"{micro_precision:.4f}\n"
        )

        file.write(
            f"Micro Recall: "
            f"{micro_recall:.4f}\n"
        )

        file.write(
            f"Micro F1: "
            f"{micro_f1:.4f}\n"
        )

        file.write(
            f"Exact Match: "
            f"{exact_match:.4f}\n"
        )

        file.write(
            f"Hamming Accuracy: "
            f"{hamming_accuracy:.4f}\n"
        )

        file.write(
            "\nPER-CLASS CONFUSION MATRICES\n"
        )

        file.write(
            "-" * 60
            +
            "\n"
        )

        for class_name, matrix in zip(
            CLASS_NAMES,
            matrices
        ):

            tn, fp, fn, tp = (
                matrix.ravel()
            )

            file.write(
                f"\n{class_name}\n"
            )

            file.write(
                f"TN={tn}, "
                f"FP={fp}, "
                f"FN={fn}, "
                f"TP={tp}\n"
            )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_TXT
    )

    print(
        "\nConfusion matrices saved to:"
    )

    print(
        CONFUSION_PATH
    )


if __name__ == "__main__":
    main()