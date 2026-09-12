import os

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

from dataset import DefectDataset, get_transforms
from model import get_model


# ============================================================
# CONFIGURATION
# ============================================================

VALID_DIR = "data/final_clean/valid"
VALID_CSV = "data/final_clean/valid/_classes.csv"

MODEL_PATH = "models/efficientnet_defect_v2.pth"

RESULTS_CSV = (
    "results/"
    "efficientnet_v2_threshold_search.csv"
)

RESULTS_TXT = (
    "results/"
    "efficientnet_v2_threshold_summary.txt"
)

BATCH_SIZE = 16

CLASS_NAMES = [
    "Burn Mark",
    "Flash",
    "Short Shot",
    "Sink Mark"
]


# ============================================================
# GET VALIDATION PROBABILITIES
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
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions
):
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

    return {
        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "micro_precision":
            micro_precision,

        "micro_recall":
            micro_recall,

        "micro_f1":
            micro_f1,

        "exact_match":
            exact_match,

        "hamming_accuracy":
            hamming_accuracy
    }


# ============================================================
# MAIN
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
        "EFFICIENTNET-B0 V2 "
        "VALIDATION THRESHOLD SEARCH"
    )

    print(
        "=" * 72
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    validation_dataset = DefectDataset(
        img_dir=VALID_DIR,
        csv_file=VALID_CSV,
        transform=get_transforms(
            is_train=False
        )
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        f"Validation images: "
        f"{len(validation_dataset)}"
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
        "\nLoaded:"
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
            validation_loader,
            device
        )
    )

    # --------------------------------------------------------
    # GLOBAL THRESHOLD SEARCH
    # --------------------------------------------------------

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    results = []

    print(
        "\n"
        + "=" * 72
    )

    print(
        "GLOBAL THRESHOLD SEARCH"
    )

    print(
        "=" * 72
    )

    for threshold in thresholds:

        predictions = (
            probabilities
            >=
            threshold
        ).astype(
            int
        )

        metrics = calculate_metrics(
            labels,
            predictions
        )

        row = {
            "threshold":
                float(
                    round(
                        threshold,
                        2
                    )
                ),

            **metrics
        }

        results.append(
            row
        )

        print(
            f"Threshold "
            f"{threshold:.2f} | "
            f"Macro F1: "
            f"{metrics['macro_f1']:.4f} | "
            f"Micro F1: "
            f"{metrics['micro_f1']:.4f} | "
            f"Hamming Acc: "
            f"{metrics['hamming_accuracy']:.4f}"
        )

    # --------------------------------------------------------
    # SELECT BEST GLOBAL THRESHOLD
    # --------------------------------------------------------

    best_result = max(
        results,
        key=lambda result:
            result[
                "macro_f1"
            ]
    )

    best_threshold = (
        best_result[
            "threshold"
        ]
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "BEST GLOBAL THRESHOLD"
    )

    print(
        "=" * 72
    )

    print(
        f"\nThreshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Macro Precision: "
        f"{best_result['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall: "
        f"{best_result['macro_recall']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{best_result['macro_f1']:.4f}"
    )

    print(
        f"Micro Precision: "
        f"{best_result['micro_precision']:.4f}"
    )

    print(
        f"Micro Recall: "
        f"{best_result['micro_recall']:.4f}"
    )

    print(
        f"Micro F1: "
        f"{best_result['micro_f1']:.4f}"
    )

    print(
        f"Exact Match: "
        f"{best_result['exact_match']:.4f}"
    )

    print(
        f"Hamming Accuracy: "
        f"{best_result['hamming_accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # PER-CLASS INFORMATION AT SELECTED THRESHOLD
    # --------------------------------------------------------

    best_predictions = (
        probabilities
        >=
        best_threshold
    ).astype(
        int
    )

    class_f1 = f1_score(
        labels,
        best_predictions,
        average=None,
        zero_division=0
    )

    class_precision = precision_score(
        labels,
        best_predictions,
        average=None,
        zero_division=0
    )

    class_recall = recall_score(
        labels,
        best_predictions,
        average=None,
        zero_division=0
    )

    print(
        "\nPer-class validation results:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"  {class_name:12s} | "
            f"Precision: "
            f"{class_precision[index]:.4f} | "
            f"Recall: "
            f"{class_recall[index]:.4f} | "
            f"F1: "
            f"{class_f1[index]:.4f}"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    dataframe = pd.DataFrame(
        results
    )

    dataframe.to_csv(
        RESULTS_CSV,
        index=False
    )

    with open(
        RESULTS_TXT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "EFFICIENTNET-B0 V2 "
            "THRESHOLD OPTIMIZATION\n"
        )

        file.write(
            "=" * 60
            +
            "\n\n"
        )

        file.write(
            f"Validation images: "
            f"{len(validation_dataset)}\n"
        )

        file.write(
            f"Selected threshold: "
            f"{best_threshold:.2f}\n\n"
        )

        for key, value in (
            best_result.items()
        ):
            file.write(
                f"{key}: "
                f"{value:.4f}\n"
            )

        file.write(
            "\nPer-class metrics:\n"
        )

        for index, class_name in enumerate(
            CLASS_NAMES
        ):
            file.write(
                f"{class_name}: "
                f"Precision="
                f"{class_precision[index]:.4f}, "
                f"Recall="
                f"{class_recall[index]:.4f}, "
                f"F1="
                f"{class_f1[index]:.4f}\n"
            )

    print(
        "\nThreshold table saved to:"
    )

    print(
        RESULTS_CSV
    )

    print(
        "\nSummary saved to:"
    )

    print(
        RESULTS_TXT
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The test set has not been used "
        "for threshold selection."
    )


if __name__ == "__main__":
    main()