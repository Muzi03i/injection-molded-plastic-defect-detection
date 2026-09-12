import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report
)

from dataset import DefectDataset, get_transforms
from model import get_model


# ============================================================
# CONFIGURATION
# ============================================================

OLD_DATA_ROOT = Path("data/raw")

NEW_TEST_DIR = Path("data/final_clean/test")
NEW_TEST_CSV = Path(
    "data/final_clean/test/_classes.csv"
)

OLD_MODEL_PATH = (
    "models/efficientnet_defect.pth"
)

V2_MODEL_PATH = (
    "models/efficientnet_defect_v2.pth"
)

OLD_THRESHOLD = 0.60
V2_THRESHOLD = 0.40

BATCH_SIZE = 16

CLASS_NAMES = [
    "Burn Mark",
    "Flash",
    "Short Shot",
    "Sink Mark"
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# SHA256
# ============================================================

def sha256_file(path):
    hasher = hashlib.sha256()

    with open(path, "rb") as file:

        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


# ============================================================
# OLD DATASET HASHES
# ============================================================

def get_old_dataset_hashes():

    hashes = set()

    image_paths = [
        path
        for path
        in OLD_DATA_ROOT.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    print(
        f"Original dataset images: "
        f"{len(image_paths)}"
    )

    for index, path in enumerate(
        image_paths,
        start=1
    ):
        hashes.add(
            sha256_file(path)
        )

        if (
            index % 50 == 0
            or
            index == len(image_paths)
        ):
            print(
                f"  Hashed "
                f"{index}/"
                f"{len(image_paths)}"
            )

    return hashes


# ============================================================
# GET GENUINELY NEW TEST INDICES
# ============================================================

def get_new_only_indices(
    dataset,
    old_hashes
):

    new_indices = []
    overlap_indices = []

    for index in range(
        len(dataset)
    ):

        filename = str(
            dataset.df.iloc[
                index
            ]["filename"]
        ).strip()

        image_path = (
            NEW_TEST_DIR
            /
            filename
        )

        image_hash = sha256_file(
            image_path
        )

        if image_hash in old_hashes:
            overlap_indices.append(
                index
            )
        else:
            new_indices.append(
                index
            )

    return (
        new_indices,
        overlap_indices
    )


# ============================================================
# MODEL PREDICTIONS
# ============================================================

def predict(
    model_path,
    loader,
    device,
    threshold
):

    model = get_model(
        num_classes=4,
        pretrained=False
    ).to(
        device
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    all_labels = []
    all_probabilities = []

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

    labels = np.vstack(
        all_labels
    ).astype(
        int
    )

    probabilities = np.vstack(
        all_probabilities
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(
        int
    )

    return labels, predictions


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions
):

    return {
        "macro_precision":
            precision_score(
                labels,
                predictions,
                average="macro",
                zero_division=0
            ),

        "macro_recall":
            recall_score(
                labels,
                predictions,
                average="macro",
                zero_division=0
            ),

        "macro_f1":
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0
            ),

        "micro_precision":
            precision_score(
                labels,
                predictions,
                average="micro",
                zero_division=0
            ),

        "micro_recall":
            recall_score(
                labels,
                predictions,
                average="micro",
                zero_division=0
            ),

        "micro_f1":
            f1_score(
                labels,
                predictions,
                average="micro",
                zero_division=0
            ),

        "exact_match":
            accuracy_score(
                labels,
                predictions
            ),

        "hamming_accuracy":
            (
                1.0
                -
                np.mean(
                    labels
                    !=
                    predictions
                )
            )
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_model_results(
    name,
    labels,
    predictions
):

    metrics = calculate_metrics(
        labels,
        predictions
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        name
    )

    print(
        "=" * 72
    )

    print(
        classification_report(
            labels,
            predictions,
            target_names=CLASS_NAMES,
            zero_division=0,
            digits=4
        )
    )

    print(
        f"Macro Precision: "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall:    "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1:        "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Micro Precision: "
        f"{metrics['micro_precision']:.4f}"
    )

    print(
        f"Micro Recall:    "
        f"{metrics['micro_recall']:.4f}"
    )

    print(
        f"Micro F1:        "
        f"{metrics['micro_f1']:.4f}"
    )

    print(
        f"Exact Match:     "
        f"{metrics['exact_match']:.4f}"
    )

    print(
        f"Hamming Accuracy:"
        f" {metrics['hamming_accuracy']:.4f}"
    )

    return metrics


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
        "FAIR OLD MODEL VS V2 COMPARISON"
    )

    print(
        "=" * 72
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # FULL V2 TEST DATASET
    # --------------------------------------------------------

    full_test_dataset = DefectDataset(
        img_dir=str(
            NEW_TEST_DIR
        ),
        csv_file=str(
            NEW_TEST_CSV
        ),
        transform=get_transforms(
            is_train=False
        )
    )

    print(
        f"\nFull V2 test set: "
        f"{len(full_test_dataset)} images"
    )

    # --------------------------------------------------------
    # FIND ORIGINAL-DATASET IMAGES
    # --------------------------------------------------------

    old_hashes = (
        get_old_dataset_hashes()
    )

    (
        new_only_indices,
        overlap_indices
    ) = get_new_only_indices(
        full_test_dataset,
        old_hashes
    )

    print(
        "\nTest-set composition:"
    )

    print(
        f"  Images from original "
        f"262-image dataset: "
        f"{len(overlap_indices)}"
    )

    print(
        f"  Genuinely new images: "
        f"{len(new_only_indices)}"
    )

    if len(
        new_only_indices
    ) == 0:
        raise RuntimeError(
            "No genuinely new test images "
            "were found."
        )

    # --------------------------------------------------------
    # COMMON TEST SUBSET
    # --------------------------------------------------------

    common_test_dataset = Subset(
        full_test_dataset,
        new_only_indices
    )

    common_loader = DataLoader(
        common_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        "\nBoth models will be evaluated "
        "on exactly these "
        f"{len(common_test_dataset)} "
        "genuinely new images."
    )

    # --------------------------------------------------------
    # OLD MODEL
    # --------------------------------------------------------

    old_labels, old_predictions = predict(
        OLD_MODEL_PATH,
        common_loader,
        device,
        OLD_THRESHOLD
    )

    old_metrics = print_model_results(
        "ORIGINAL EFFICIENTNET "
        "(262-IMAGE DATASET)",
        old_labels,
        old_predictions
    )

    # --------------------------------------------------------
    # V2 MODEL
    # --------------------------------------------------------

    v2_labels, v2_predictions = predict(
        V2_MODEL_PATH,
        common_loader,
        device,
        V2_THRESHOLD
    )

    v2_metrics = print_model_results(
        "EFFICIENTNET V2 "
        "(975-IMAGE DATASET)",
        v2_labels,
        v2_predictions
    )

    # --------------------------------------------------------
    # DIRECT COMPARISON
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "DIRECT COMPARISON"
    )

    print(
        "=" * 72
    )

    print(
        "\nMetric                 "
        "OLD        V2       Change"
    )

    print(
        "-" * 55
    )

    comparison_metrics = [
        (
            "Macro Precision",
            "macro_precision"
        ),
        (
            "Macro Recall",
            "macro_recall"
        ),
        (
            "Macro F1",
            "macro_f1"
        ),
        (
            "Micro Precision",
            "micro_precision"
        ),
        (
            "Micro Recall",
            "micro_recall"
        ),
        (
            "Micro F1",
            "micro_f1"
        ),
        (
            "Exact Match",
            "exact_match"
        ),
        (
            "Hamming Accuracy",
            "hamming_accuracy"
        )
    ]

    for display_name, key in (
        comparison_metrics
    ):

        old_value = (
            old_metrics[
                key
            ]
        )

        new_value = (
            v2_metrics[
                key
            ]
        )

        change = (
            new_value
            -
            old_value
        )

        print(
            f"{display_name:20s} "
            f"{old_value:8.4f} "
            f"{new_value:8.4f} "
            f"{change:+8.4f}"
        )


if __name__ == "__main__":
    main()