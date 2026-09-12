import csv
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from PIL import Image, ImageOps
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

NEW_TEST_DIR = Path(
    "data/final_clean/test"
)

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

# Same strict duplicate criteria used when
# constructing data/final_clean.
PHASH_THRESHOLD = 4
DHASH_THRESHOLD = 6
COSINE_THRESHOLD = 0.985

FILTER_REPORT = (
    "results/"
    "strict_comparison_filter_report.csv"
)

RESULTS_FILE = (
    "results/"
    "strict_old_vs_v2_comparison.txt"
)


# ============================================================
# HASHING
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

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


def phash(image):
    gray = np.array(
        image.convert("L").resize(
            (32, 32),
            Image.Resampling.LANCZOS
        ),
        dtype=np.float32
    )

    transformed = cv2.dct(
        gray
    )

    low_frequency = (
        transformed[:8, :8]
    )

    values = (
        low_frequency.flatten()
    )

    median = np.median(
        values[1:]
    )

    return (
        values > median
    ).astype(
        np.uint8
    )


def dhash(image):
    gray = np.array(
        image.convert("L").resize(
            (9, 8),
            Image.Resampling.LANCZOS
        )
    )

    return (
        gray[:, 1:]
        >
        gray[:, :-1]
    ).flatten().astype(
        np.uint8
    )


def similarity_vector(image):
    gray = np.array(
        image.convert("L").resize(
            (64, 64),
            Image.Resampling.LANCZOS
        ),
        dtype=np.float32
    )

    vector = gray.flatten()

    vector = (
        vector
        -
        np.mean(vector)
    )

    norm = np.linalg.norm(
        vector
    )

    if norm > 0:
        vector = (
            vector
            /
            norm
        )

    return vector


def hamming_distance(
    first,
    second
):
    return int(
        np.count_nonzero(
            first != second
        )
    )


def cosine_similarity(
    first,
    second
):
    return float(
        np.dot(
            first,
            second
        )
    )


# ============================================================
# IMAGE FEATURES
# ============================================================

def calculate_image_features(path):
    with Image.open(path) as image:

        image = image.convert(
            "RGB"
        )

        mirrored = ImageOps.mirror(
            image
        )

        return {
            "sha256":
                sha256_file(path),

            "phash":
                phash(image),

            "dhash":
                dhash(image),

            "vector":
                similarity_vector(image),

            "flip_phash":
                phash(mirrored),

            "flip_dhash":
                dhash(mirrored),

            "flip_vector":
                similarity_vector(mirrored)
        }


# ============================================================
# STRICT NEAR-DUPLICATE CHECK
# ============================================================

def strong_near_duplicate(
    old_features,
    new_features
):
    # Normal orientation
    p_distance = hamming_distance(
        old_features["phash"],
        new_features["phash"]
    )

    d_distance = hamming_distance(
        old_features["dhash"],
        new_features["dhash"]
    )

    if (
        p_distance <= PHASH_THRESHOLD
        and
        d_distance <= DHASH_THRESHOLD
    ):
        similarity = cosine_similarity(
            old_features["vector"],
            new_features["vector"]
        )

        if similarity >= COSINE_THRESHOLD:
            return (
                True,
                p_distance,
                d_distance,
                similarity,
                "normal"
            )

    # Horizontal flip
    p_distance = hamming_distance(
        old_features["phash"],
        new_features["flip_phash"]
    )

    d_distance = hamming_distance(
        old_features["dhash"],
        new_features["flip_dhash"]
    )

    if (
        p_distance <= PHASH_THRESHOLD
        and
        d_distance <= DHASH_THRESHOLD
    ):
        similarity = cosine_similarity(
            old_features["vector"],
            new_features["flip_vector"]
        )

        if similarity >= COSINE_THRESHOLD:
            return (
                True,
                p_distance,
                d_distance,
                similarity,
                "horizontal_flip"
            )

    return (
        False,
        None,
        None,
        None,
        None
    )


# ============================================================
# LOAD ORIGINAL DATASET FEATURES
# ============================================================

def load_old_features():

    paths = sorted([
        path
        for path in OLD_DATA_ROOT.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ])

    print(
        f"Original dataset images: "
        f"{len(paths)}"
    )

    records = []

    for index, path in enumerate(
        paths,
        start=1
    ):
        records.append({
            "path":
                path,

            "features":
                calculate_image_features(
                    path
                )
        })

        if (
            index % 50 == 0
            or
            index == len(paths)
        ):
            print(
                f"  Processed "
                f"{index}/{len(paths)}"
            )

    return records


# ============================================================
# FILTER V2 TEST SET
# ============================================================

def find_strictly_new_images(
    dataset,
    old_records
):

    kept_indices = []
    excluded_indices = []

    report_rows = []

    print(
        "\nChecking V2 test images "
        "against entire original dataset..."
    )

    total = len(
        dataset
    )

    for index in range(
        total
    ):

        filename = str(
            dataset.df.iloc[
                index
            ]["filename"]
        ).strip()

        new_path = (
            NEW_TEST_DIR
            /
            filename
        )

        new_features = (
            calculate_image_features(
                new_path
            )
        )

        match_found = False

        for old_record in old_records:

            old_features = (
                old_record[
                    "features"
                ]
            )

            # ----------------------------------------------
            # EXACT DUPLICATE
            # ----------------------------------------------

            if (
                new_features["sha256"]
                ==
                old_features["sha256"]
            ):

                report_rows.append({
                    "new_filename":
                        filename,

                    "status":
                        "excluded",

                    "match_type":
                        "exact",

                    "old_image":
                        str(
                            old_record[
                                "path"
                            ]
                        ),

                    "phash_distance":
                        0,

                    "dhash_distance":
                        0,

                    "cosine_similarity":
                        1.0,

                    "orientation":
                        "identical"
                })

                excluded_indices.append(
                    index
                )

                match_found = True
                break

            # ----------------------------------------------
            # STRICT NEAR DUPLICATE
            # ----------------------------------------------

            (
                is_duplicate,
                p_distance,
                d_distance,
                similarity,
                orientation
            ) = strong_near_duplicate(
                old_features,
                new_features
            )

            if is_duplicate:

                report_rows.append({
                    "new_filename":
                        filename,

                    "status":
                        "excluded",

                    "match_type":
                        "strong_near_duplicate",

                    "old_image":
                        str(
                            old_record[
                                "path"
                            ]
                        ),

                    "phash_distance":
                        p_distance,

                    "dhash_distance":
                        d_distance,

                    "cosine_similarity":
                        similarity,

                    "orientation":
                        orientation
                })

                excluded_indices.append(
                    index
                )

                match_found = True
                break

        if not match_found:

            kept_indices.append(
                index
            )

            report_rows.append({
                "new_filename":
                    filename,

                "status":
                    "kept",

                "match_type":
                    "",

                "old_image":
                    "",

                "phash_distance":
                    "",

                "dhash_distance":
                    "",

                "cosine_similarity":
                    "",

                "orientation":
                    ""
            })

        if (
            (index + 1) % 25 == 0
            or
            index + 1 == total
        ):
            print(
                f"  Checked "
                f"{index + 1}/{total}"
            )

    return (
        kept_indices,
        excluded_indices,
        report_rows
    )


# ============================================================
# MODEL PREDICTION
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

            all_labels.append(
                labels.numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
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
        probabilities >= threshold
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
                    labels != predictions
                )
            )
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
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

    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4
    )

    print(
        report
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

    return metrics, report


# ============================================================
# SAVE FILTER REPORT
# ============================================================

def save_filter_report(
    rows
):

    os.makedirs(
        "results",
        exist_ok=True
    )

    with open(
        FILTER_REPORT,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "new_filename",
                "status",
                "match_type",
                "old_image",
                "phash_distance",
                "dhash_distance",
                "cosine_similarity",
                "orientation"
            ]
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


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
        "STRICT OLD MODEL VS V2 COMPARISON"
    )

    print(
        "=" * 72
    )

    print(
        f"\nDevice: {device}"
    )

    print(
        "\nStrict filtering criteria:"
    )

    print(
        f"  pHash <= "
        f"{PHASH_THRESHOLD}"
    )

    print(
        f"  dHash <= "
        f"{DHASH_THRESHOLD}"
    )

    print(
        f"  cosine similarity >= "
        f"{COSINE_THRESHOLD}"
    )

    # --------------------------------------------------------
    # FULL TEST DATASET
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
        f"{len(full_test_dataset)}"
    )

    # --------------------------------------------------------
    # ORIGINAL DATASET FEATURES
    # --------------------------------------------------------

    old_records = (
        load_old_features()
    )

    # --------------------------------------------------------
    # STRICT FILTER
    # --------------------------------------------------------

    (
        kept_indices,
        excluded_indices,
        report_rows
    ) = find_strictly_new_images(
        full_test_dataset,
        old_records
    )

    save_filter_report(
        report_rows
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "STRICT FILTER RESULTS"
    )

    print(
        "=" * 72
    )

    print(
        f"\nFull V2 test images: "
        f"{len(full_test_dataset)}"
    )

    print(
        f"Excluded exact/near duplicates: "
        f"{len(excluded_indices)}"
    )

    print(
        f"Strictly unseen images remaining: "
        f"{len(kept_indices)}"
    )

    if len(
        kept_indices
    ) < 20:
        print(
            "\nWARNING:"
        )

        print(
            "The strict comparison subset is "
            "very small, so results may be unstable."
        )

    if len(
        kept_indices
    ) == 0:
        raise RuntimeError(
            "No strictly unseen images remain."
        )

    # --------------------------------------------------------
    # COMMON STRICT SUBSET
    # --------------------------------------------------------

    strict_dataset = Subset(
        full_test_dataset,
        kept_indices
    )

    strict_loader = DataLoader(
        strict_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # ORIGINAL MODEL
    # --------------------------------------------------------

    old_labels, old_predictions = (
        predict(
            OLD_MODEL_PATH,
            strict_loader,
            device,
            OLD_THRESHOLD
        )
    )

    (
        old_metrics,
        old_report
    ) = print_results(
        "ORIGINAL EFFICIENTNET "
        "(262-IMAGE TRAINING DATASET)",
        old_labels,
        old_predictions
    )

    # --------------------------------------------------------
    # V2 MODEL
    # --------------------------------------------------------

    v2_labels, v2_predictions = (
        predict(
            V2_MODEL_PATH,
            strict_loader,
            device,
            V2_THRESHOLD
        )
    )

    (
        v2_metrics,
        v2_report
    ) = print_results(
        "EFFICIENTNET V2 "
        "(975-IMAGE SOURCE DATASET)",
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
        "STRICT DIRECT COMPARISON"
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

    comparison_lines = []

    for display_name, key in (
        comparison_metrics
    ):

        old_value = (
            old_metrics[
                key
            ]
        )

        v2_value = (
            v2_metrics[
                key
            ]
        )

        change = (
            v2_value
            -
            old_value
        )

        line = (
            f"{display_name:20s} "
            f"{old_value:8.4f} "
            f"{v2_value:8.4f} "
            f"{change:+8.4f}"
        )

        comparison_lines.append(
            line
        )

        print(
            line
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "STRICT OLD MODEL VS V2 COMPARISON\n"
        )

        file.write(
            "=" * 60
            +
            "\n\n"
        )

        file.write(
            f"Full V2 test images: "
            f"{len(full_test_dataset)}\n"
        )

        file.write(
            f"Excluded exact/near duplicates: "
            f"{len(excluded_indices)}\n"
        )

        file.write(
            f"Strictly unseen images: "
            f"{len(kept_indices)}\n\n"
        )

        file.write(
            "Strict duplicate thresholds:\n"
        )

        file.write(
            f"pHash <= "
            f"{PHASH_THRESHOLD}\n"
        )

        file.write(
            f"dHash <= "
            f"{DHASH_THRESHOLD}\n"
        )

        file.write(
            f"Cosine similarity >= "
            f"{COSINE_THRESHOLD}\n\n"
        )

        file.write(
            "ORIGINAL MODEL\n"
        )

        file.write(
            "-" * 60
            +
            "\n"
        )

        file.write(
            old_report
        )

        file.write(
            "\n\nV2 MODEL\n"
        )

        file.write(
            "-" * 60
            +
            "\n"
        )

        file.write(
            v2_report
        )

        file.write(
            "\n\nDIRECT COMPARISON\n"
        )

        file.write(
            "-" * 60
            +
            "\n"
        )

        for line in comparison_lines:
            file.write(
                line
                +
                "\n"
            )

    print(
        "\nFilter report saved to:"
    )

    print(
        FILTER_REPORT
    )

    print(
        "\nComparison results saved to:"
    )

    print(
        RESULTS_FILE
    )


if __name__ == "__main__":
    main()