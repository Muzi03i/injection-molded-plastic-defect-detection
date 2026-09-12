import hashlib
from pathlib import Path
from collections import defaultdict
import csv

import cv2
import numpy as np
from PIL import Image, ImageOps


OLD_ROOT = Path("data/raw")
NEW_ROOT = Path("new_dataset/Final Project.v1i.coco")

OUTPUT_FILE = Path(
    "results/final_project_duplicate_report.csv"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

PHASH_THRESHOLD = 6
DHASH_THRESHOLD = 10


def collect_images(root):
    images = []

    for path in root.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            images.append(path)

    return sorted(images)


def get_split(path):
    parts = [
        p.lower()
        for p in path.parts
    ]

    if "train" in parts:
        return "train"

    if "valid" in parts:
        return "valid"

    if "val" in parts:
        return "valid"

    if "test" in parts:
        return "test"

    return "unknown"


def sha256_hash(path):
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


def phash_image(image):
    """
    64-bit perceptual DCT hash.
    """

    gray = np.array(
        image.convert("L").resize(
            (32, 32),
            Image.Resampling.LANCZOS
        ),
        dtype=np.float32
    )

    dct = cv2.dct(
        gray
    )

    low_freq = dct[
        :8,
        :8
    ]

    values = low_freq.flatten()

    # Ignore DC component when calculating median
    median = np.median(
        values[1:]
    )

    bits = (
        values > median
    ).astype(
        np.uint8
    )

    return bits


def dhash_image(image):
    """
    64-bit difference hash.
    """

    gray = np.array(
        image.convert("L").resize(
            (9, 8),
            Image.Resampling.LANCZOS
        )
    )

    difference = (
        gray[:, 1:]
        >
        gray[:, :-1]
    )

    return difference.flatten().astype(
        np.uint8
    )


def image_hashes(path):
    with Image.open(path) as image:
        image = image.convert("RGB")

        normal_phash = phash_image(
            image
        )

        normal_dhash = dhash_image(
            image
        )

        # Also calculate horizontal-flip hashes
        flipped = ImageOps.mirror(
            image
        )

        flipped_phash = phash_image(
            flipped
        )

        flipped_dhash = dhash_image(
            flipped
        )

    return {
        "phash":
            normal_phash,

        "dhash":
            normal_dhash,

        "flip_phash":
            flipped_phash,

        "flip_dhash":
            flipped_dhash
    }


def hamming_distance(hash1, hash2):
    return int(
        np.count_nonzero(
            hash1 != hash2
        )
    )


def perceptual_distance(
    first,
    second
):
    """
    Compare normal orientation as well as
    horizontal-flipped orientation.

    Returns the best matching distances.
    """

    normal_p = hamming_distance(
        first["phash"],
        second["phash"]
    )

    normal_d = hamming_distance(
        first["dhash"],
        second["dhash"]
    )

    flipped_p = hamming_distance(
        first["phash"],
        second["flip_phash"]
    )

    flipped_d = hamming_distance(
        first["dhash"],
        second["flip_dhash"]
    )

    if (
        flipped_p + flipped_d
        <
        normal_p + normal_d
    ):
        return (
            flipped_p,
            flipped_d,
            "horizontal_flip"
        )

    return (
        normal_p,
        normal_d,
        "normal"
    )


def preload(paths, label):
    records = []

    print(
        f"\nProcessing {label}..."
    )

    total = len(paths)

    for index, path in enumerate(
        paths,
        start=1
    ):
        try:
            record = {
                "path":
                    path,

                "split":
                    get_split(path),

                "sha256":
                    sha256_hash(path),

                "hashes":
                    image_hashes(path)
            }

            records.append(
                record
            )

        except Exception as error:
            print(
                f"Could not process "
                f"{path}: {error}"
            )

        if (
            index % 50 == 0
            or index == total
        ):
            print(
                f"  {index}/{total}"
            )

    return records


def compare_old_to_new(
    old_records,
    new_records
):
    matches = []

    print(
        "\nComparing ORIGINAL dataset "
        "against FINAL PROJECT dataset..."
    )

    exact_lookup = defaultdict(
        list
    )

    for new in new_records:
        exact_lookup[
            new["sha256"]
        ].append(
            new
        )

    exact_pairs = set()

    # --------------------------------------------------
    # EXACT DUPLICATES
    # --------------------------------------------------

    for old in old_records:
        for new in exact_lookup.get(
            old["sha256"],
            []
        ):
            pair = (
                str(old["path"]),
                str(new["path"])
            )

            exact_pairs.add(
                pair
            )

            matches.append({
                "type":
                    "EXACT",

                "old_split":
                    old["split"],

                "new_split":
                    new["split"],

                "old_image":
                    str(
                        old["path"]
                    ),

                "new_image":
                    str(
                        new["path"]
                    ),

                "phash_distance":
                    0,

                "dhash_distance":
                    0,

                "orientation":
                    "identical"
            })

    # --------------------------------------------------
    # PERCEPTUAL DUPLICATES
    # --------------------------------------------------

    total_pairs = (
        len(old_records)
        *
        len(new_records)
    )

    checked = 0

    for old_index, old in enumerate(
        old_records,
        start=1
    ):
        for new in new_records:
            pair = (
                str(old["path"]),
                str(new["path"])
            )

            if pair in exact_pairs:
                continue

            p_distance, d_distance, orientation = (
                perceptual_distance(
                    old["hashes"],
                    new["hashes"]
                )
            )

            if (
                p_distance
                <= PHASH_THRESHOLD
                and
                d_distance
                <= DHASH_THRESHOLD
            ):
                matches.append({
                    "type":
                        "NEAR",

                    "old_split":
                        old["split"],

                    "new_split":
                        new["split"],

                    "old_image":
                        str(
                            old["path"]
                        ),

                    "new_image":
                        str(
                            new["path"]
                        ),

                    "phash_distance":
                        p_distance,

                    "dhash_distance":
                        d_distance,

                    "orientation":
                        orientation
                })

            checked += 1

        if (
            old_index % 25 == 0
            or
            old_index == len(
                old_records
            )
        ):
            print(
                f"  Old images checked: "
                f"{old_index}/"
                f"{len(old_records)}"
            )

    return matches


def check_new_cross_split(
    new_records
):
    matches = []

    print(
        "\nChecking FINAL PROJECT dataset "
        "for perceptual cross-split leakage..."
    )

    split_groups = defaultdict(
        list
    )

    for record in new_records:
        split_groups[
            record["split"]
        ].append(
            record
        )

    split_pairs = [
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test")
    ]

    for first_split, second_split in split_pairs:

        print(
            f"  {first_split} "
            f"vs {second_split}"
        )

        first_records = (
            split_groups[
                first_split
            ]
        )

        second_records = (
            split_groups[
                second_split
            ]
        )

        for first in first_records:
            for second in second_records:

                if (
                    first["sha256"]
                    ==
                    second["sha256"]
                ):
                    matches.append({
                        "type":
                            "NEW_EXACT_CROSS_SPLIT",

                        "old_split":
                            first_split,

                        "new_split":
                            second_split,

                        "old_image":
                            str(
                                first["path"]
                            ),

                        "new_image":
                            str(
                                second["path"]
                            ),

                        "phash_distance":
                            0,

                        "dhash_distance":
                            0,

                        "orientation":
                            "identical"
                    })

                    continue

                p_distance, d_distance, orientation = (
                    perceptual_distance(
                        first["hashes"],
                        second["hashes"]
                    )
                )

                if (
                    p_distance
                    <= PHASH_THRESHOLD
                    and
                    d_distance
                    <= DHASH_THRESHOLD
                ):
                    matches.append({
                        "type":
                            "NEW_NEAR_CROSS_SPLIT",

                        "old_split":
                            first_split,

                        "new_split":
                            second_split,

                        "old_image":
                            str(
                                first["path"]
                            ),

                        "new_image":
                            str(
                                second["path"]
                            ),

                        "phash_distance":
                            p_distance,

                        "dhash_distance":
                            d_distance,

                        "orientation":
                            orientation
                    })

    return matches


def save_report(
    matches
):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    matches = sorted(
        matches,
        key=lambda row: (
            row["type"],
            int(
                row[
                    "phash_distance"
                ]
            ),
            int(
                row[
                    "dhash_distance"
                ]
            )
        )
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "type",
                "old_split",
                "new_split",
                "phash_distance",
                "dhash_distance",
                "orientation",
                "old_image",
                "new_image"
            ]
        )

        writer.writeheader()

        writer.writerows(
            matches
        )


def print_summary(matches):
    counts = defaultdict(
        int
    )

    for match in matches:
        counts[
            match["type"]
        ] += 1

    print(
        "\n"
        + "=" * 72
    )

    print(
        "DUPLICATE ANALYSIS SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        f"Exact OLD ↔ NEW duplicates: "
        f"{counts['EXACT']}"
    )

    print(
        f"Near OLD ↔ NEW duplicates:  "
        f"{counts['NEAR']}"
    )

    print(
        f"Exact cross-split duplicates "
        f"inside NEW dataset: "
        f"{counts['NEW_EXACT_CROSS_SPLIT']}"
    )

    print(
        f"Near cross-split duplicates "
        f"inside NEW dataset: "
        f"{counts['NEW_NEAR_CROSS_SPLIT']}"
    )

    print(
        "\nReport saved to:"
    )

    print(
        OUTPUT_FILE
    )


def main():

    print(
        "=" * 72
    )

    print(
        "DATASET DUPLICATE / LEAKAGE CHECK"
    )

    print(
        "=" * 72
    )

    old_paths = collect_images(
        OLD_ROOT
    )

    new_paths = collect_images(
        NEW_ROOT
    )

    print(
        f"\nOriginal dataset images: "
        f"{len(old_paths)}"
    )

    print(
        f"Final Project images:    "
        f"{len(new_paths)}"
    )

    if len(old_paths) == 0:
        raise RuntimeError(
            "No images found in data/raw"
        )

    if len(new_paths) == 0:
        raise RuntimeError(
            "No images found in Final Project dataset"
        )

    old_records = preload(
        old_paths,
        "original dataset"
    )

    new_records = preload(
        new_paths,
        "Final Project dataset"
    )

    old_new_matches = (
        compare_old_to_new(
            old_records,
            new_records
        )
    )

    new_split_matches = (
        check_new_cross_split(
            new_records
        )
    )

    all_matches = (
        old_new_matches
        +
        new_split_matches
    )

    save_report(
        all_matches
    )

    print_summary(
        all_matches
    )


if __name__ == "__main__":
    main()