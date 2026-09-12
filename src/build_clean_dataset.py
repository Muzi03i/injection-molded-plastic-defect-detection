import csv
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_ROOT = Path(
    "new_dataset/Final Project.v1i.coco"
)

OUTPUT_ROOT = Path(
    "data/final_clean"
)

SOURCE_SPLITS = [
    "train",
    "valid",
    "test"
]

CLASS_NAMES = [
    "Burn Mark",
    "Flash",
    "Short Shot",
    "Sink Mark"
]

NORMALIZED_CLASSES = [
    "burn mark",
    "flash",
    "short shot",
    "sink mark"
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name
    in enumerate(NORMALIZED_CLASSES)
}

SPLIT_RATIOS = {
    "train": 0.70,
    "valid": 0.15,
    "test": 0.15
}

RANDOM_SEED = 42


# Very strict near-duplicate thresholds.
# These are intentionally stricter than the earlier
# exploratory duplicate search.
PHASH_THRESHOLD = 4
DHASH_THRESHOLD = 6
COSINE_THRESHOLD = 0.985


# ============================================================
# CLASS-NAME NORMALIZATION
# ============================================================

def normalize_class_name(name):
    name = name.lower().strip()

    name = re.sub(
        r"[_-]+",
        " ",
        name
    )

    name = re.sub(
        r"^defect\s+",
        "",
        name
    )

    name = re.sub(
        r"\s+",
        " ",
        name
    )

    return name.strip()


# ============================================================
# LOAD COCO DATASET
# ============================================================

def load_records():
    records = []

    print(
        "=" * 72
    )

    print(
        "LOADING FINAL PROJECT DATASET"
    )

    print(
        "=" * 72
    )

    record_id = 0

    for source_split in SOURCE_SPLITS:

        split_folder = (
            SOURCE_ROOT
            / source_split
        )

        annotation_path = (
            split_folder
            / "_annotations.coco.json"
        )

        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Missing annotation file: "
                f"{annotation_path}"
            )

        with open(
            annotation_path,
            "r",
            encoding="utf-8"
        ) as file:
            coco = json.load(file)

        categories = {
            category["id"]:
                normalize_class_name(
                    category["name"]
                )
            for category
            in coco.get(
                "categories",
                []
            )
        }

        image_labels = defaultdict(
            set
        )

        for annotation in coco.get(
            "annotations",
            []
        ):
            category_name = (
                categories.get(
                    annotation[
                        "category_id"
                    ]
                )
            )

            if (
                category_name
                in CLASS_TO_INDEX
            ):
                image_labels[
                    annotation[
                        "image_id"
                    ]
                ].add(
                    category_name
                )

        split_count = 0

        for image_info in coco.get(
            "images",
            []
        ):
            filename = (
                image_info[
                    "file_name"
                ]
            )

            source_path = (
                split_folder
                / filename
            )

            if not source_path.exists():
                print(
                    "WARNING: Missing image: "
                    f"{source_path}"
                )

                continue

            labels = np.zeros(
                len(CLASS_NAMES),
                dtype=np.int32
            )

            for class_name in (
                image_labels.get(
                    image_info["id"],
                    set()
                )
            ):
                labels[
                    CLASS_TO_INDEX[
                        class_name
                    ]
                ] = 1

            records.append({
                "record_id":
                    record_id,

                "source_split":
                    source_split,

                "source_path":
                    source_path,

                "original_filename":
                    filename,

                "labels":
                    labels
            })

            record_id += 1
            split_count += 1

        print(
            f"{source_split:5s}: "
            f"{split_count} images"
        )

    print(
        f"\nTotal images loaded: "
        f"{len(records)}"
    )

    return records


# ============================================================
# HASH FUNCTIONS
# ============================================================

def sha256_file(path):
    hasher = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as file:

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
        transformed[
            :8,
            :8
        ]
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

    differences = (
        gray[:, 1:]
        >
        gray[:, :-1]
    )

    return differences.flatten().astype(
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

def calculate_features(records):
    print(
        "\n"
        + "=" * 72
    )

    print(
        "CALCULATING IMAGE FEATURES"
    )

    print(
        "=" * 72
    )

    total = len(records)

    for index, record in enumerate(
        records,
        start=1
    ):
        path = record[
            "source_path"
        ]

        record[
            "sha256"
        ] = sha256_file(
            path
        )

        with Image.open(
            path
        ) as image:

            image = (
                image.convert(
                    "RGB"
                )
            )

            mirrored = (
                ImageOps.mirror(
                    image
                )
            )

            record[
                "phash"
            ] = phash(
                image
            )

            record[
                "dhash"
            ] = dhash(
                image
            )

            record[
                "vector"
            ] = similarity_vector(
                image
            )

            record[
                "flip_phash"
            ] = phash(
                mirrored
            )

            record[
                "flip_dhash"
            ] = dhash(
                mirrored
            )

            record[
                "flip_vector"
            ] = similarity_vector(
                mirrored
            )

        if (
            index % 50 == 0
            or
            index == total
        ):
            print(
                f"{index}/{total}"
            )


# ============================================================
# UNION-FIND FOR DUPLICATE GROUPS
# ============================================================

class UnionFind:

    def __init__(
        self,
        size
    ):
        self.parent = list(
            range(size)
        )

        self.rank = [
            0
        ] * size

    def find(
        self,
        item
    ):
        if (
            self.parent[item]
            != item
        ):
            self.parent[item] = (
                self.find(
                    self.parent[item]
                )
            )

        return self.parent[item]

    def union(
        self,
        first,
        second
    ):
        root_first = self.find(
            first
        )

        root_second = self.find(
            second
        )

        if (
            root_first
            ==
            root_second
        ):
            return

        if (
            self.rank[root_first]
            <
            self.rank[root_second]
        ):
            self.parent[
                root_first
            ] = root_second

        elif (
            self.rank[root_first]
            >
            self.rank[root_second]
        ):
            self.parent[
                root_second
            ] = root_first

        else:
            self.parent[
                root_second
            ] = root_first

            self.rank[
                root_first
            ] += 1


# ============================================================
# STRONG NEAR-DUPLICATE TEST
# ============================================================

def strong_near_duplicate(
    first,
    second
):
    # Normal orientation
    p_distance = hamming_distance(
        first["phash"],
        second["phash"]
    )

    d_distance = hamming_distance(
        first["dhash"],
        second["dhash"]
    )

    if (
        p_distance
        <= PHASH_THRESHOLD
        and
        d_distance
        <= DHASH_THRESHOLD
    ):
        similarity = (
            cosine_similarity(
                first["vector"],
                second["vector"]
            )
        )

        if (
            similarity
            >= COSINE_THRESHOLD
        ):
            return True

    # Horizontal flip
    p_distance = hamming_distance(
        first["phash"],
        second[
            "flip_phash"
        ]
    )

    d_distance = hamming_distance(
        first["dhash"],
        second[
            "flip_dhash"
        ]
    )

    if (
        p_distance
        <= PHASH_THRESHOLD
        and
        d_distance
        <= DHASH_THRESHOLD
    ):
        similarity = (
            cosine_similarity(
                first["vector"],
                second[
                    "flip_vector"
                ]
            )
        )

        if (
            similarity
            >= COSINE_THRESHOLD
        ):
            return True

    return False


# ============================================================
# BUILD SIMILARITY GROUPS
# ============================================================

def build_similarity_groups(
    records
):
    print(
        "\n"
        + "=" * 72
    )

    print(
        "GROUPING EXACT / STRONG NEAR-DUPLICATES"
    )

    print(
        "=" * 72
    )

    count = len(records)

    union_find = UnionFind(
        count
    )

    exact_pairs = 0
    near_pairs = 0

    for first_index in range(
        count
    ):
        first = records[
            first_index
        ]

        for second_index in range(
            first_index + 1,
            count
        ):
            second = records[
                second_index
            ]

            if (
                first["sha256"]
                ==
                second["sha256"]
            ):
                union_find.union(
                    first_index,
                    second_index
                )

                exact_pairs += 1
                continue

            if strong_near_duplicate(
                first,
                second
            ):
                union_find.union(
                    first_index,
                    second_index
                )

                near_pairs += 1

        if (
            (first_index + 1)
            % 50 == 0
            or
            first_index + 1
            == count
        ):
            print(
                f"Compared "
                f"{first_index + 1}/"
                f"{count} images"
            )

    groups = defaultdict(
        list
    )

    for index in range(
        count
    ):
        root = union_find.find(
            index
        )

        groups[
            root
        ].append(
            index
        )

    group_list = list(
        groups.values()
    )

    multi_image_groups = [
        group
        for group in group_list
        if len(group) > 1
    ]

    images_in_duplicate_groups = sum(
        len(group)
        for group
        in multi_image_groups
    )

    largest_group = max(
        len(group)
        for group
        in group_list
    )

    print(
        "\nSimilarity-group results:"
    )

    print(
        f"Exact duplicate pairs: "
        f"{exact_pairs}"
    )

    print(
        f"Strong near-duplicate pairs: "
        f"{near_pairs}"
    )

    print(
        f"Total groups: "
        f"{len(group_list)}"
    )

    print(
        f"Groups containing >1 image: "
        f"{len(multi_image_groups)}"
    )

    print(
        f"Images in multi-image groups: "
        f"{images_in_duplicate_groups}"
    )

    print(
        f"Largest group: "
        f"{largest_group} images"
    )

    return group_list


# ============================================================
# GROUP INFORMATION
# ============================================================

def prepare_group_information(
    groups,
    records
):
    group_information = []

    global_labels = np.zeros(
        len(CLASS_NAMES),
        dtype=np.int64
    )

    for record in records:
        global_labels += (
            record["labels"]
        )

    for group_id, indices in enumerate(
        groups
    ):
        label_counts = np.zeros(
            len(CLASS_NAMES),
            dtype=np.int64
        )

        for index in indices:
            label_counts += (
                records[index][
                    "labels"
                ]
            )

        # Rare-class priority
        rarity_score = 0.0

        for class_index in range(
            len(CLASS_NAMES)
        ):
            total = max(
                int(
                    global_labels[
                        class_index
                    ]
                ),
                1
            )

            rarity_score += (
                label_counts[
                    class_index
                ]
                /
                total
            )

        group_information.append({
            "group_id":
                group_id,

            "indices":
                indices,

            "size":
                len(indices),

            "label_counts":
                label_counts,

            "rarity_score":
                rarity_score
        })

    return (
        group_information,
        global_labels
    )


# ============================================================
# GROUP-SAFE SPLITTING
# ============================================================

def split_groups(
    groups,
    records,
    global_labels
):
    print(
        "\n"
        + "=" * 72
    )

    print(
        "CREATING GROUP-SAFE 70 / 15 / 15 SPLIT"
    )

    print(
        "=" * 72
    )

    random.seed(
        RANDOM_SEED
    )

    total_images = len(
        records
    )

    target_sizes = {
        split:
            total_images
            *
            ratio
        for split, ratio
        in SPLIT_RATIOS.items()
    }

    target_labels = {
        split:
            global_labels.astype(
                np.float64
            )
            *
            ratio
        for split, ratio
        in SPLIT_RATIOS.items()
    }

    current_sizes = {
        split: 0
        for split
        in SPLIT_RATIOS
    }

    current_labels = {
        split:
            np.zeros(
                len(CLASS_NAMES),
                dtype=np.float64
            )
        for split
        in SPLIT_RATIOS
    }

    assignments = {}

    # Random number only breaks exact ties.
    for group in groups:
        group[
            "tie_break"
        ] = random.random()

    # Put larger / rarer groups first.
    sorted_groups = sorted(
        groups,
        key=lambda group: (
            group["rarity_score"],
            group["size"],
            group["tie_break"]
        ),
        reverse=True
    )

    for group in sorted_groups:

        best_split = None
        best_score = None

        for split in SPLIT_RATIOS:

            target_size = max(
                target_sizes[
                    split
                ],
                1.0
            )

            size_need = max(
                target_size
                -
                current_sizes[
                    split
                ],
                0.0
            ) / target_size

            label_need_score = 0.0

            for class_index in range(
                len(CLASS_NAMES)
            ):
                group_count = (
                    group[
                        "label_counts"
                    ][
                        class_index
                    ]
                )

                if group_count == 0:
                    continue

                target_label = max(
                    target_labels[
                        split
                    ][
                        class_index
                    ],
                    1.0
                )

                remaining_label_need = max(
                    target_label
                    -
                    current_labels[
                        split
                    ][
                        class_index
                    ],
                    0.0
                )

                deficit_fraction = (
                    remaining_label_need
                    /
                    target_label
                )

                rarity_weight = (
                    group_count
                    /
                    max(
                        global_labels[
                            class_index
                        ],
                        1
                    )
                )

                label_need_score += (
                    deficit_fraction
                    *
                    rarity_weight
                )

            projected_size = (
                current_sizes[
                    split
                ]
                +
                group[
                    "size"
                ]
            )

            overfill = max(
                projected_size
                -
                target_size,
                0.0
            ) / target_size

            score = (
                1.0
                *
                size_need
                +
                2.5
                *
                label_need_score
                -
                3.0
                *
                overfill
            )

            if (
                best_score is None
                or
                score > best_score
            ):
                best_score = score
                best_split = split

        assignments[
            group[
                "group_id"
            ]
        ] = best_split

        current_sizes[
            best_split
        ] += group[
            "size"
        ]

        current_labels[
            best_split
        ] += group[
            "label_counts"
        ]

    print(
        "\nSplit results:"
    )

    for split in SPLIT_RATIOS:

        print(
            f"\n{split.upper()}"
        )

        print(
            f"Images: "
            f"{current_sizes[split]}"
        )

        for class_index, class_name in enumerate(
            CLASS_NAMES
        ):
            print(
                f"  {class_name:12s}: "
                f"{int(current_labels[split][class_index])}"
            )

    return assignments


# ============================================================
# CREATE FINAL DATASET
# ============================================================

def create_dataset(
    group_information,
    assignments,
    records
):
    if OUTPUT_ROOT.exists():
        raise RuntimeError(
            "\n"
            f"{OUTPUT_ROOT} already exists.\n"
            "The script will not overwrite it.\n"
            "Delete or rename that folder first "
            "if you intentionally want to rebuild."
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "WRITING CLEAN DATASET"
    )

    print(
        "=" * 72
    )

    for split in SPLIT_RATIOS:
        (
            OUTPUT_ROOT
            /
            split
        ).mkdir(
            parents=True,
            exist_ok=False
        )

    csv_rows = {
        split: []
        for split in SPLIT_RATIOS
    }

    manifest_rows = []

    split_counters = {
        split: 0
        for split in SPLIT_RATIOS
    }

    for group in group_information:

        group_id = (
            group[
                "group_id"
            ]
        )

        destination_split = (
            assignments[
                group_id
            ]
        )

        for record_index in (
            group[
                "indices"
            ]
        ):
            record = records[
                record_index
            ]

            split_counters[
                destination_split
            ] += 1

            suffix = (
                record[
                    "source_path"
                ].suffix.lower()
            )

            new_filename = (
                f"fp_"
                f"{destination_split}_"
                f"{split_counters[destination_split]:04d}"
                f"{suffix}"
            )

            destination_path = (
                OUTPUT_ROOT
                /
                destination_split
                /
                new_filename
            )

            shutil.copy2(
                record[
                    "source_path"
                ],
                destination_path
            )

            row = {
                "filename":
                    new_filename
            }

            for class_index, class_name in enumerate(
                CLASS_NAMES
            ):
                row[
                    class_name
                ] = int(
                    record[
                        "labels"
                    ][
                        class_index
                    ]
                )

            csv_rows[
                destination_split
            ].append(
                row
            )

            manifest_row = {
                "split":
                    destination_split,

                "group_id":
                    group_id,

                "new_filename":
                    new_filename,

                "source_split":
                    record[
                        "source_split"
                    ],

                "original_filename":
                    record[
                        "original_filename"
                    ],

                "source_path":
                    str(
                        record[
                            "source_path"
                        ]
                    )
            }

            for class_index, class_name in enumerate(
                CLASS_NAMES
            ):
                manifest_row[
                    class_name
                ] = int(
                    record[
                        "labels"
                    ][
                        class_index
                    ]
                )

            manifest_rows.append(
                manifest_row
            )

    # --------------------------------------------------------
    # WRITE PER-SPLIT CLASS CSV FILES
    # --------------------------------------------------------

    fieldnames = [
        "filename"
    ] + CLASS_NAMES

    for split, rows in csv_rows.items():

        csv_path = (
            OUTPUT_ROOT
            /
            split
            /
            "_classes.csv"
        )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                rows
            )

    # --------------------------------------------------------
    # WRITE MANIFEST
    # --------------------------------------------------------

    manifest_path = (
        OUTPUT_ROOT
        /
        "split_manifest.csv"
    )

    manifest_fields = [
        "split",
        "group_id",
        "new_filename",
        "source_split",
        "original_filename",
        "source_path"
    ] + CLASS_NAMES

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=manifest_fields
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )

    return csv_rows


# ============================================================
# VERIFY GROUP LEAKAGE
# ============================================================

def verify_no_group_leakage(
    group_information,
    assignments
):
    group_to_split = defaultdict(
        set
    )

    for group in group_information:
        group_id = (
            group[
                "group_id"
            ]
        )

        group_to_split[
            group_id
        ].add(
            assignments[
                group_id
            ]
        )

    bad_groups = [
        group_id
        for group_id, splits
        in group_to_split.items()
        if len(splits) > 1
    ]

    if bad_groups:
        raise RuntimeError(
            "Similarity-group leakage detected."
        )

    print(
        "\nSimilarity-group leakage check: PASS"
    )


# ============================================================
# WRITE SUMMARY
# ============================================================

def write_summary(
    csv_rows,
    group_information,
    global_labels
):
    summary_path = (
        OUTPUT_ROOT
        /
        "dataset_summary.txt"
    )

    lines = []

    lines.append(
        "FINAL CLEAN DATASET SUMMARY"
    )

    lines.append(
        "=" * 60
    )

    lines.append(
        ""
    )

    lines.append(
        f"Total images: "
        f"{sum(len(rows) for rows in csv_rows.values())}"
    )

    lines.append(
        f"Similarity groups: "
        f"{len(group_information)}"
    )

    lines.append(
        ""
    )

    lines.append(
        "Global target-label occurrences:"
    )

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        lines.append(
            f"  {class_name}: "
            f"{int(global_labels[class_index])}"
        )

    for split in [
        "train",
        "valid",
        "test"
    ]:
        rows = csv_rows[
            split
        ]

        lines.append(
            ""
        )

        lines.append(
            split.upper()
        )

        lines.append(
            f"Images: {len(rows)}"
        )

        for class_name in CLASS_NAMES:
            count = sum(
                int(
                    row[
                        class_name
                    ]
                )
                for row in rows
            )

            lines.append(
                f"  {class_name}: "
                f"{count}"
            )

        negative_count = sum(
            1
            for row in rows
            if sum(
                int(
                    row[
                        class_name
                    ]
                )
                for class_name
                in CLASS_NAMES
            )
            == 0
        )

        lines.append(
            f"  No selected target defect: "
            f"{negative_count}"
        )

    lines.append(
        ""
    )

    lines.append(
        "Split ratios requested:"
    )

    lines.append(
        "  Train: 70%"
    )

    lines.append(
        "  Validation: 15%"
    )

    lines.append(
        "  Test: 15%"
    )

    lines.append(
        ""
    )

    lines.append(
        "Duplicate grouping thresholds:"
    )

    lines.append(
        f"  pHash <= {PHASH_THRESHOLD}"
    )

    lines.append(
        f"  dHash <= {DHASH_THRESHOLD}"
    )

    lines.append(
        f"  cosine similarity >= "
        f"{COSINE_THRESHOLD}"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "\n".join(
                lines
            )
        )

    print(
        "\n"
        + "\n".join(
            lines
        )
    )

    print(
        "\nSummary saved to:"
    )

    print(
        summary_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    records = load_records()

    if len(records) != 975:
        print(
            "\nWARNING:"
        )

        print(
            "Expected approximately 975 images, "
            f"but loaded {len(records)}."
        )

    calculate_features(
        records
    )

    similarity_groups = (
        build_similarity_groups(
            records
        )
    )

    (
        group_information,
        global_labels
    ) = prepare_group_information(
        similarity_groups,
        records
    )

    assignments = split_groups(
        group_information,
        records,
        global_labels
    )

    verify_no_group_leakage(
        group_information,
        assignments
    )

    csv_rows = create_dataset(
        group_information,
        assignments,
        records
    )

    write_summary(
        csv_rows,
        group_information,
        global_labels
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "CLEAN DATASET BUILD COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "\nCreated:"
    )

    print(
        OUTPUT_ROOT
    )

    print(
        "\nOriginal data/raw was NOT modified."
    )


if __name__ == "__main__":
    main()