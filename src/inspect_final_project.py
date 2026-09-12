import json
import re
from pathlib import Path
from collections import Counter, defaultdict


DATASET_ROOT = Path(
    "new_dataset/Final Project.v1i.coco"
)

SPLITS = [
    "train",
    "valid",
    "test"
]

TARGET_CLASSES = {
    "burn mark",
    "flash",
    "short shot",
    "sink mark"
}


def normalize_class_name(name):
    name = name.lower().strip()

    # Treat underscores and hyphens as spaces
    name = re.sub(
        r"[_-]+",
        " ",
        name
    )

    # Remove 'defect' prefix
    name = re.sub(
        r"^defect\s+",
        "",
        name
    )

    # Remove repeated spaces
    name = re.sub(
        r"\s+",
        " ",
        name
    )

    return name.strip()


def get_original_base(filename):
    """
    Convert a Roboflow-generated filename such as:

    example_png.rf.123456.jpg

    into:

    example_png
    """

    name = Path(filename).stem

    if ".rf." in name:
        name = name.split(".rf.")[0]

    return name


def inspect_split(split):
    annotation_path = (
        DATASET_ROOT
        / split
        / "_annotations.coco.json"
    )

    with open(
        annotation_path,
        "r",
        encoding="utf-8"
    ) as file:
        coco = json.load(file)

    images = coco.get(
        "images",
        []
    )

    annotations = coco.get(
        "annotations",
        []
    )

    categories = coco.get(
        "categories",
        []
    )

    category_id_to_name = {
        category["id"]:
            normalize_class_name(
                category["name"]
            )
        for category in categories
    }

    image_id_to_filename = {
        image["id"]:
            image["file_name"]
        for image in images
    }

    # Classes occurring in each generated image
    image_classes = defaultdict(set)

    for annotation in annotations:
        category_name = (
            category_id_to_name[
                annotation["category_id"]
            ]
        )

        image_classes[
            annotation["image_id"]
        ].add(
            category_name
        )

    # --------------------------------------------------
    # GENERATED IMAGE COUNTS
    # --------------------------------------------------

    generated_class_counts = Counter()

    for class_set in image_classes.values():
        for class_name in class_set:
            generated_class_counts[
                class_name
            ] += 1

    # --------------------------------------------------
    # SOURCE IMAGE COUNTS
    # --------------------------------------------------

    source_classes = defaultdict(set)

    for image in images:
        image_id = image["id"]

        source_name = get_original_base(
            image["file_name"]
        )

        source_classes[
            source_name
        ].update(
            image_classes.get(
                image_id,
                set()
            )
        )

    source_class_counts = Counter()

    for class_set in source_classes.values():
        for class_name in class_set:
            source_class_counts[
                class_name
            ] += 1

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"SPLIT: {split.upper()}"
    )

    print(
        "=" * 72
    )

    print(
        f"Generated images:      {len(images)}"
    )

    print(
        f"COCO annotations:      {len(annotations)}"
    )

    print(
        f"Unique source images:  {len(source_classes)}"
    )

    print(
        "\nNormalized categories:"
    )

    normalized_categories = sorted(
        set(
            category_id_to_name.values()
        )
    )

    for class_name in normalized_categories:
        print(
            f"  {class_name}"
        )

    print(
        "\nGenerated image count per class:"
    )

    for class_name, count in sorted(
        generated_class_counts.items()
    ):
        print(
            f"  {class_name:20s}: "
            f"{count}"
        )

    print(
        "\nUNIQUE SOURCE IMAGE COUNT PER CLASS:"
    )

    for class_name, count in sorted(
        source_class_counts.items()
    ):
        print(
            f"  {class_name:20s}: "
            f"{count}"
        )

    print(
        "\nTARGET CLASSES:"
    )

    for target in sorted(
        TARGET_CLASSES
    ):
        generated_count = (
            generated_class_counts.get(
                target,
                0
            )
        )

        source_count = (
            source_class_counts.get(
                target,
                0
            )
        )

        print(
            f"  {target:15s} | "
            f"Generated: {generated_count:3d} | "
            f"Sources: {source_count:3d}"
        )

    target_sources = {
        source_name
        for source_name, classes
        in source_classes.items()
        if classes & TARGET_CLASSES
    }

    print(
        "\nUnique source images containing "
        "at least one target class: "
        f"{len(target_sources)}"
    )

    return {
        "source_names":
            set(source_classes.keys())
    }


def check_split_overlap(split_data):
    print(
        "\n"
        + "=" * 72
    )

    print(
        "SOURCE-IMAGE OVERLAP BETWEEN SPLITS"
    )

    print(
        "=" * 72
    )

    pairs = [
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test")
    ]

    total_overlap = 0

    for first, second in pairs:
        overlap = (
            split_data[first][
                "source_names"
            ]
            &
            split_data[second][
                "source_names"
            ]
        )

        total_overlap += len(
            overlap
        )

        print(
            f"{first:5s} vs "
            f"{second:5s}: "
            f"{len(overlap)}"
        )

    if total_overlap == 0:
        print(
            "\nNo apparent source-image "
            "leakage between splits."
        )
    else:
        print(
            "\nWARNING: Possible source-image "
            "leakage exists."
        )


def main():
    split_data = {}

    for split in SPLITS:
        split_data[split] = (
            inspect_split(
                split
            )
        )

    check_split_overlap(
        split_data
    )


if __name__ == "__main__":
    main()