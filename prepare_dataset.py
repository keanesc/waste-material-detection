"""
Converts COCO format to YOLO format and creates simplified class mappings.
Includes dataset splitting and duplicate removal.
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def create_class_mapping():
    plastic_items = {
        4,
        5,
        7,
        21,
        24,
        27,
        29,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        47,
        48,
        49,
        54,
        55,
        57,
    }
    paper_items = {13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 33, 34, 35, 56}
    metal_items = {0, 1, 8, 10, 11, 12, 28, 50, 52}

    return {"plastic": plastic_items, "paper": paper_items, "metal": metal_items}


def simplify_annotations(input_json, output_json):
    output_path = Path(output_json)

    if output_path.exists():
        print(f"Simplified annotations already exist at {output_json}")
        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(
            f"Found {len(data['categories'])} categories, {len(data['images'])} images"
        )
        return output_json

    print("Creating simplified annotations...")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    mappings = create_class_mapping()

    old_to_new = {}
    for old_cat in data["categories"]:
        old_id = old_cat["id"]
        if old_id in mappings["plastic"]:
            old_to_new[old_id] = 1
        elif old_id in mappings["paper"]:
            old_to_new[old_id] = 2
        elif old_id in mappings["metal"]:
            old_to_new[old_id] = 3
        else:
            old_to_new[old_id] = 0

    new_categories = [
        {"id": 0, "name": "general", "supercategory": "waste"},
        {"id": 1, "name": "plastic", "supercategory": "waste"},
        {"id": 2, "name": "paper", "supercategory": "waste"},
        {"id": 3, "name": "metal", "supercategory": "waste"},
    ]

    for ann in data["annotations"]:
        ann["category_id"] = old_to_new[ann["category_id"]]

    data["categories"] = new_categories

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"Created simplified annotations with {len(new_categories)} categories")
    print(f"Saved to {output_json}")

    return output_json


def convert_to_yolo(annotations_json, taco_dir, output_dir):
    """
    Manually converts annotations and symlinks images.
    """

    output_path = Path(output_dir)
    labels_dir = output_path / "labels"
    images_dir = output_path / "images"

    if (
        labels_dir.exists()
        and any(labels_dir.glob("*.txt"))
        and images_dir.exists()
        and any(images_dir.iterdir())
    ):
        label_count = len(list(labels_dir.glob("*.txt")))
        image_count = len(list(images_dir.glob("*.*")))
        print(f"YOLO format data already exists at {output_dir}")
        print(f"Found {label_count} label files and {image_count} images")
        return output_dir

    print("Converting COCO format to YOLO format...")

    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(annotations_json, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images_dict = {img["id"]: img for img in coco_data["images"]}

    annotations_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    converted_images = 0
    converted_annotations = 0

    for img_id, img_info in images_dict.items():
        img_width = img_info["width"]
        img_height = img_info["height"]
        img_filename = Path(img_info["file_name"])

        label_filename = img_filename.stem + ".txt"
        label_path = labels_dir / label_filename

        image_annotations = annotations_by_image.get(img_id, [])

        if image_annotations:
            with open(label_path, "w", encoding="utf-8") as f:
                for ann in image_annotations:
                    # COCO format: [x_min, y_min, width, height]
                    # YOLO format: [class_id, x_center, y_center, width, height] (normalized)

                    category_id = ann["category_id"]
                    bbox = ann["bbox"]

                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height

                    f.write(
                        f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    )
                    converted_annotations += 1

            converted_images += 1

            src_image_path = taco_dir / img_info["file_name"]
            dst_image_path = images_dir / img_filename.name

            if src_image_path.exists() and not dst_image_path.exists():
                dst_image_path.symlink_to(src_image_path)

    print("Conversion complete")
    print(
        f"Created {converted_images} label files with {converted_annotations} annotations"
    )
    print(f"Linked {converted_images} images")
    print(f"Output saved to {output_dir}")

    return output_dir


def remove_duplicate_images(output_dir):
    """
    Remove duplicate image files (keep one version when both .JPG and .jpg exist).
    This fixes the issue where images have multiple extensions but only one label file.
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"

    if not images_dir.exists():
        print("No images directory found, skipping duplicate removal")
        return 0

    image_files = [
        f
        for f in images_dir.glob("*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    stem_to_files = defaultdict(list)
    for img in image_files:
        stem_to_files[img.stem].append(img)

    duplicates_removed = 0
    for _stem, files in stem_to_files.items():
        if len(files) > 1:
            files.sort(key=lambda x: x.suffix)

            to_remove = files[1:]

            for file_to_remove in to_remove:
                file_to_remove.unlink()
                duplicates_removed += 1

    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate image files")
    else:
        print("No duplicate images found")

    return duplicates_removed


def split_dataset(
    output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42
):
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"

    if (images_dir / "train").exists() and (images_dir / "val").exists():
        print("Dataset already split into train/val/test")
        return

    print("Splitting dataset into train/val/test...")

    random.seed(seed)

    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(exist_ok=True)
        (labels_dir / split).mkdir(exist_ok=True)

    image_files = [
        f
        for f in images_dir.glob("*")
        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    if not image_files:
        print("No images to split (already split or not converted yet)")
        return

    print(f"Found {len(image_files)} images to split")

    random.shuffle(image_files)

    total = len(image_files)
    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    print(f"Train: {len(train_files)} images")
    print(f"Val: {len(val_files)} images")
    print(f"Test: {len(test_files)} images")

    # Function to move files
    def move_files(file_list, split_name):
        moved_images = 0
        moved_labels = 0

        for img_path in file_list:
            # Move image
            dest_img = images_dir / split_name / img_path.name
            shutil.move(str(img_path), str(dest_img))
            moved_images += 1

            # Move corresponding label
            label_name = img_path.stem + ".txt"
            label_path = labels_dir / label_name

            if label_path.exists():
                dest_label = labels_dir / split_name / label_name
                shutil.move(str(label_path), str(dest_label))
                moved_labels += 1

        return moved_images, moved_labels

    # Move files to respective directories
    move_files(train_files, "train")
    move_files(val_files, "val")
    move_files(test_files, "test")

    print("✓ Dataset split complete!")


def regenerate_missing_labels(output_dir, annotations_json):
    output_path = Path(output_dir)

    with open(annotations_json, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    stem_to_image = {}
    for img in coco_data["images"]:
        stem = Path(img["file_name"]).stem
        stem_to_image[stem] = img

    annotations_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    total_created = 0

    for split in ["train", "val", "test"]:
        images_dir = output_path / "images" / split
        labels_dir = output_path / "labels" / split

        if not images_dir.exists():
            continue

        image_files = [
            f
            for f in images_dir.glob("*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        created_in_split = 0

        for img_file in image_files:
            stem = img_file.stem
            label_file = labels_dir / f"{stem}.txt"

            if label_file.exists():
                continue

            if stem not in stem_to_image:
                continue

            img_data = stem_to_image[stem]
            img_id = img_data["id"]
            img_width = img_data["width"]
            img_height = img_data["height"]

            image_annotations = annotations_by_image.get(img_id, [])

            if not image_annotations:
                continue

            with open(label_file, "w", encoding="utf-8") as f:
                for ann in image_annotations:
                    category_id = ann["category_id"]
                    bbox = ann["bbox"]

                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height

                    f.write(
                        f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    )

            created_in_split += 1

        total_created += created_in_split

    if total_created > 0:
        print(f"Regenerated {total_created} missing label files")
    else:
        print("No missing labels found")

    return total_created


def create_data_yaml(output_dir, dataset_name="waste"):
    yaml_path = Path(output_dir) / f"{dataset_name}.yaml"

    yaml_content = f"""path: {output_dir}
train: images/train
val: images/val
test: images/test

names:
  0: general
  1: plastic
  2: paper
  3: metal
"""

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Created dataset config at {yaml_path}")

    return yaml_path


def main():
    project_root = Path(__file__).parent
    taco_dir = project_root / "TACO" / "data"
    original_annotations = taco_dir / "annotations.json"
    simplified_annotations = project_root / "data" / "annotations_simplified.json"
    yolo_output_dir = project_root / "data" / "yolo_dataset"

    print("TACO Dataset Preparation for YOLOv8")

    print("\n[1/6] Simplifying annotations...")
    simplify_annotations(original_annotations, simplified_annotations)

    print("\n[2/6] Converting to YOLO format...")
    convert_to_yolo(simplified_annotations, taco_dir, yolo_output_dir)

    print("\n[3/6] Removing duplicate images...")
    remove_duplicate_images(yolo_output_dir)

    print("\n[4/6] Splitting dataset...")
    split_dataset(yolo_output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

    print("\n[5/6] Checking for missing labels...")
    regenerate_missing_labels(yolo_output_dir, simplified_annotations)

    print("\n[6/6] Creating dataset configuration...")
    yaml_path = create_data_yaml(yolo_output_dir)

    print("\nDataset Statistics")

    for split in ["train", "val", "test"]:
        images_dir = yolo_output_dir / "images" / split
        labels_dir = yolo_output_dir / "labels" / split

        if images_dir.exists():
            image_count = len(
                [
                    f
                    for f in images_dir.glob("*")
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ]
            )
            label_count = len(list(labels_dir.glob("*.txt")))
            print(
                f"{split.upper():5s}: {image_count:3d} images, {label_count:3d} labels"
            )

    print("\nDataset preparation complete")
    print(f"Dataset location: {yolo_output_dir}")
    print(f"Config file: {yaml_path}")
    print("\nNext: Run 'python train_waste_detector.py' to start training")

    return yaml_path


if __name__ == "__main__":
    main()
