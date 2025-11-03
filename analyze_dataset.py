"""
Analyze dataset for class imbalance and quality issues.
"""

from collections import Counter
from pathlib import Path


def analyze_class_distribution(dataset_dir):
    dataset_path = Path(dataset_dir)

    class_names = {0: "general", 1: "plastic", 2: "paper", 3: "metal"}

    for split in ["train", "val", "test"]:
        labels_dir = dataset_path / "labels" / split

        if not labels_dir.exists():
            continue

        class_counts = Counter()
        total_boxes = 0
        total_files = 0
        files_per_class = {i: 0 for i in range(4)}
        empty_files = 0

        for label_file in labels_dir.glob("*.txt"):
            total_files += 1
            file_classes = set()

            with open(label_file, "r") as f:
                lines = f.readlines()

            if not lines:
                empty_files += 1
                continue

            for line in lines:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    file_classes.add(class_id)
                    total_boxes += 1

            for cls in file_classes:
                files_per_class[cls] += 1

        print(f"\n{'=' * 60}")
        print(f"{split.upper()} SET ANALYSIS")
        print(f"{'=' * 60}")
        print(f"Total files: {total_files}")
        print(f"Empty files: {empty_files}")
        print(f"Total bounding boxes: {total_boxes}")
        print(
            f"Average boxes per image: {total_boxes / max(total_files - empty_files, 1):.2f}"
        )

        print("\nClass Distribution (by bounding boxes):")
        print(f"{'Class':<15} {'Count':>8} {'Percentage':>12} {'Images':>10}")
        print("-" * 50)

        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
            images = files_per_class[class_id]
            print(
                f"{class_names[class_id]:<15} {count:>8} {percentage:>11.1f}% {images:>10}"
            )

        # Check for class imbalance
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / max(min_count, 1)

            print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}x")
            if imbalance_ratio > 10:
                print("WARNING: Severe class imbalance detected!")
                print("   Consider using class weights or resampling.")
            elif imbalance_ratio > 5:
                print("WARNING: Moderate class imbalance detected.")


def analyze_bbox_sizes(dataset_dir):
    """Analyze bounding box sizes to detect small objects."""
    dataset_path = Path(dataset_dir)

    print(f"\n{'=' * 60}")
    print("BOUNDING BOX SIZE ANALYSIS")
    print(f"{'=' * 60}")

    for split in ["train", "val", "test"]:
        labels_dir = dataset_path / "labels" / split

        if not labels_dir.exists():
            continue

        small_boxes = 0  # < 2% of image area
        medium_boxes = 0  # 2-10%
        large_boxes = 0  # > 10%
        total = 0

        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        width = float(parts[3])
                        height = float(parts[4])
                        area = width * height

                        total += 1
                        if area < 0.02:
                            small_boxes += 1
                        elif area < 0.10:
                            medium_boxes += 1
                        else:
                            large_boxes += 1

        if total > 0:
            print(f"\n{split.upper()}:")
            print(
                f"  Small boxes (< 2%):   {small_boxes:>4} ({small_boxes / total * 100:>5.1f}%)"
            )
            print(
                f"  Medium boxes (2-10%): {medium_boxes:>4} ({medium_boxes / total * 100:>5.1f}%)"
            )
            print(
                f"  Large boxes (> 10%):  {large_boxes:>4} ({large_boxes / total * 100:>5.1f}%)"
            )

            if small_boxes / total > 0.5:
                print(
                    "  Many small objects - consider using multi-scale training"
                )


def main():
    project_root = Path(__file__).parent
    dataset_dir = project_root / "data" / "yolo_dataset"

    if not dataset_dir.exists():
        print(f"Error: Dataset not found at {dataset_dir}")
        print("Run 'python prepare_dataset.py' first")
        return

    print("\n" + "=" * 60)
    print("WASTE DETECTION DATASET ANALYSIS")
    print("=" * 60)

    analyze_class_distribution(dataset_dir)
    analyze_bbox_sizes(dataset_dir)

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR ACCURACY IMPROVEMENT")
    print("=" * 60)
    print("""
1. Dataset Size: Current dataset is small (~239 images)
   - Use full TACO dataset (1,500 images)
   - Enable data augmentation (hsv, rotation, flipping, mosaic)
   
2. Class Imbalance: Check results above
   - If severe, use class_weights in training
   - Or apply weighted loss
   
3. Small Objects: If many small objects detected
   - Use multi-scale training: imgsz=[480, 640, 800]
   - Increase image resolution to 800 or 1024
   
4. Model Selection:
   - Use YOLOv8m or YOLOv8l for better accuracy
   - Trade-off: slower inference but higher mAP
   
5. Training Epochs:
   - Increase from 30 to 100-200 epochs
   - Use early stopping with patience=50
   
6. Validation:
   - Monitor mAP50 and mAP50-95 metrics
   - Target: mAP50 > 0.5 (50%) for good performance
""")


if __name__ == "__main__":
    main()
