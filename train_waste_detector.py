from pathlib import Path

from ultralytics import YOLO  # type: ignore


def train_waste_detector(
    data_yaml,
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu",
    model_size="m",  # n, s, m, l, x
    resume=False,
):
    project_dir = Path("runs/detect")
    existing_runs = (
        list(project_dir.glob("waste_detector*")) if project_dir.exists() else []
    )

    if existing_runs and resume:
        latest_run = max(existing_runs, key=lambda p: p.stat().st_mtime)
        weights_path = latest_run / "weights" / "best.pt"

        if weights_path.exists():
            print(f"Found existing model at {weights_path}")
            print("Loading existing model...")
            model = YOLO(str(weights_path))
            print(
                "To retrain from scratch, set resume=False or delete runs/detect/waste_detector* directories"
            )
            return model, None

    print("Training model for Waste Detection")
    print(f"Loading pretrained YOLOv8{model_size} weights...")

    model_path = f"models/yolov8{model_size}.pt"
    model = YOLO(model_path)

    print("\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    print(f"  Dataset: {data_yaml}")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/detect",
        name="waste_detector",
        exist_ok=True,
        verbose=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Data augmentation
        hsv_h=0.015,  # HSV-Hue augmentation (0.0-1.0)
        hsv_s=0.7,  # HSV-Saturation augmentation (0.0-1.0)
        hsv_v=0.4,  # HSV-Value augmentation (0.0-1.0)
        degrees=10.0,  # Image rotation (+/- deg)
        translate=0.1,  # Image translation (+/- fraction)
        scale=0.5,  # Image scale (+/- gain)
        shear=2.0,  # Image shear (+/- deg)
        perspective=0.0,  # Image perspective (+/- fraction)
        flipud=0.0,  # Vertical flip probability
        fliplr=0.5,  # Horizontal flip probability
        mosaic=1.0,  # Mosaic augmentation probability
        mixup=0.15,  # MixUp augmentation probability
        copy_paste=0.1,  # Copy-paste augmentation probability
        # Advanced training settings
        cos_lr=True,  # Use cosine learning rate scheduler
        close_mosaic=10,  # Disable mosaic in last N epochs for better final accuracy
        patience=50,  # Early stopping patience (epochs without improvement)
        # Loss function weights (tune if class imbalance)
        box=7.5,  # Box loss gain
        cls=0.5,  # Class loss gain (reduce if many classes)
        dfl=1.5,  # DFL loss gain
        # Validation settings
        val=True,
        save=True,
        save_period=-1,  # Save checkpoint every N epochs (-1 to disable)
        plots=True,  # Generate training plots
    )

    print("Training complete")
    return model, results


def validate_model(model, data_yaml=None):
    print("Validation Metrics")

    if data_yaml:
        metrics = model.val(data=str(data_yaml))
    else:
        metrics = model.val()

    print(f"Precision:    {metrics.box.mp:.3f}")
    print(f"Recall:       {metrics.box.mr:.3f}")
    print(f"mAP50:        {metrics.box.map50:.3f}")
    print(f"mAP50-95:     {metrics.box.map:.3f}")

    print("\nMetrics Interpretation:")
    if metrics.box.map50 > 0.7:
        print("Excellent performance (mAP50 > 0.7)")
    elif metrics.box.map50 > 0.5:
        print("Good performance (mAP50 > 0.5)")
    elif metrics.box.map50 > 0.3:
        print("Fair performance (mAP50 > 0.3) - needs improvement")
    else:
        print("Poor performance (mAP50 < 0.3) - significant issues")

    return metrics


def test_model(model, data_yaml):
    print("\nEvaluating on test set...")
    metrics = model.val(split="test", data=str(data_yaml))

    print("Test metrics:")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

    return metrics


def predict_sample(model, image_path, conf=0.35, iou=0.45):
    if not Path(image_path).exists():
        print(f"Warning: Image not found at {image_path}")
        return None

    print("\nRunning inference...")
    print(f"Confidence threshold: {conf}")
    print(f"IoU threshold: {iou}")

    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        save=True,
        project="runs/detect",
        name="predictions",
        exist_ok=True,
    )

    if results:
        print(f"\nProcessed {len(results)} image(s)")

        class_names = model.names
        total_detections = 0

        for i, result in enumerate(results):
            if len(result.boxes) > 0:
                print(f"\nImage {i + 1}: {len(result.boxes)} objects")
                class_counts = {}
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = class_names[cls_id]
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    total_detections += 1

                for cls_name, count in sorted(class_counts.items()):
                    print(f"  {cls_name}: {count}")
            else:
                print(f"\nImage {i + 1}: No objects detected")

        print(f"\nTotal detections: {total_detections}")

    return results


def main():
    project_root = Path(__file__).parent
    data_yaml = project_root / "data" / "yolo_dataset" / "waste.yaml"

    if not data_yaml.exists():
        print("Error: Dataset not found!")
        print(f"Expected: {data_yaml}")
        print("\nRun 'python prepare_dataset.py' first")
        return

    model, results = train_waste_detector(
        data_yaml=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        device="cpu",
        model_size="m",
        resume=False,
    )

    # Validate on validation set
    validate_model(model, data_yaml)

    # Test on test set
    test_model(model, data_yaml)

    sample_images = project_root / "TACO" / "data" / "batch_1"
    if sample_images.exists():
        print("\nRunning inference on sample images...")
        predict_sample(model, sample_images, conf=0.35, iou=0.45)

    print("\nEnd")
    print("Model saved at: runs/detect/waste_detector/weights/best.pt")
    print("Results saved at: runs/detect/waste_detector/")
    print("\nNext steps:")
    print("1. Review training plots in runs/detect/waste_detector/")
    print("2. Analyze confusion matrix for class-specific issues")
    print("3. If accuracy is low, try larger model or more epochs")


if __name__ == "__main__":
    main()
