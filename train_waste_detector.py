from pathlib import Path

import torch
from ultralytics import YOLO  # type: ignore


def train_waste_detector(
    data_yaml,
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu",
    model_size="n",  # n, s, m, l, x
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
        workers=0,  # Disable multiprocessing to avoid crashes
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

    # Auto-detect CUDA availability
    if torch.cuda.is_available():
        device = "cuda"
        model_size = "s"
        batch = 16
        print(f"CUDA detected - using GPU with YOLOv8{model_size}")
    else:
        device = "cpu"
        model_size = "n"
        batch = 4
        print(f"No CUDA - using CPU with YOLOv8{model_size}")

    model, results = train_waste_detector(
        data_yaml=data_yaml,
        epochs=100,
        imgsz=640,
        batch=batch,
        device=device,
        model_size=model_size,
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
