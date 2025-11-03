from pathlib import Path

from ultralytics import YOLO  # type: ignore


def train_waste_detector(data_yaml, epochs=50, imgsz=640, batch=16, device="cpu"):
    project_dir = Path("runs/detect")
    existing_runs = (
        list(project_dir.glob("waste_detector*")) if project_dir.exists() else []
    )

    if existing_runs:
        latest_run = max(existing_runs, key=lambda p: p.stat().st_mtime)
        weights_path = latest_run / "weights" / "best.pt"

        if weights_path.exists():
            print(f"Found existing model at {weights_path}")
            print("Loading existing model...")
            model = YOLO(str(weights_path))
            print("To retrain, delete runs/detect/waste_detector* directories")
            return model, None

    print("Training YOLOv8 for Waste Detection")
    print("Loading YOLOv8n pretrained model...")
    model = YOLO("models/yolov8n.pt")

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Dataset: {data_yaml}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")

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
    )

    print("\nTraining complete")
    return model, results


def validate_model(model):
    print("\nValidating model...")
    metrics = model.val()

    print("Validation metrics:")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

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


def predict_sample(model, image_path, conf=0.25):
    if not Path(image_path).exists():
        print(f"Warning: Image not found at {image_path}")
        return None

    print("\nRunning inference...")
    results = model.predict(
        source=str(image_path),
        conf=conf,
        save=True,
        project="runs/detect",
        name="predictions",
        exist_ok=True,
    )

    if results:
        print(f"Processed {len(results)} image(s)")

        class_names = model.names
        for i, result in enumerate(results):
            if len(result.boxes) > 0:
                print(f"\nImage {i + 1}: {len(result.boxes)} objects")
                class_counts = {}
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = class_names[cls_id]
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                for cls_name, count in class_counts.items():
                    print(f"  {cls_name}: {count}")
            else:
                print(f"\nImage {i + 1}: No objects detected")

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
        epochs=50,
        imgsz=640,
        batch=16,
        device="cpu",  # Change to "cuda" if GPU available
    )

    validate_model(model)
    test_model(model, data_yaml)

    sample_images = project_root / "TACO" / "data" / "batch_1"
    if sample_images.exists():
        print("\nRunning inference on sample images...")
        predict_sample(model, sample_images, conf=0.25)

    print("\nEnd")
    print("Model saved at: runs/detect/waste_detector/weights/best.pt")
    print("Results saved at: runs/detect/waste_detector/")


if __name__ == "__main__":
    main()
