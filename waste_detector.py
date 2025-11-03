from pathlib import Path

from ultralytics import YOLO


def load_model(model_path=None):
    if model_path is None:
        model_path = Path("runs/detect/waste_detector/weights/best.pt")

    model_path = Path(model_path)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Train the model first: python train_waste_detector.py")
        return None

    print(f"Loading model from {model_path}")
    model = YOLO(str(model_path))

    return model


def detect_waste(model, image_path, conf=0.35, iou=0.45, save=True):
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return None

    print(f"\nDetecting waste in: {image_path}")

    # Run detection
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        save=save,
        project="runs/detect",
        name="inference",
        exist_ok=True,
    )

    waste_summary = {"general": 0, "plastic": 0, "paper": 0, "metal": 0}
    total_objects = 0

    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                confidence = float(box.conf[0])

                waste_summary[cls_name] += 1
                total_objects += 1
                print(f"  {cls_name}: {confidence:.2f}")

    print("\nDetection Summary")
    print(f"Total objects: {total_objects}")

    for waste_type, count in waste_summary.items():
        if count > 0:
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"  {waste_type.capitalize()}: {count} ({percentage:.1f}%)")

    if save:
        print("\nAnnotated images saved to runs/detect/inference/")

    return {"total": total_objects, "categories": waste_summary, "results": results}


def main():
    import sys

    model = load_model()
    if model is None:
        return

    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "TACO"
        print(f"No image specified, using: {image_path}")
        print("Usage: python waste_detector.py <image_path>")

    summary = detect_waste(model, image_path, conf=0.35, save=True)

    if summary:
        print("\nEnd")


if __name__ == "__main__":
    main()
