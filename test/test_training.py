"""Quick test to identify training crash cause"""

from pathlib import Path

from ultralytics import YOLO

print("Starting minimal training test...")

data_yaml = Path("data/yolo_dataset/waste.yaml")

if not data_yaml.exists():
    print(f"Error: {data_yaml} not found")
    exit(1)

print("Loading model...")
model = YOLO("models/yolov8n.pt")

print("Starting training with minimal config...")
print("This will run just 1 epoch to test for crashes")

try:
    results = model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=640,
        batch=1,  # Start with batch=1 to isolate the issue
        device="cpu",
        project="runs/detect",
        name="test_run",
        exist_ok=True,
        verbose=True,
        workers=0,  # Disable multiprocessing
    )
    print("\nSUCCESS: Training completed first epoch without crashing!")
    print("The issue may be batch size or workers related.")

except Exception as e:
    print(f"\nCRASH DETECTED: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback

    traceback.print_exc()
