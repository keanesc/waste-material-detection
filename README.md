# Waste Detection with YOLOv8

Finetune YOLOv8 object detection model to identify and count waste types (general, plastic, paper, metal) in images for smart recycling and automated sorting systems.

## Dataset

This project uses the [TACO dataset](http://tacodataset.org/) which contains 1,500 images with 4,784 annotations across 60 waste categories.

### Simplified Categories

The original 60 categories are mapped to 4 main waste types:

- **General**: Non-recyclable waste
- **Plastic**: Bottles, containers, bags, wrappers, etc.
- **Paper**: Cardboard, cartons, paper bags, etc.
- **Metal**: Cans, foil, bottle caps, etc.

## Quick Start

### 1. Prepare the Dataset

Convert TACO annotations from COCO format to YOLO format:

```bash
pixi run prepare
```

This script:

- Simplifies 60 categories → 4 waste types
- Converts COCO format → YOLO format
- Creates dataset configuration YAML
- **Idempotent**: Safe to run multiple times (skips if already done)

### 2. Train the Model

Train YOLOv8 on the waste detection dataset:

```bash
pixi run train
```

Training parameters:

- Model: YOLOv8n (nano - fastest)
- Epochs: 50
- Image size: 640x640
- Batch size: 16
- Device: CPU (change to "cuda" for GPU)

**Idempotent**: If a trained model exists, it will load it instead of retraining.

### 3. Use the Trained Model

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/waste_detector/weights/best.pt')

# Predict on images
results = model.predict('image.jpg', conf=0.25)

# Count waste types
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        confidence = float(box.conf[0])
        print(f"{cls_name}: {confidence:.2f}")
```

## Project Structure

```
recyclable-material-detection/
├── prepare_dataset.py               # Dataset preparation script
├── train_waste_detector.py          # Training script
├── TACO/                            # TACO dataset
│   └── data/
│       ├── annotations.json         # Original annotations
│       └── batch_*/                 # Image batches
└── data/
    ├── annotations_simplified.json  # Simplified annotations
    └── yolo_dataset/                # YOLO format dataset
        ├── images/                  # Images
        ├── labels/                  # YOLO labels
        └── waste.yaml               # Dataset config
```

## Requirements

Install dependencies using pixi:

```bash
pixi install
```

## Key Features

✅ **Simple**: Minimal code, easy to understand  
✅ **Idempotent**: Safe to run scripts multiple times  
✅ **Automated**: End-to-end pipeline from dataset to trained model  
✅ **Practical**: Real-world waste categories for recycling systems

## Use Cases

- **Smart Recycling Bins**: Automatically sort waste into correct bins
- **Waste Auditing**: Count and categorize waste in facilities
- **Recycling Centers**: Automated sorting on conveyor belts
- **Environmental Monitoring**: Track waste types in public spaces

## Performance

The model is evaluated on:

- **Precision**: How many detections are correct
- **Recall**: How many objects are detected
- **mAP50**: Mean Average Precision at 50% IoU
- **mAP50-95**: Mean Average Precision at 50-95% IoU

Metrics are displayed after training and saved to `runs/detect/waste_detector/`.

## License

- Code: GNU GPLv3 License
- TACO Dataset: Check [TACO repository](https://github.com/pedropro/TACO) for license
