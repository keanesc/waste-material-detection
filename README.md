# Waste Detection with YOLOv8

YOLOv8 object detection model for identifying waste types: general, plastic, paper, and metal.

## Dataset

This project uses the [TACO dataset](http://tacodataset.org/) which contains 1,500 images with 4,784 annotations across 60 waste categories.

### Simplified Categories

The original 60 categories are mapped to 4 main waste types:

- General: Non-recyclable waste
- Plastic: Bottles, containers, bags, wrappers
- Paper: Cardboard, cartons, paper bags
- Metal: Cans, foil, bottle caps

## Quick Start

### 1. Prepare Dataset

```bash
pixi run prepare
```

Converts COCO annotations to YOLO format and simplifies categories.

### 2. Train Model

```bash
pixi run train
```

Trains YOLOv8n for 50 epochs (640x640 images, batch size 16).

### 3. Use Model

```python
from ultralytics import YOLO

model = YOLO('runs/detect/waste_detector/weights/best.pt')
results = model.predict('image.jpg', conf=0.25)
```

## Installation

```bash
pixi install
```

## Use Cases

- **Smart Recycling Bins**: Automatically sort waste into correct bins
- **Waste Auditing**: Count and categorize waste in facilities
- **Recycling Centers**: Automated sorting on conveyor belts
- **Environmental Monitoring**: Track waste types in public spaces

## Performance

The model is evaluated on:

- Precision: How many detections are correct
- Recall: How many objects are detected
- mAP50: Mean Average Precision at 50% IoU
- mAP50-95: Mean Average Precision at 50-95% IoU

Metrics are displayed after training and saved to `runs/detect/waste_detector/`.

## License

- Code: GNU GPLv3 License
- TACO Dataset: Check [TACO repository](https://github.com/pedropro/TACO) for license
