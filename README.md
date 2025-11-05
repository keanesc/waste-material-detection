# Recyclable Material Detection

Automated waste sorting using YOLOv8 object detection. Classifies waste into four recyclable categories: **general**, **plastic**, **paper**, and **metal**.

## Overview

This project trains a YOLOv8s model on the TACO dataset to identify and classify waste materials in images. The goal is to enable automated waste sorting systems for recycling facilities and smart bins.

**Key Results:**

- mAP@0.5: **0.270** (27% average precision)
- Best performance: Plastic (0.420) and Metal (0.348)
- Trained on 4,784 annotations from 1,500+ images
- 100 epochs, ~2.4 hours training time

## Dataset

Uses the [TACO dataset](http://tacodataset.org/) with 60 original waste categories consolidated into 4 recyclable streams:

- **General** (31.7%): Non-recyclable waste
- **Plastic** (46.5%): Bottles, containers, bags, wrappers
- **Paper** (10.4%): Cardboard, cartons, paper bags
- **Metal** (11.4%): Cans, foil, bottle caps

Split: 70% training / 15% validation / 15% test

## Quick Start

### 1. Prepare Dataset

```bash
pixi run download  # Clones TACO repository and downloads images
pixi run prepare
```

Converts TACO dataset from COCO format to YOLO format and maps 60 categories to 4 simplified classes.

### 2. Train Model

```bash
pixi run train
```

Trains YOLOv8s for 100 epochs (640×640 resolution, batch size 16, AdamW optimizer).

### 3. Use Model

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/waste_detector/weights/best.pt')

# Predict on image
results = model.predict('image.jpg', conf=0.25)

# Display results
results[0].show()
```

## Performance

| Metric       | Value | Description                            |
| ------------ | ----- | -------------------------------------- |
| Precision    | 0.354 | 35% of detections are correct          |
| Recall       | 0.331 | Detects 33% of actual waste items      |
| mAP@0.5      | 0.270 | Average precision at 50% IoU threshold |
| mAP@0.5:0.95 | 0.202 | Average precision across IoU 50-95%    |

### Per-Class Results

| Class   | mAP@0.5 | Notes                                      |
| ------- | ------- | ------------------------------------------ |
| Plastic | 0.420   | Best performer (distinctive shapes)        |
| Metal   | 0.348   | Good (reflective surfaces, defined shapes) |
| Paper   | 0.273   | Moderate (struggles with crumpled items)   |
| General | 0.060   | Weak (high visual diversity)               |

**Strengths:** Works well on prominent items with good lighting (bottles, cans)  
**Limitations:** Struggles with small objects, occlusion, poor lighting, and deformed waste

## Use Cases

- **Smart Recycling Bins**: Automatically sort waste into correct bins
- **Waste Auditing**: Count and categorize waste in facilities
- **Recycling Centers**: Automated sorting on conveyor belts
- **Environmental Monitoring**: Track waste types in public spaces

## Future Improvements

- Collect more data for underrepresented classes (general waste, paper)
- Train larger models (YOLOv8m/l) at higher resolutions (1024px)
- Fine-tune on domain-specific data (specific bin types, lighting conditions)
- Add depth sensors or multi-spectral imaging for transparent materials
- Deploy optimized models (INT8 quantization, TensorRT) on edge devices

## Project Structure

```text
├── prepare_dataset.py      # Convert TACO to YOLO format
├── train_waste_detector.py # Train YOLOv8 model
├── waste_detector.py       # Inference script
├── generate_report.py      # Generate performance report
├── data/                   # Dataset files
├── models/                 # Pre-trained weights
├── runs/                   # Training outputs
└── report/                 # Technical report and figures
```

## Documentation

See [Technical Report](report/technical_report.md) for detailed methodology, results, and analysis.

## License

- Code: GNU GPLv3 License
- TACO Dataset: Check [TACO repository](https://github.com/pedropro/TACO) for license
