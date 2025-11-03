from pathlib import Path


def check_prerequisites():
    print("Checking prerequisites...")

    project_root = Path(__file__).parent
    taco_dir = project_root / "TACO" / "data"
    annotations = taco_dir / "annotations.json"

    issues = []

    if not taco_dir.exists():
        issues.append("TACO")

    if not annotations.exists():
        issues.append("TACO annotations.json not found")

    if issues:
        print("\nPrerequisites missing:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("All prerequisites found")
    return True


def run_pipeline():
    print("\nWASTE DETECTION PIPELINE")

    if not check_prerequisites():
        return

    print("\nSTEP 1: Prepare Dataset")

    try:
        from prepare_dataset import main as prepare_main

        prepare_main()
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return

    print("\nSTEP 2: Train Model")
    print(
        "Note: Training will take a while. Press Ctrl+C to skip if already trained."
    )

    try:
        from train_waste_detector import main as train_main

        train_main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    print("\nSTEP 3: Test Inference")

    try:
        from waste_detector import detect_waste, load_model

        model = load_model()
        if model:
            sample_dir = Path("TACO")
            if sample_dir.exists():
                detect_waste(model, sample_dir, conf=0.25, save=True)
            else:
                print("Warning: Sample images not found for testing")
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    print("\nPIPELINE COMPLETE")


def show_usage():
    print("\nUSAGE EXAMPLES")

    print("\n1. Prepare dataset:")
    print("   python prepare_dataset.py")

    print("\n2. Train model:")
    print("   python train_waste_detector.py")

    print("\n3. Run inference on images:")
    print("   python waste_detector.py path/to/image.jpg")
    print("   python waste_detector.py path/to/image_folder/")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_usage()
    else:
        run_pipeline()
