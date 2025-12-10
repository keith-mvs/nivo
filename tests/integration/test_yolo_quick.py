"""Quick test of YOLOv8 analyzer on sample images."""

import time
from pathlib import Path
from src.core.analyzers.ml_vision_yolo import YOLOVisionAnalyzer

def test_yolo():
    """Test YOLO analyzer on 10 sample images."""

    # Find test images
    camera_roll = Path("C:/Users/kjfle/Pictures/Camera Roll")
    all_images = list(camera_roll.glob("*.jpg"))[:10]

    if not all_images:
        all_images = list(camera_roll.glob("*.heic"))[:10]

    if not all_images:
        print("ERROR: No test images found in Camera Roll")
        return

    test_images = [str(p) for p in all_images]

    print("=" * 80)
    print("YOLO ANALYZER QUICK TEST")
    print("=" * 80)
    print(f"Test images: {len(test_images)}")
    print(f"Sample: {Path(test_images[0]).name}")
    print()

    # Initialize analyzer
    print("=" * 80)
    print("Initializing YOLOv8 Analyzer")
    print("=" * 80)
    analyzer = YOLOVisionAnalyzer(
        use_gpu=True,
        batch_size=16,
        yolo_model="yolov8n.pt",  # Nano (fastest)
        min_confidence=0.6,
        precision="fp16"
    )
    print()

    # Run analysis
    print("=" * 80)
    print("Running Analysis")
    print("=" * 80)
    start = time.time()
    results = analyzer.analyze_batch(test_images, show_progress=True)
    elapsed = time.time() - start

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {len(test_images)/elapsed:.1f} img/sec")
    print()

    # Show sample results
    if results and len(results) > 0:
        print("Sample result:")
        sample = results[0]
        print(f"  Image: {Path(test_images[0]).name}")
        print(f"  Scene: {sample.get('primary_scene', 'unknown')}")
        print(f"  Scene confidence: {sample.get('scene_confidence', 0):.2f}")
        print(f"  Objects detected: {sample.get('object_count', 0)}")
        if sample.get('objects'):
            print(f"  Top objects: {', '.join([o['object'] for o in sample['objects'][:5]])}")
    print()

    # Clear GPU cache
    analyzer.clear_cache()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_yolo()
