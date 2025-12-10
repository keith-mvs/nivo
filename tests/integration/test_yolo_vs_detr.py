"""Benchmark YOLOv8 vs DETR object detection performance."""

import time
from pathlib import Path
from src.core.analyzers.ml_vision import MLVisionAnalyzer
from src.core.analyzers.ml_vision_yolo import YOLOVisionAnalyzer

def benchmark():
    """Compare YOLO vs DETR on 20 sample images."""

    # Find test images
    camera_roll = Path("C:/Users/kjfle/Pictures/Camera Roll")
    all_images = list(camera_roll.glob("*.jpg"))[:20]

    if not all_images:
        all_images = list(camera_roll.glob("*.heic"))[:20]

    if not all_images:
        print("ERROR: No test images found in Camera Roll")
        return

    test_images = [str(p) for p in all_images]

    print("=" * 80)
    print(f"BENCHMARK: YOLO vs DETR Object Detection")
    print("=" * 80)
    print(f"Test images: {len(test_images)}")
    print(f"Sample: {Path(test_images[0]).name}")
    print()

    # Baseline analyzer (DETR)
    print("=" * 80)
    print("1. BASELINE (DETR, facebook/detr-resnet-50)")
    print("=" * 80)
    analyzer_detr = MLVisionAnalyzer(
        use_gpu=True,
        batch_size=8,
        min_confidence=0.6
    )

    start = time.time()
    results_detr = analyzer_detr.analyze_batch(test_images, show_progress=True)
    time_detr = time.time() - start

    print()
    print(f"DETR time: {time_detr:.2f}s")
    print(f"Throughput: {len(test_images)/time_detr:.1f} img/sec")
    print()

    # YOLOv8 analyzer
    print("=" * 80)
    print("2. YOLO (YOLOv8-nano, Ultralytics)")
    print("=" * 80)
    analyzer_yolo = YOLOVisionAnalyzer(
        use_gpu=True,
        batch_size=16,
        yolo_model="yolov8n.pt",
        min_confidence=0.6,
        precision="fp16"
    )

    start = time.time()
    results_yolo = analyzer_yolo.analyze_batch(test_images, show_progress=True)
    time_yolo = time.time() - start

    print()
    print(f"YOLO time: {time_yolo:.2f}s")
    print(f"Throughput: {len(test_images)/time_yolo:.1f} img/sec")
    print()

    # Calculate speedup
    speedup = time_detr / time_yolo

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"DETR:  {time_detr:6.2f}s  ({len(test_images)/time_detr:5.1f} img/sec)")
    print(f"YOLO:  {time_yolo:6.2f}s  ({len(test_images)/time_yolo:5.1f} img/sec)")
    print(f"Speedup:   {speedup:.2f}x")
    print()

    # Verify results match
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    # Compare first result
    if results_detr and results_yolo:
        detr_scene = results_detr[0].get("primary_scene", "unknown")
        yolo_scene = results_yolo[0].get("primary_scene", "unknown")

        print(f"First image: {Path(test_images[0]).name}")
        print(f"  DETR scene:  {detr_scene}")
        print(f"  YOLO scene:  {yolo_scene}")
        print(f"  Match: {'YES' if detr_scene == yolo_scene else 'NO'}")

        detr_objects = len(results_detr[0].get("objects", []))
        yolo_objects = len(results_yolo[0].get("objects", []))

        print(f"  DETR objects:  {detr_objects}")
        print(f"  YOLO objects:  {yolo_objects}")

        # Show detected objects
        if results_detr[0].get("objects"):
            print(f"  DETR detections: {', '.join([o['object'] for o in results_detr[0]['objects'][:5]])}")
        if results_yolo[0].get("objects"):
            print(f"  YOLO detections: {', '.join([o['object'] for o in results_yolo[0]['objects'][:5]])}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model comparison:")
    print(f"  DETR: Transformer-based (slower, fewer classes)")
    print(f"  YOLO: Anchor-free CNN (faster, 80 COCO classes)")
    print(f"Performance gain: {speedup:.2f}x")
    print()

    if speedup >= 2.0:
        print("STATUS: SIGNIFICANT SPEEDUP (>2x faster)")
    elif speedup >= 1.5:
        print("STATUS: GOOD SPEEDUP (1.5-2x faster)")
    elif speedup >= 1.2:
        print("STATUS: MODERATE SPEEDUP (1.2-1.5x faster)")
    else:
        print("STATUS: MINIMAL SPEEDUP (<1.2x faster)")

    print("=" * 80)


if __name__ == "__main__":
    benchmark()
