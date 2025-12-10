"""Compare baseline DETR vs YOLO analysis results.

This script compares two analysis result files and generates a performance report.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load analysis results from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_errors(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze error patterns in results."""
    total = len(results)
    errors = [r for r in results if "error" in r]
    object_errors = [r for r in results if "object_error" in r]
    scene_errors = [r for r in results if "scene_error" in r]

    return {
        "total_images": total,
        "total_errors": len(errors),
        "object_detection_errors": len(object_errors),
        "scene_classification_errors": len(scene_errors),
        "success_rate": (total - len(errors)) / total * 100 if total > 0 else 0,
    }


def analyze_detections(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze object detection statistics."""
    valid_results = [r for r in results if "objects" in r and "error" not in r]

    if not valid_results:
        return {
            "images_with_detections": 0,
            "total_objects": 0,
            "avg_objects_per_image": 0,
            "unique_object_types": 0,
        }

    total_objects = sum(r.get("object_count", 0) for r in valid_results)
    images_with_objects = sum(1 for r in valid_results if r.get("object_count", 0) > 0)

    # Collect all unique object types
    all_objects = set()
    for r in valid_results:
        if "objects" in r:
            for obj in r["objects"]:
                all_objects.add(obj["object"])

    return {
        "images_with_detections": images_with_objects,
        "total_objects": total_objects,
        "avg_objects_per_image": total_objects / len(valid_results) if valid_results else 0,
        "unique_object_types": len(all_objects),
        "top_objects": get_top_objects(valid_results, limit=10),
    }


def get_top_objects(results: List[Dict[str, Any]], limit: int = 10) -> List[tuple]:
    """Get most frequently detected objects."""
    object_counts = {}

    for r in results:
        if "objects" in r:
            for obj in r["objects"]:
                obj_name = obj["object"]
                object_counts[obj_name] = object_counts.get(obj_name, 0) + 1

    # Sort by count descending
    sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_objects[:limit]


def analyze_scenes(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze scene classification statistics."""
    valid_results = [r for r in results if "primary_scene" in r and "error" not in r]

    if not valid_results:
        return {
            "total_classified": 0,
            "unique_scenes": 0,
            "avg_confidence": 0,
        }

    scene_counts = {}
    total_confidence = 0

    for r in valid_results:
        scene = r.get("primary_scene", "unknown")
        scene_counts[scene] = scene_counts.get(scene, 0) + 1
        total_confidence += r.get("scene_confidence", 0)

    # Sort by count descending
    top_scenes = sorted(scene_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_classified": len(valid_results),
        "unique_scenes": len(scene_counts),
        "avg_confidence": total_confidence / len(valid_results) if valid_results else 0,
        "top_scenes": top_scenes,
    }


def compare_results(baseline_file: str, yolo_file: str):
    """Compare baseline and YOLO results."""
    print("=" * 80)
    print("ANALYSIS COMPARISON: Baseline (DETR) vs YOLO")
    print("=" * 80)
    print()

    # Load results
    print("Loading results...")
    baseline = load_results(baseline_file)
    yolo = load_results(yolo_file)

    print(f"Baseline: {len(baseline)} images")
    print(f"YOLO:     {len(yolo)} images")
    print()

    # Error analysis
    print("=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)

    baseline_errors = analyze_errors(baseline)
    yolo_errors = analyze_errors(yolo)

    print(f"{'Metric':<40} {'Baseline':<15} {'YOLO':<15}")
    print("-" * 70)
    print(f"{'Total images':<40} {baseline_errors['total_images']:<15} {yolo_errors['total_images']:<15}")
    print(f"{'Total errors':<40} {baseline_errors['total_errors']:<15} {yolo_errors['total_errors']:<15}")
    print(f"{'Object detection errors':<40} {baseline_errors['object_detection_errors']:<15} {yolo_errors['object_detection_errors']:<15}")
    print(f"{'Scene classification errors':<40} {baseline_errors['scene_classification_errors']:<15} {yolo_errors['scene_classification_errors']:<15}")
    print(f"{'Success rate':<40} {baseline_errors['success_rate']:.1f}%{' ':<11} {yolo_errors['success_rate']:.1f}%")
    print()

    # Object detection analysis
    print("=" * 80)
    print("OBJECT DETECTION")
    print("=" * 80)

    baseline_detections = analyze_detections(baseline)
    yolo_detections = analyze_detections(yolo)

    print(f"{'Metric':<40} {'Baseline':<15} {'YOLO':<15}")
    print("-" * 70)
    print(f"{'Images with detections':<40} {baseline_detections['images_with_detections']:<15} {yolo_detections['images_with_detections']:<15}")
    print(f"{'Total objects detected':<40} {baseline_detections['total_objects']:<15} {yolo_detections['total_objects']:<15}")
    print(f"{'Avg objects per image':<40} {baseline_detections['avg_objects_per_image']:<15.2f} {yolo_detections['avg_objects_per_image']:<15.2f}")
    print(f"{'Unique object types':<40} {baseline_detections['unique_object_types']:<15} {yolo_detections['unique_object_types']:<15}")
    print()

    print("Top 10 detected objects (Baseline):")
    for obj, count in baseline_detections.get('top_objects', []):
        print(f"  {obj:<30} {count:>6}")
    print()

    print("Top 10 detected objects (YOLO):")
    for obj, count in yolo_detections.get('top_objects', []):
        print(f"  {obj:<30} {count:>6}")
    print()

    # Scene classification analysis
    print("=" * 80)
    print("SCENE CLASSIFICATION")
    print("=" * 80)

    baseline_scenes = analyze_scenes(baseline)
    yolo_scenes = analyze_scenes(yolo)

    print(f"{'Metric':<40} {'Baseline':<15} {'YOLO':<15}")
    print("-" * 70)
    print(f"{'Total classified':<40} {baseline_scenes['total_classified']:<15} {yolo_scenes['total_classified']:<15}")
    print(f"{'Unique scenes':<40} {baseline_scenes['unique_scenes']:<15} {yolo_scenes['unique_scenes']:<15}")
    print(f"{'Avg confidence':<40} {baseline_scenes['avg_confidence']:<15.3f} {yolo_scenes['avg_confidence']:<15.3f}")
    print()

    print("Top 10 scenes (Baseline):")
    for scene, count in baseline_scenes.get('top_scenes', []):
        print(f"  {scene:<30} {count:>6}")
    print()

    print("Top 10 scenes (YOLO):")
    for scene, count in yolo_scenes.get('top_scenes', []):
        print(f"  {scene:<30} {count:>6}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Determine winner in each category
    print("Performance comparison:")

    if yolo_errors['success_rate'] > baseline_errors['success_rate']:
        print(f"  [WINNER] YOLO has {yolo_errors['success_rate'] - baseline_errors['success_rate']:.1f}% higher success rate")
    elif baseline_errors['success_rate'] > yolo_errors['success_rate']:
        print(f"  [WINNER] Baseline has {baseline_errors['success_rate'] - yolo_errors['success_rate']:.1f}% higher success rate")
    else:
        print(f"  [TIE] Equal success rate")

    if yolo_detections['total_objects'] > baseline_detections['total_objects']:
        print(f"  [WINNER] YOLO detected {yolo_detections['total_objects'] - baseline_detections['total_objects']} more objects")
    elif baseline_detections['total_objects'] > yolo_detections['total_objects']:
        print(f"  [WINNER] Baseline detected {baseline_detections['total_objects'] - yolo_detections['total_objects']} more objects")
    else:
        print(f"  [TIE] Equal object detection count")

    if yolo_detections['unique_object_types'] > baseline_detections['unique_object_types']:
        print(f"  [WINNER] YOLO detected {yolo_detections['unique_object_types'] - baseline_detections['unique_object_types']} more unique object types")
    elif baseline_detections['unique_object_types'] > yolo_detections['unique_object_types']:
        print(f"  [WINNER] Baseline detected {baseline_detections['unique_object_types'] - yolo_detections['unique_object_types']} more unique object types")
    else:
        print(f"  [TIE] Equal unique object types")

    print()
    print("Note: Performance speed comparison requires timing data from benchmark script.")
    print("Run test_yolo_vs_detr.py for speed comparison.")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <baseline_json> <yolo_json>")
        print()
        print("Example:")
        print("  python compare_results.py camera_roll_full.json camera_roll_yolo.json")
        sys.exit(1)

    baseline_file = sys.argv[1]
    yolo_file = sys.argv[2]

    # Verify files exist
    if not Path(baseline_file).exists():
        print(f"ERROR: Baseline file not found: {baseline_file}")
        sys.exit(1)

    if not Path(yolo_file).exists():
        print(f"ERROR: YOLO file not found: {yolo_file}")
        sys.exit(1)

    compare_results(baseline_file, yolo_file)
