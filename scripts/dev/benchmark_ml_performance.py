"""Benchmark ML vision analyzers: PyTorch baseline vs YOLO vs TensorRT."""

import sys
import time
from pathlib import Path
from typing import Dict, List
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.analyzers.ml_vision import MLVisionAnalyzer
from src.core.analyzers.ml_vision_yolo import YOLOVisionAnalyzer
from src.core.analyzers.ml_vision_tensorrt import TensorRTVisionAnalyzer
import glob


def find_test_images(num_images: int = 10) -> List[str]:
    """Find test images from Pictures folder."""
    # User's Pictures folder with subfolders by format
    base_paths = [
        r"C:\Users\kjfle\Pictures\jpeg",
        r"C:\Users\kjfle\Pictures\png",
        r"C:\Users\kjfle\Pictures\jpg",
        r"D:\Pictures\Camera Roll",
        r"C:\Users\kjfle\OneDrive\Pictures\Camera Roll"
    ]

    image_paths = []
    for base_path in base_paths:
        if Path(base_path).exists():
            patterns = ["*.jpg", "*.jpeg", "*.png"]
            for pattern in patterns:
                found = glob.glob(f"{base_path}/{pattern}")
                image_paths.extend(found[:num_images])
                if len(image_paths) >= num_images:
                    break
            if image_paths:
                break

    return image_paths[:num_images]


def benchmark_analyzer(
    analyzer,
    name: str,
    image_paths: List[str],
    batch_size: int
) -> Dict:
    """Benchmark a single analyzer."""
    print(f"\n{'=' * 70}")
    print(f"BENCHMARKING: {name}")
    print(f"{'=' * 70}")
    print(f"Images: {len(image_paths)}")
    print(f"Batch size: {batch_size}")

    # Warmup (model loading)
    print("\nWarming up (loading models)...")
    warmup_start = time.time()
    try:
        _ = analyzer.analyze_batch(image_paths[:1], show_progress=False)
        warmup_time = time.time() - warmup_start
        print(f"Warmup complete ({warmup_time:.2f}s)")
    except Exception as e:
        print(f"Warmup failed: {e}")
        return {"error": str(e)}

    # Benchmark
    print("\nRunning benchmark...")
    start_time = time.time()

    try:
        results = analyzer.analyze_batch(image_paths, show_progress=True)
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        images_per_sec = len(image_paths) / total_time
        time_per_image = total_time / len(image_paths)

        # Get memory usage
        if hasattr(analyzer, 'get_memory_usage'):
            memory_info = analyzer.get_memory_usage()
        else:
            memory_info = {}

        benchmark_results = {
            "name": name,
            "success": True,
            "num_images": len(image_paths),
            "batch_size": batch_size,
            "total_time_sec": total_time,
            "images_per_sec": images_per_sec,
            "time_per_image_sec": time_per_image,
            "warmup_time_sec": warmup_time,
            "memory": memory_info,
            "sample_results": results[:2] if results else []
        }

        print(f"\n{'=' * 70}")
        print(f"RESULTS: {name}")
        print(f"{'=' * 70}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Images/sec: {images_per_sec:.2f}")
        print(f"Time/image: {time_per_image * 1000:.2f}ms")

        if memory_info:
            print(f"\nMemory Usage:")
            for key, value in memory_info.items():
                print(f"  {key}: {value:.2f}")

        return benchmark_results

    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "name": name,
            "success": False,
            "error": str(e)
        }


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("ML VISION ANALYZER BENCHMARK")
    print("=" * 70)

    # Find test images
    num_test_images = 20
    print(f"\nFinding {num_test_images} test images...")
    test_images = find_test_images(num_test_images)

    if not test_images:
        print("ERROR: No test images found!")
        print("Please ensure Camera Roll has images or update the paths")
        return

    print(f"Found {len(test_images)} images")
    for i, img in enumerate(test_images[:5], 1):
        print(f"  {i}. {Path(img).name}")
    if len(test_images) > 5:
        print(f"  ... and {len(test_images) - 5} more")

    # Run benchmarks
    all_results = []

    # 1. PyTorch Baseline (DETR)
    print("\n" + "=" * 70)
    print("BENCHMARK 1: PyTorch Baseline (CLIP + DETR)")
    print("=" * 70)
    try:
        analyzer_baseline = MLVisionAnalyzer(
            use_gpu=True,
            batch_size=8,
            min_confidence=0.6
        )
        results_baseline = benchmark_analyzer(
            analyzer_baseline,
            "PyTorch Baseline (DETR)",
            test_images,
            batch_size=8
        )
        all_results.append(results_baseline)
    except Exception as e:
        print(f"Baseline benchmark failed: {e}")
        all_results.append({"name": "PyTorch Baseline", "success": False, "error": str(e)})

    # 2. YOLO Analyzer
    print("\n" + "=" * 70)
    print("BENCHMARK 2: YOLO Analyzer (CLIP + YOLOv8)")
    print("=" * 70)
    try:
        analyzer_yolo = YOLOVisionAnalyzer(
            use_gpu=True,
            batch_size=16,
            min_confidence=0.6
        )
        results_yolo = benchmark_analyzer(
            analyzer_yolo,
            "YOLO Analyzer (YOLOv8)",
            test_images,
            batch_size=16
        )
        all_results.append(results_yolo)
    except Exception as e:
        print(f"YOLO benchmark failed: {e}")
        all_results.append({"name": "YOLO Analyzer", "success": False, "error": str(e)})

    # 3. TensorRT Analyzer (torch_tensorrt)
    print("\n" + "=" * 70)
    print("BENCHMARK 3: TensorRT Analyzer (FP16)")
    print("=" * 70)
    try:
        analyzer_trt = TensorRTVisionAnalyzer(
            use_gpu=True,
            batch_size=16,
            min_confidence=0.6,
            use_tensorrt=True,
            precision="fp16"
        )
        results_trt = benchmark_analyzer(
            analyzer_trt,
            "TensorRT FP16",
            test_images,
            batch_size=16
        )
        all_results.append(results_trt)
    except Exception as e:
        print(f"TensorRT benchmark failed: {e}")
        all_results.append({"name": "TensorRT FP16", "success": False, "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print("\n{:<30} {:>15} {:>15} {:>15}".format(
        "Analyzer", "Time (s)", "Images/sec", "Speedup"
    ))
    print("-" * 70)

    baseline_time = None
    for result in all_results:
        if not result.get("success"):
            print(f"{result['name']:<30} {'FAILED':<15}")
            continue

        total_time = result["total_time_sec"]
        throughput = result["images_per_sec"]

        if baseline_time is None:
            baseline_time = total_time
            speedup = 1.0
        else:
            speedup = baseline_time / total_time

        print(f"{result['name']:<30} {total_time:>14.2f}s {throughput:>14.2f} {speedup:>14.2f}x")

    # Save results
    output_file = "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "test_images_count": len(test_images),
            "results": all_results,
            "summary": {
                "baseline_time": baseline_time,
                "best_time": min([r["total_time_sec"] for r in all_results if r.get("success")], default=None),
                "max_speedup": max([baseline_time / r["total_time_sec"] for r in all_results if r.get("success") and baseline_time], default=None)
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
