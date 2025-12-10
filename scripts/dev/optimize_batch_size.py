"""Optimize batch size and max frames based on GPU memory and performance."""

import os
import time
import torch
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from src.core.engine import ImageEngine
from src.core.analyzers.video_analyzer import VideoAnalyzer
from src.core.utils.video_io import get_video_info


def get_gpu_memory() -> Tuple[float, float]:
    """
    Get GPU memory usage.

    Returns:
        (used_gb, total_gb)
    """
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return used, total
    return 0.0, 0.0


def get_system_memory() -> Tuple[float, float]:
    """
    Get system RAM usage.

    Returns:
        (used_gb, total_gb)
    """
    mem = psutil.virtual_memory()
    return mem.used / 1024**3, mem.total / 1024**3


def benchmark_configuration(
    video_paths: List[str],
    max_frames: int,
    batch_size: int,
    verbose: bool = True
) -> Dict:
    """
    Benchmark a specific configuration.

    Args:
        video_paths: List of test videos
        max_frames: Maximum frames to analyze per video
        batch_size: Number of videos to process in batch
        verbose: Print progress

    Returns:
        Performance metrics
    """
    if verbose:
        print(f"\nTesting: max_frames={max_frames}, batch_size={batch_size}")

    # Initialize
    engine = ImageEngine()
    analyzer = VideoAnalyzer(
        ml_analyzer=engine.ml_analyzer,
        content_analyzer=engine.content_analyzer,
        keyframe_threshold=30.0,
    )

    # Measure baseline memory
    torch.cuda.empty_cache()
    gpu_before, gpu_total = get_gpu_memory()
    ram_before, ram_total = get_system_memory()

    # Run benchmark
    start_time = time.time()
    try:
        results = analyzer.analyze_batch(
            video_paths[:batch_size],
            extract_keyframes_only=True,
            max_frames=max_frames,
            show_progress=verbose
        )
        success = True
        error_msg = None
    except Exception as e:
        success = False
        error_msg = str(e)
        results = []

    elapsed = time.time() - start_time

    # Measure peak memory
    gpu_after, _ = get_gpu_memory()
    ram_after, _ = get_system_memory()

    metrics = {
        "max_frames": max_frames,
        "batch_size": batch_size,
        "success": success,
        "error": error_msg,
        "videos_processed": len(results),
        "total_time_sec": elapsed,
        "time_per_video_sec": elapsed / batch_size if batch_size > 0 else 0,
        "gpu_memory_used_gb": gpu_after - gpu_before,
        "gpu_memory_peak_gb": gpu_after,
        "gpu_memory_total_gb": gpu_total,
        "gpu_utilization_pct": (gpu_after / gpu_total * 100) if gpu_total > 0 else 0,
        "ram_memory_used_gb": ram_after - ram_before,
        "ram_memory_peak_gb": ram_after,
    }

    if verbose:
        if success:
            print(f"  Time: {elapsed:.1f}s ({metrics['time_per_video_sec']:.2f}s/video)")
            print(f"  GPU Memory: {metrics['gpu_memory_used_gb']:.2f}GB "
                  f"({metrics['gpu_utilization_pct']:.1f}% utilization)")
        else:
            print(f"  FAILED: {error_msg}")

    return metrics


def find_optimal_configuration(video_dir: str, num_test_videos: int = 10) -> Dict:
    """
    Find optimal batch size and max frames configuration.

    Args:
        video_dir: Directory containing test videos
        num_test_videos: Number of videos to test with

    Returns:
        Optimal configuration
    """
    print("=" * 60)
    print("VIDEO ANALYSIS OPTIMIZATION")
    print("=" * 60)

    # Find test videos
    video_paths = []
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        video_paths.extend(Path(video_dir).rglob(f"*{ext}"))
        if len(video_paths) >= num_test_videos:
            break

    video_paths = [str(p) for p in video_paths[:num_test_videos]]

    if len(video_paths) < num_test_videos:
        print(f"Warning: Only found {len(video_paths)} videos")

    print(f"\nTest videos: {len(video_paths)}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    # Get GPU memory
    _, gpu_total = get_gpu_memory()
    print(f"GPU Memory: {gpu_total:.2f} GB")

    # Test configurations
    configurations = [
        # (max_frames, batch_size)
        (20, 1),
        (30, 1),
        (50, 1),
        (30, 5),
        (30, 10),
        (50, 5),
        (50, 10),
    ]

    print("\n" + "=" * 60)
    print("BENCHMARKING CONFIGURATIONS")
    print("=" * 60)

    results = []
    for max_frames, batch_size in configurations:
        metrics = benchmark_configuration(
            video_paths,
            max_frames=max_frames,
            batch_size=min(batch_size, len(video_paths)),
            verbose=True
        )
        results.append(metrics)

        # Clear cache between tests
        torch.cuda.empty_cache()
        time.sleep(2)

    # Find optimal configuration
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Filter successful runs
    successful = [r for r in results if r["success"]]

    if not successful:
        print("\n[ERROR] No successful configurations found!")
        return None

    # Sort by throughput (videos per second)
    successful.sort(
        key=lambda x: x["videos_processed"] / x["total_time_sec"],
        reverse=True
    )

    print("\nTop 3 configurations by throughput:")
    for i, config in enumerate(successful[:3], 1):
        throughput = config["videos_processed"] / config["total_time_sec"]
        print(f"\n{i}. max_frames={config['max_frames']}, "
              f"batch_size={config['batch_size']}")
        print(f"   Throughput: {throughput:.2f} videos/sec")
        print(f"   Time/video: {config['time_per_video_sec']:.2f}s")
        print(f"   GPU Memory: {config['gpu_memory_peak_gb']:.2f}GB "
              f"({config['gpu_utilization_pct']:.1f}%)")

    # Recommend configuration
    optimal = successful[0]
    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 60)
    print(f"\nmax_frames: {optimal['max_frames']}")
    print(f"batch_size: {optimal['batch_size']}")
    print(f"\nExpected performance:")
    print(f"  - {optimal['time_per_video_sec']:.2f} seconds per video")
    print(f"  - {1/optimal['time_per_video_sec']:.2f} videos per minute")
    print(f"  - {60/optimal['time_per_video_sec']:.1f} videos per hour")
    print(f"\nGPU utilization: {optimal['gpu_utilization_pct']:.1f}%")
    print(f"Peak memory: {optimal['gpu_memory_peak_gb']:.2f}GB / {gpu_total:.2f}GB")

    # Estimate full library time
    full_library_size = 1817  # User's library
    estimated_hours = (full_library_size * optimal['time_per_video_sec']) / 3600
    print(f"\nEstimated time for {full_library_size} videos: {estimated_hours:.1f} hours")

    return optimal


def generate_config_recommendation(optimal: Dict, output_file: str = None):
    """Generate configuration file with recommendations."""
    config_text = f"""# Optimized Configuration for Video Analysis
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Recommended Settings

Use these settings with analyze_full_library.py:

```bash
python analyze_full_library.py "C:\\Users\\kjfle\\Videos" \\
    --batch-size {optimal['batch_size']} \\
    --max-frames {optimal['max_frames']}
```

## Performance Metrics

- **Throughput**: {1/optimal['time_per_video_sec']:.2f} videos/minute
- **Time per video**: {optimal['time_per_video_sec']:.2f} seconds
- **GPU utilization**: {optimal['gpu_utilization_pct']:.1f}%
- **GPU memory**: {optimal['gpu_memory_peak_gb']:.2f}GB peak

## Full Library Estimate

For 1,817 videos:
- **Total time**: {(1817 * optimal['time_per_video_sec']) / 3600:.1f} hours
- **Videos per hour**: {3600 / optimal['time_per_video_sec']:.0f}

## Configuration Rationale

- `max_frames={optimal['max_frames']}`: Balances quality and speed
- `batch_size={optimal['batch_size']}`: Optimizes GPU utilization without OOM

## Alternative Settings

If you encounter GPU memory issues:
- Reduce `--max-frames` to 20
- Reduce `--batch-size` to 50

If you want higher quality analysis:
- Increase `--max-frames` to 50
- May increase processing time by 20-30%
"""

    if output_file:
        with open(output_file, 'w') as f:
            f.write(config_text)
        print(f"\nConfiguration saved to: {output_file}")
    else:
        print("\n" + "=" * 60)
        print(config_text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize video analysis batch size and frame count"
    )
    parser.add_argument(
        "video_dir",
        help="Directory containing test videos"
    )
    parser.add_argument(
        "--num-test-videos",
        type=int,
        default=10,
        help="Number of videos to test with (default: 10)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Save configuration to file"
    )

    args = parser.parse_args()

    # Run optimization
    optimal = find_optimal_configuration(args.video_dir, args.num_test_videos)

    if optimal:
        generate_config_recommendation(optimal, args.output)
