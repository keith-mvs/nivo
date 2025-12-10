"""Batch video analysis script."""

import os
import sys
import json
from pathlib import Path

from src.core.engine import ImageEngine
from src.core.analyzers.video_analyzer import VideoAnalyzer


def analyze_video_batch(video_paths, output_file, max_frames=30):
    """Analyze a batch of videos."""
    print(f"\n{'='*60}")
    print(f"Video Batch Analysis")
    print(f"{'='*60}\n")

    # Initialize engine
    print("Initializing ML models...")
    engine = ImageEngine()

    # Create video analyzer
    video_analyzer = VideoAnalyzer(
        ml_analyzer=engine.ml_analyzer,
        content_analyzer=engine.content_analyzer,
        keyframe_threshold=30.0,
        min_scene_duration=1.0,
    )

    # Show system info
    if engine.ml_analyzer:
        print(f"Device: {engine.ml_analyzer.device}")
        if engine.ml_analyzer.device.type == "cuda":
            import torch
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Analyze videos
    print(f"Analyzing {len(video_paths)} videos...")
    print(f"Max frames per video: {max_frames}")
    print()

    results = video_analyzer.analyze_batch(
        video_paths,
        extract_keyframes_only=True,
        max_frames=max_frames,
        show_progress=True,
    )

    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Analysis Complete")
    print(f"{'='*60}\n")

    successful = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    print(f"Successfully analyzed: {len(successful)} videos")
    if errors:
        print(f"Errors: {len(errors)} videos")
        for err_result in errors[:5]:
            print(f"  - {Path(err_result.get('file_path', 'unknown')).name}: {err_result.get('error', 'unknown error')}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")

    # Show sample tags
    if successful:
        print(f"\nSample content tags from first analyzed video:")
        first = successful[0]
        print(f"File: {Path(first['file_path']).name}")
        print(f"Duration: {first.get('duration_formatted', 'N/A')}")
        print(f"Resolution: {first.get('resolution', 'N/A')}")

        if "content_tags" in first:
            tags = first["content_tags"]
            for category, tag_list in tags.items():
                if tag_list:
                    print(f"  {category}: {', '.join(tag_list)}")

    print(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze batch of videos")
    parser.add_argument("--input", "-i", required=True, help="JSON file with video paths")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument("--max-frames", type=int, default=30, help="Max frames per video")

    args = parser.parse_args()

    # Load video paths
    with open(args.input) as f:
        video_paths = json.load(f)

    # Analyze
    analyze_video_batch(video_paths, args.output, args.max_frames)
