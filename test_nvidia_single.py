"""Test NVIDIA Vision analyzer on a single image."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.analyzers.ml_vision_nvidia import NVIDIAVisionAnalyzer


def test_nvidia_analyzer():
    """Test NVIDIA analyzer on a single Batch_1 image."""
    # Test image path
    test_image = Path("D:/Pictures/Batch_1/20240925_020819000_iOS.heic")

    if not test_image.exists():
        print(f"ERROR: Test image not found: {test_image}")
        print("Please update the path to a valid image from Batch_1")
        return

    print("="*70)
    print("NVIDIA Vision Analyzer - Single Image Test")
    print("="*70)
    print()
    print(f"Test image: {test_image.name}")
    print(f"Model: Llama 3.2 Vision 11B (cloud-based)")
    print()

    try:
        # Initialize analyzer
        print("Initializing NVIDIA Build API client...")
        analyzer = NVIDIAVisionAnalyzer(
            model="llama-vision",
            batch_size=1,
            min_confidence=0.3
        )
        print("✓ Analyzer initialized")
        print()

        # Analyze image
        print("Analyzing image (this may take 5-10 seconds)...")
        result = analyzer.analyze_image(str(test_image))
        print("✓ Analysis complete")
        print()

        # Display results
        print("="*70)
        print("RESULTS")
        print("="*70)
        print()

        # Scene classification
        scene = result.get("primary_scene", "unknown")
        print(f"Scene: {scene}")
        print()

        # Objects detected
        objects = result.get("objects_detected", [])
        object_count = result.get("object_count", 0)
        print(f"Objects detected: {object_count}")
        if objects:
            print(f"  {', '.join(objects[:5])}")
            if len(objects) > 5:
                print(f"  ... +{len(objects) - 5} more")
        print()

        # AI-generated tags
        ai_tags = result.get("ai_generated_tags", [])
        if ai_tags:
            print(f"AI-generated tags ({len(ai_tags)}):")
            print(f"  {', '.join(ai_tags)}")
            print()

        # Full result (for debugging)
        print("="*70)
        print("FULL RESULT (JSON)")
        print("="*70)
        import json
        print(json.dumps(result, indent=2))

    except ValueError as e:
        print()
        print("ERROR: API key not found!")
        print()
        print(str(e))
        print()
        print("=" * 70)
        print("HOW TO SET YOUR NVIDIA API KEY")
        print("=" * 70)
        print()
        print("Option 1: Set environment variable (PowerShell):")
        print('  $env:NVIDIA_API_KEY = "nvapi-xxxxx"')
        print()
        print("Option 2: Set environment variable (cmd):")
        print('  set NVIDIA_API_KEY=nvapi-xxxxx')
        print()
        print("Option 3: Add to your system environment variables")
        print()
        print("Get your API key at: https://build.nvidia.com")
        print()

    except Exception as e:
        print()
        print(f"ERROR: {type(e).__name__}: {e}")
        print()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_nvidia_analyzer()
