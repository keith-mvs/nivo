"""Test NVIDIA Build API with actual images."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.adapters.nvidia_build import RetailObjectDetector, VisionLanguageModel


def test_with_image(image_path: str):
    """Test both NVIDIA Build API endpoints with a real image."""

    print("=" * 70)
    print("NVIDIA BUILD API - LIVE TEST")
    print("=" * 70)
    print(f"\nTest image: {Path(image_path).name}")
    print()

    # Test 1: Retail Object Detection
    print("-" * 70)
    print("Testing Retail Object Detection")
    print("-" * 70)

    try:
        detector = RetailObjectDetector()
        results = detector.detect_products(image_path, confidence_threshold=0.5)

        print(f"[OK] API call successful")
        print(f"\nResults:")
        print(f"  Products found: {results['product_count']}")
        print(f"  Has products: {results['has_products']}")

        if results['has_products']:
            print(f"  Product types: {results['product_types']}")
            print(f"  Dominant product: {results['dominant_product']}")

            print(f"\n  Detections:")
            for i, det in enumerate(results['detections'][:5], 1):
                print(f"    {i}. {det['class']} (confidence: {det['confidence']:.2f})")

        print()

    except Exception as e:
        print(f"[ERROR] Retail detection failed: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Test 2: Vision-Language Model
    print("-" * 70)
    print("Testing Vision-Language Model (Image Description)")
    print("-" * 70)

    try:
        vlm = VisionLanguageModel()
        description = vlm.describe_image(image_path)

        print(f"[OK] API call successful")
        print(f"\nGenerated description:")
        print(f'  "{description}"')
        print()

    except Exception as e:
        print(f"[ERROR] Vision-language failed: {e}")
        import traceback
        traceback.print_exc()
        print()

    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Use first command line argument or default test image
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # Find a test image
        import glob
        images = glob.glob("C:/Users/kjfle/Pictures/jpeg/*.jpeg")
        if images:
            test_image = images[0]
        else:
            print("ERROR: No test images found")
            print("Usage: python test_nvidia_api_live.py <image_path>")
            sys.exit(1)

    test_with_image(test_image)
