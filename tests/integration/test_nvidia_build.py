"""Test NVIDIA Build API integration."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.adapters.nvidia_build import RetailObjectDetector, VisionLanguageModel


def test_api_key():
    """Test if API key is configured."""
    print("=" * 60)
    print("NVIDIA BUILD API - SETUP TEST")
    print("=" * 60)

    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        print("\n[FAIL] NVIDIA_API_KEY not found in environment")
        print("\nTo set up:")
        print("1. Visit https://build.nvidia.com")
        print("2. Sign up/log in with NVIDIA account")
        print("3. Generate API key")
        print("4. Set environment variable:")
        print("   PowerShell: $env:NVIDIA_API_KEY='your-api-key'")
        print("   Or create .env file with: NVIDIA_API_KEY=your-api-key")
        return False

    print(f"\n[OK] API key found: {api_key[:8]}...{api_key[-4:]}")
    return True


def test_retail_detector():
    """Test retail object detection."""
    print("\n" + "-" * 60)
    print("Testing Retail Object Detector")
    print("-" * 60)

    try:
        detector = RetailObjectDetector()
        print("[OK] Retail detector initialized")

        # Test with a sample image if available
        test_images = [
            "test_product.jpg",
            "test_images/product_sample.jpg",
            # Add path to any test image you have
        ]

        test_image = None
        for img_path in test_images:
            if os.path.exists(img_path):
                test_image = img_path
                break

        if test_image:
            print(f"\nTesting detection on: {test_image}")
            results = detector.detect_products(test_image)

            print(f"[OK] Detection successful")
            print(f"  - Products found: {results['product_count']}")
            print(f"  - Product types: {results['product_types']}")
            print(f"  - Dominant product: {results['dominant_product']}")

            if results['detections']:
                print("\n  Top 3 detections:")
                for i, det in enumerate(results['detections'][:3], 1):
                    print(f"    {i}. {det['class']}: {det['confidence']:.2%}")
        else:
            print("[SKIP] No test image found")
            print("       Create test_product.jpg to test detection")

        return True

    except Exception as e:
        print(f"[FAIL] Retail detector error: {e}")
        return False


def test_vision_language():
    """Test vision-language model."""
    print("\n" + "-" * 60)
    print("Testing Vision-Language Model")
    print("-" * 60)

    try:
        vlm = VisionLanguageModel()
        print("[OK] Vision-language model initialized")

        # Test with sample image if available
        test_images = [
            "test_scene.jpg",
            "test_images/scene_sample.jpg",
        ]

        test_image = None
        for img_path in test_images:
            if os.path.exists(img_path):
                test_image = img_path
                break

        if test_image:
            print(f"\nTesting description on: {test_image}")
            description = vlm.describe_image(test_image)

            print(f"[OK] Description generated")
            print(f"\n  Description: \"{description}\"")

            # Test tag generation
            tags = vlm.generate_searchable_tags(test_image)
            print(f"\n  Generated tags: {tags}")
        else:
            print("[SKIP] No test image found")
            print("       Create test_scene.jpg to test descriptions")

        return True

    except Exception as e:
        print(f"[FAIL] Vision-language error: {e}")
        return False


def test_error_handling():
    """Test error handling."""
    print("\n" + "-" * 60)
    print("Testing Error Handling")
    print("-" * 60)

    try:
        detector = RetailObjectDetector()

        # Test with non-existent file
        try:
            detector.detect_products("nonexistent.jpg")
            print("[FAIL] Should have raised FileNotFoundError")
            return False
        except FileNotFoundError:
            print("[OK] FileNotFoundError handled correctly")

        # Test with invalid confidence threshold
        if os.path.exists("test_product.jpg"):
            try:
                detector.detect_products("test_product.jpg", confidence_threshold=1.5)
                print("[FAIL] Should have raised ValueError")
                return False
            except ValueError:
                print("[OK] ValueError handled correctly")

        return True

    except Exception as e:
        print(f"[FAIL] Error handling test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    results = {}

    # Test 1: API Key
    results['api_key'] = test_api_key()

    if not results['api_key']:
        print("\n" + "=" * 60)
        print("SETUP REQUIRED")
        print("=" * 60)
        print("\nPlease set up NVIDIA_API_KEY before running other tests.")
        return False

    # Test 2: Retail Detector
    results['retail'] = test_retail_detector()

    # Test 3: Vision-Language
    results['vlm'] = test_vision_language()

    # Test 4: Error Handling
    results['error_handling'] = test_error_handling()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        print("\nNVIDIA Build API is ready to use.")
        print("\nNext steps:")
        print("1. Integrate with video analyzer")
        print("2. Process your video library with retail detection")
        print("3. Generate searchable descriptions")
    else:
        print("\n[WARNING] Some tests failed")
        print("Please check the error messages above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
