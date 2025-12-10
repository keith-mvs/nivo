"""Quick test script for video engine functionality."""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all video modules can be imported."""
    print("Testing video engine imports...")

    try:
        from src.core.utils.video_io import (
            is_supported_video,
            get_video_info,
            extract_frames,
            extract_keyframes,
        )
        print("  [OK] video_io module")
    except Exception as e:
        print(f"  [FAIL] video_io module: {e}")
        return False

    try:
        from src.core.analyzers.video_analyzer import VideoAnalyzer
        print("  [OK] VideoAnalyzer module")
    except Exception as e:
        print(f"  [FAIL] VideoAnalyzer module: {e}")
        return False

    try:
        import cv2
        print(f"  [OK] OpenCV {cv2.__version__}")
    except Exception as e:
        print(f"  [FAIL] OpenCV: {e}")
        return False

    # Optional dependencies
    optional = {
        'moviepy': 'MoviePy',
        'scenedetect': 'PySceneDetect',
        'ffmpeg': 'ffmpeg-python',
    }

    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [WARN] {name} not installed (optional)")

    return True


def test_video_io():
    """Test video I/O functions with a sample video if available."""
    print("\nTesting video I/O functions...")

    from src.core.utils.video_io import is_supported_video, get_video_info

    # Test file type detection
    assert is_supported_video("test.mp4") == False  # Doesn't exist
    print("  [OK] File type detection")

    # Try to find a sample video
    sample_paths = [
        "C:\\Windows\\Media\\*.mp4",  # Windows sample media
        "sample.mp4",
        "test.mp4",
    ]

    sample_video = None
    import glob
    for pattern in sample_paths:
        matches = glob.glob(pattern)
        if matches:
            sample_video = matches[0]
            break

    if sample_video and os.path.exists(sample_video):
        try:
            info = get_video_info(sample_video)
            print(f"  [OK] Metadata extraction from {os.path.basename(sample_video)}")
            print(f"      Duration: {info['duration_formatted']}")
            print(f"      Resolution: {info['resolution']}")
        except Exception as e:
            print(f"  [FAIL] Metadata extraction: {e}")
            return False
    else:
        print("  [SKIP] No sample video found for testing")

    return True


def test_analyzers():
    """Test analyzer initialization."""
    print("\nTesting analyzer initialization...")

    try:
        from src.core.analyzers.video_analyzer import VideoAnalyzer
        from src.core.analyzers.ml_vision import MLVisionAnalyzer
        from src.core.analyzers.content import ContentAnalyzer

        # Initialize analyzers without GPU for quick test
        content_analyzer = ContentAnalyzer(num_workers=2)
        print("  [OK] ContentAnalyzer initialized")

        # Video analyzer
        video_analyzer = VideoAnalyzer(
            ml_analyzer=None,  # Skip ML for quick test
            content_analyzer=content_analyzer,
        )
        print("  [OK] VideoAnalyzer initialized")

        return True
    except Exception as e:
        print(f"  [FAIL] Analyzer initialization: {e}")
        return False


def test_cli_commands():
    """Test that CLI commands are registered."""
    print("\nTesting CLI commands...")

    try:
        from src.ui.cli import cli

        # Get list of registered commands
        commands = list(cli.commands.keys())

        video_commands = [cmd for cmd in commands if 'video' in cmd]

        if video_commands:
            print(f"  [OK] Video commands registered: {', '.join(video_commands)}")
        else:
            print("  [WARN] No video commands found")

        return True
    except Exception as e:
        print(f"  [FAIL] CLI command check: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Video Engine Quick Test")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Video I/O", test_video_io),
        ("Analyzers", test_analyzers),
        ("CLI Commands", test_cli_commands),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] {name} test crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[WARNING] Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
