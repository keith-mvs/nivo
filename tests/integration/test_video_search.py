"""Test video search functionality with various filters."""

import sys
from src.core.database.video_db import VideoDatabase


def test_basic_search():
    """Test basic search functionality."""
    print("\n=== Test 1: Basic Search ===\n")

    with VideoDatabase("video_library.db") as db:
        # Test activity search
        print("Searching for fitness videos...")
        results = db.search(categories={"activities": ["fitness"]}, limit=5)
        print(f"Found {len(results)} fitness videos")
        for video in results[:3]:
            print(f"  - {video['file_name']} ({video['duration_formatted']})")

        # Test quality search
        print("\nSearching for high-quality videos...")
        results = db.search(categories={"quality": ["high-quality"]}, limit=5)
        print(f"Found {len(results)} high-quality videos")
        for video in results[:3]:
            print(f"  - {video['file_name']} (Quality: {video['quality_avg']:.1f})")


def test_combined_filters():
    """Test combined filter searches."""
    print("\n=== Test 2: Combined Filters ===\n")

    with VideoDatabase("video_library.db") as db:
        # Activity + quality
        print("Searching for high-quality fitness videos...")
        results = db.search(
            categories={
                "activities": ["fitness"],
                "quality": ["high-quality"]
            },
            limit=5
        )
        print(f"Found {len(results)} videos matching both criteria")

        # Duration range
        print("\nSearching for videos 30-120 seconds...")
        results = db.search(
            min_duration=30,
            max_duration=120,
            limit=10
        )
        print(f"Found {len(results)} videos in duration range")
        for video in results[:3]:
            print(f"  - {video['file_name']} ({video['duration_formatted']})")


def test_resolution_filters():
    """Test resolution-based searches."""
    print("\n=== Test 3: Resolution Filters ===\n")

    with VideoDatabase("video_library.db") as db:
        for res in ["4k", "1080p", "720p"]:
            results = db.search(resolution=res)
            print(f"{res}: {len(results)} videos")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Test 4: Edge Cases ===\n")

    with VideoDatabase("video_library.db") as db:
        # Empty results
        print("Testing non-existent tag...")
        results = db.search(tags=["nonexistent_tag"])
        print(f"Empty search: {len(results)} results (expected 0)")

        # Very restrictive filters
        print("\nTesting very restrictive filters...")
        results = db.search(
            categories={"quality": ["high-quality"]},
            resolution="4k",
            min_duration=300,
            min_quality=95
        )
        print(f"Restrictive search: {len(results)} results")

        # Limit test
        print("\nTesting limit parameter...")
        results_no_limit = db.search()
        results_with_limit = db.search(limit=10)
        print(f"No limit: {len(results_no_limit)} results")
        print(f"With limit=10: {len(results_with_limit)} results")


def test_statistics():
    """Test database statistics."""
    print("\n=== Test 5: Statistics ===\n")

    with VideoDatabase("video_library.db") as db:
        stats = db.get_stats()
        print(f"Total videos: {stats['total_videos']}")
        print(f"Total size: {stats['total_size_mb']:.2f} MB")
        print(f"Average duration: {stats['avg_duration_sec']:.1f} seconds")
        print(f"Average quality: {stats['avg_quality']:.1f}/100")

        print("\nTags by category:")
        for category, tags in sorted(stats['all_tags'].items()):
            print(f"  {category}: {len(tags)} unique tags")


def test_tag_listing():
    """Test tag listing functionality."""
    print("\n=== Test 6: Available Tags ===\n")

    with VideoDatabase("video_library.db") as db:
        all_tags = db.get_all_tags()
        for category, tags in sorted(all_tags.items()):
            print(f"\n{category.upper()}:")
            for tag in sorted(tags)[:10]:  # Show first 10
                print(f"  - {tag}")
            if len(tags) > 10:
                print(f"  ... and {len(tags) - 10} more")


def test_performance():
    """Test search performance."""
    import time

    print("\n=== Test 7: Performance ===\n")

    with VideoDatabase("video_library.db") as db:
        # Simple search
        start = time.time()
        results = db.search(limit=100)
        elapsed = (time.time() - start) * 1000
        print(f"Simple search (100 results): {elapsed:.1f}ms")

        # Complex search
        start = time.time()
        results = db.search(
            categories={
                "activities": ["fitness", "outdoor"],
                "quality": ["high-quality"]
            },
            min_duration=30,
            max_duration=300,
            min_quality=80,
            limit=50
        )
        elapsed = (time.time() - start) * 1000
        print(f"Complex multi-filter search: {elapsed:.1f}ms")

        # Full table scan
        start = time.time()
        stats = db.get_stats()
        elapsed = (time.time() - start) * 1000
        print(f"Statistics calculation: {elapsed:.1f}ms")


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("VIDEO SEARCH TEST SUITE")
    print("=" * 60)

    try:
        test_basic_search()
        test_combined_filters()
        test_resolution_filters()
        test_edge_cases()
        test_statistics()
        test_tag_listing()
        test_performance()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
