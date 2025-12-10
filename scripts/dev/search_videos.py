"""Search and filter videos in the database."""

import argparse
import json
from pathlib import Path

from src.core.database.video_db import VideoDatabase


def main():
    """Search videos with various filters."""
    parser = argparse.ArgumentParser(description="Search video library")
    parser.add_argument("--db", default="video_library.db", help="Database file")
    parser.add_argument("--tags", nargs="+", help="Search by tags (OR logic)")
    parser.add_argument("--activity", nargs="+", help="Search by activity tags")
    parser.add_argument("--scene", nargs="+", help="Search by scene tags")
    parser.add_argument("--quality", nargs="+", help="Search by quality tags")
    parser.add_argument("--min-duration", type=float, help="Minimum duration (seconds)")
    parser.add_argument("--max-duration", type=float, help="Maximum duration (seconds)")
    parser.add_argument("--min-quality-score", type=float, help="Minimum quality score")
    parser.add_argument("--resolution", help="Resolution filter (e.g., 1080p, 4k)")
    parser.add_argument("--limit", type=int, default=50, help="Max results")
    parser.add_argument("--stats", action="store_true", help="Show database stats")
    parser.add_argument("--list-tags", action="store_true", help="List all available tags")
    parser.add_argument("--output", "-o", help="Save results to JSON file")

    args = parser.parse_args()

    # Open database
    with VideoDatabase(args.db) as db:
        # Show stats
        if args.stats:
            stats = db.get_stats()
            print("\n=== Video Library Statistics ===\n")
            print(f"Total videos: {stats['total_videos']}")
            print(f"Total size: {stats['total_size_mb']:.2f} MB ({stats['total_size_mb']/1024:.2f} GB)")
            print(f"Average duration: {stats['avg_duration_sec']:.1f} seconds")
            print(f"Average quality: {stats['avg_quality']:.1f}/100")
            return

        # List all tags
        if args.list_tags:
            all_tags = db.get_all_tags()
            print("\n=== Available Tags by Category ===\n")
            for category, tags in sorted(all_tags.items()):
                print(f"{category}:")
                for tag in sorted(tags):
                    print(f"  - {tag}")
                print()
            return

        # Build search criteria
        categories = {}
        if args.activity:
            categories["activities"] = args.activity
        if args.scene:
            categories["scenes"] = args.scene
        if args.quality:
            categories["quality"] = args.quality

        # Search
        results = db.search(
            tags=args.tags,
            categories=categories if categories else None,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            min_quality=args.min_quality_score,
            resolution=args.resolution,
            limit=args.limit,
        )

        # Display results
        print(f"\n=== Search Results ({len(results)} videos) ===\n")

        if not results:
            print("No videos found matching your criteria.")
            return

        for i, video in enumerate(results, 1):
            print(f"{i}. {video['file_name']}")
            print(f"   Path: {video['file_path']}")
            print(f"   Duration: {video['duration_formatted']}")
            print(f"   Resolution: {video['resolution']}")
            if video['quality_avg']:
                print(f"   Quality: {video['quality_avg']:.1f}/100")

            # Show tags
            full_data = json.loads(video['full_analysis'])
            if 'content_tags' in full_data:
                tags = full_data['content_tags']
                tag_str = []
                for cat, tag_list in tags.items():
                    if tag_list:
                        tag_str.append(f"{cat}: {', '.join(tag_list)}")
                if tag_str:
                    print(f"   Tags: {'; '.join(tag_str)}")
            print()

        # Save to file if requested
        if args.output:
            output_data = []
            for video in results:
                full_data = json.loads(video['full_analysis'])
                output_data.append(full_data)

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, default=str)

            print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
