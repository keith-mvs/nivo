"""Quick video search tool - search your 1,817 analyzed videos."""

import sqlite3
import json
import sys
from typing import List, Dict


def search_videos(
    quality: str = None,
    resolution: str = None,
    scene: str = None,
    activity: str = None,
    min_duration: float = None,
    max_duration: float = None,
    limit: int = 20
) -> List[Dict]:
    """
    Search video library with filters.

    Args:
        quality: "high", "good", or "low"
        resolution: "4k", "1080p", "720p"
        scene: Scene tag (e.g., "screenshot", "outdoor scene", "sunset")
        activity: Activity tag (e.g., "outdoor", "cooking")
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        limit: Maximum results to return

    Returns:
        List of matching videos
    """
    conn = sqlite3.connect("video_library.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Build query
    where_clauses = []
    params = []

    if quality:
        quality_map = {
            "high": "high-quality",
            "good": "good-quality",
            "low": "low-quality"
        }
        tag = quality_map.get(quality.lower())
        if tag:
            where_clauses.append("""
                id IN (SELECT video_id FROM tags WHERE category='quality' AND tag=?)
            """)
            params.append(tag)

    if resolution:
        where_clauses.append("resolution LIKE ?")
        if resolution.lower() == "4k":
            params.append("%3840%")
        elif resolution.lower() == "1080p":
            params.append("%1920%")
        elif resolution.lower() == "720p":
            params.append("%720%")

    if scene:
        where_clauses.append("""
            id IN (SELECT video_id FROM tags WHERE category='scenes' AND tag LIKE ?)
        """)
        params.append(f"%{scene}%")

    if activity:
        where_clauses.append("""
            id IN (SELECT video_id FROM tags WHERE category='activities' AND tag LIKE ?)
        """)
        params.append(f"%{activity}%")

    if min_duration:
        where_clauses.append("duration_sec >= ?")
        params.append(min_duration)

    if max_duration:
        where_clauses.append("duration_sec <= ?")
        params.append(max_duration)

    # Build full query
    query = "SELECT * FROM videos"
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY quality_avg DESC, duration_sec DESC"
    query += f" LIMIT {limit}"

    cursor.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return results


def print_results(results: List[Dict]):
    """Print search results in readable format."""
    if not results:
        print("\nNo videos found matching criteria.")
        return

    print(f"\n=== Found {len(results)} videos ===\n")

    for i, video in enumerate(results, 1):
        print(f"{i}. {video['file_name']}")
        print(f"   Path: {video['file_path']}")
        print(f"   Duration: {video['duration_formatted']} | "
              f"Resolution: {video['resolution']} | "
              f"Quality: {video['quality_avg']:.1f}/100")

        # Show tags
        if video['full_analysis']:
            analysis = json.loads(video['full_analysis'])
            tags = analysis.get('content_tags', {})

            scenes = tags.get('scenes', [])
            activities = tags.get('activities', [])

            tag_parts = []
            if scenes:
                tag_parts.append(f"Scenes: {', '.join(scenes[:3])}")
            if activities:
                tag_parts.append(f"Activities: {', '.join(activities[:2])}")

            if tag_parts:
                print(f"   Tags: {' | '.join(tag_parts)}")

        print()


def main():
    """CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Search your 1,817 analyzed videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # High-quality 4K videos
  python search.py --quality high --resolution 4k

  # Outdoor scenes
  python search.py --scene outdoor

  # Short clips under 15 seconds
  python search.py --max-duration 15 --limit 30

  # High-quality screenshots
  python search.py --quality high --scene screenshot

  # Longer videos for editing
  python search.py --min-duration 30 --quality high
        """
    )

    parser.add_argument("--quality", choices=["high", "good", "low"],
                        help="Quality tier")
    parser.add_argument("--resolution", choices=["4k", "1080p", "720p"],
                        help="Video resolution")
    parser.add_argument("--scene", help="Scene tag (e.g., 'outdoor', 'sunset')")
    parser.add_argument("--activity", help="Activity tag (e.g., 'outdoor', 'cooking')")
    parser.add_argument("--min-duration", type=float,
                        help="Minimum duration in seconds")
    parser.add_argument("--max-duration", type=float,
                        help="Maximum duration in seconds")
    parser.add_argument("--limit", type=int, default=20,
                        help="Maximum results (default: 20)")
    parser.add_argument("-o", "--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Search
    results = search_videos(
        quality=args.quality,
        resolution=args.resolution,
        scene=args.scene,
        activity=args.activity,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        limit=args.limit
    )

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved {len(results)} results to {args.output}")
    else:
        print_results(results)


if __name__ == "__main__":
    main()
