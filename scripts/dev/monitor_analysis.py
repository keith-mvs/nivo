"""Monitor video analysis progress in real-time."""

import os
import json
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


def load_progress():
    """Load progress from JSON file."""
    if os.path.exists("analysis_progress.json"):
        with open("analysis_progress.json") as f:
            return json.load(f)
    return None


def get_database_stats():
    """Get statistics from database."""
    if not os.path.exists("video_library.db"):
        return None

    conn = sqlite3.connect("video_library.db")
    cursor = conn.cursor()

    # Total videos
    cursor.execute("SELECT COUNT(*) FROM videos")
    total = cursor.fetchone()[0]

    # Recent videos (last hour)
    one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
    cursor.execute(
        "SELECT COUNT(*) FROM videos WHERE analysis_date >= ?",
        (one_hour_ago,)
    )
    recent = cursor.fetchone()[0]

    # Quality distribution
    cursor.execute("""
        SELECT
            CASE
                WHEN quality_avg >= 90 THEN 'high'
                WHEN quality_avg >= 70 THEN 'good'
                ELSE 'low'
            END as quality_tier,
            COUNT(*) as count
        FROM videos
        WHERE quality_avg IS NOT NULL
        GROUP BY quality_tier
    """)
    quality_dist = dict(cursor.fetchall())

    # Resolution distribution
    cursor.execute("""
        SELECT resolution, COUNT(*) as count
        FROM videos
        GROUP BY resolution
        ORDER BY count DESC
        LIMIT 5
    """)
    resolution_dist = cursor.fetchall()

    # Average duration
    cursor.execute("SELECT AVG(duration_sec) FROM videos")
    avg_duration = cursor.fetchone()[0] or 0

    conn.close()

    return {
        "total_videos": total,
        "videos_last_hour": recent,
        "quality_distribution": quality_dist,
        "resolution_distribution": resolution_dist,
        "avg_duration_sec": avg_duration,
    }


def format_duration(seconds):
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def estimate_completion(progress, total_videos):
    """Estimate time to completion."""
    if not progress or "start_time" not in progress:
        return None

    completed = len(progress.get("completed", []))
    if completed == 0:
        return None

    start = datetime.fromisoformat(progress["start_time"])
    elapsed = (datetime.now() - start).total_seconds()

    # Calculate rate
    rate = completed / elapsed  # videos per second

    # Estimate remaining time
    remaining = total_videos - completed
    if rate > 0:
        eta_seconds = remaining / rate
        eta_time = datetime.now() + timedelta(seconds=eta_seconds)
        return {
            "rate_per_min": rate * 60,
            "eta_seconds": eta_seconds,
            "eta_time": eta_time,
            "elapsed_seconds": elapsed,
        }

    return None


def display_progress(clear_screen=True):
    """Display current progress."""
    if clear_screen:
        os.system('cls' if os.name == 'nt' else 'clear')

    print("=" * 70)
    print("VIDEO ANALYSIS PROGRESS MONITOR")
    print("=" * 70)
    print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load progress file
    progress = load_progress()
    if progress:
        completed = len(progress.get("completed", []))
        errors = len(progress.get("errors", []))
        last_batch = progress.get("last_batch", 0)

        print(f"Batches completed: {last_batch}")
        print(f"Videos analyzed: {completed}")
        print(f"Errors: {errors}")

        if errors > 0:
            print("\nRecent errors:")
            for error in progress["errors"][-5:]:
                print(f"  - {error['path']}: {error['error']}")

    else:
        print("Progress file not found - analysis may not have started yet.")

    # Database stats
    print("\n" + "-" * 70)
    print("DATABASE STATISTICS")
    print("-" * 70 + "\n")

    db_stats = get_database_stats()
    if db_stats:
        print(f"Total videos in database: {db_stats['total_videos']}")
        print(f"Videos added (last hour): {db_stats['videos_last_hour']}")
        print(f"Average duration: {format_duration(db_stats['avg_duration_sec'])}")

        print("\nQuality distribution:")
        for tier, count in db_stats['quality_distribution'].items():
            pct = (count / db_stats['total_videos'] * 100) if db_stats['total_videos'] > 0 else 0
            print(f"  {tier}: {count} ({pct:.1f}%)")

        print("\nTop resolutions:")
        for res, count in db_stats['resolution_distribution']:
            pct = (count / db_stats['total_videos'] * 100) if db_stats['total_videos'] > 0 else 0
            print(f"  {res}: {count} ({pct:.1f}%)")

    else:
        print("Database not found - no analysis results yet.")

    # Estimate completion
    print("\n" + "-" * 70)
    print("COMPLETION ESTIMATE")
    print("-" * 70 + "\n")

    total_videos = 1817  # User's library size
    if progress and db_stats:
        completed = db_stats['total_videos']
        remaining = total_videos - completed
        pct_complete = (completed / total_videos * 100) if total_videos > 0 else 0

        # Progress bar
        bar_width = 50
        filled = int(bar_width * completed / total_videos)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"Progress: [{bar}] {pct_complete:.1f}%")
        print(f"Completed: {completed}/{total_videos} videos")
        print(f"Remaining: {remaining} videos")

        # Estimate time
        estimate = estimate_completion(progress, total_videos)
        if estimate:
            print(f"\nProcessing rate: {estimate['rate_per_min']:.1f} videos/minute")
            print(f"Elapsed time: {format_duration(estimate['elapsed_seconds'])}")
            print(f"Estimated remaining: {format_duration(estimate['eta_seconds'])}")
            print(f"Estimated completion: {estimate['eta_time'].strftime('%Y-%m-%d %H:%M:%S')}")

    else:
        print("Not enough data for estimation yet.")

    print("\n" + "=" * 70)
    print("Press Ctrl+C to stop monitoring")
    print("=" * 70)


def monitor_continuous(interval_seconds=30):
    """Monitor progress continuously."""
    try:
        while True:
            display_progress(clear_screen=True)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def monitor_once():
    """Display progress once and exit."""
    display_progress(clear_screen=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor video analysis progress"
    )
    parser.add_argument(
        "--continuous",
        "-c",
        action="store_true",
        help="Monitor continuously (refresh every 30s)"
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=30,
        help="Refresh interval in seconds (default: 30)"
    )

    args = parser.parse_args()

    if args.continuous:
        monitor_continuous(args.interval)
    else:
        monitor_once()
