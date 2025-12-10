"""Batch process entire video library with progress tracking and resumability."""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

from src.core.engine import ImageEngine
from src.core.analyzers.video_analyzer import VideoAnalyzer
from src.core.database.video_db import VideoDatabase
from src.core.utils.video_io import is_supported_video


class LibraryAnalyzer:
    """Batch video library analyzer with progress tracking."""

    def __init__(
        self,
        video_dir: str,
        db_path: str = "video_library.db",
        batch_size: int = 100,
        max_frames: int = 30,
        resume_file: str = "analysis_progress.json",
    ):
        """
        Initialize library analyzer.

        Args:
            video_dir: Root directory containing videos
            db_path: Database file path
            batch_size: Number of videos to process per batch
            max_frames: Maximum frames to analyze per video
            resume_file: File to track analysis progress
        """
        self.video_dir = video_dir
        self.db_path = db_path
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.resume_file = resume_file

        # Initialize components
        print("Initializing Image Engine...")
        self.engine = ImageEngine()
        self.video_analyzer = VideoAnalyzer(
            ml_analyzer=self.engine.ml_analyzer,
            content_analyzer=self.engine.content_analyzer,
            keyframe_threshold=30.0,
            min_scene_duration=1.0,
        )
        self.db = VideoDatabase(db_path)

    def scan_videos(self) -> list:
        """Scan directory for all videos."""
        print(f"\nScanning {self.video_dir} for videos...")
        videos = []

        for root, dirs, files in os.walk(self.video_dir):
            for file in files:
                path = os.path.join(root, file)
                if is_supported_video(path):
                    videos.append(path)

        print(f"Found {len(videos)} videos")
        return sorted(videos)

    def load_progress(self) -> dict:
        """Load analysis progress from file."""
        if os.path.exists(self.resume_file):
            with open(self.resume_file) as f:
                progress = json.load(f)
                print(f"\nResuming from previous session:")
                print(f"  Already processed: {len(progress['completed'])}")
                print(f"  Errors: {len(progress['errors'])}")
                return progress
        else:
            return {
                "completed": [],
                "errors": [],
                "last_batch": 0,
                "start_time": datetime.now().isoformat(),
            }

    def save_progress(self, progress: dict):
        """Save analysis progress to file."""
        with open(self.resume_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def analyze_library(self, resume: bool = True):
        """
        Analyze entire video library in batches.

        Args:
            resume: Resume from previous progress if available
        """
        # Scan for videos
        all_videos = self.scan_videos()

        # Load progress
        progress = self.load_progress() if resume else {
            "completed": [],
            "errors": [],
            "last_batch": 0,
            "start_time": datetime.now().isoformat(),
        }

        # Filter out already processed
        completed_set = set(progress["completed"])
        remaining = [v for v in all_videos if v not in completed_set]

        if not remaining:
            print("\nAll videos have been analyzed!")
            self.show_summary()
            return

        print(f"\n{'='*60}")
        print(f"Batch Analysis Plan")
        print(f"{'='*60}")
        print(f"Total videos: {len(all_videos)}")
        print(f"Already completed: {len(progress['completed'])}")
        print(f"Remaining: {len(remaining)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Estimated batches: {(len(remaining) + self.batch_size - 1) // self.batch_size}")
        print(f"{'='*60}\n")

        # Process in batches
        for batch_num in range(0, len(remaining), self.batch_size):
            batch_videos = remaining[batch_num:batch_num + self.batch_size]
            batch_idx = batch_num // self.batch_size + 1
            total_batches = (len(remaining) + self.batch_size - 1) // self.batch_size

            print(f"\n{'='*60}")
            print(f"Batch {batch_idx}/{total_batches} ({len(batch_videos)} videos)")
            print(f"{'='*60}\n")

            # Analyze batch
            try:
                results = self.video_analyzer.analyze_batch(
                    batch_videos,
                    extract_keyframes_only=True,
                    max_frames=self.max_frames,
                    show_progress=True,
                )

                # Import to database
                print("\nImporting to database...")
                import_stats = self.db.import_analysis(results)
                print(f"  Inserted: {import_stats['inserted']}")
                print(f"  Updated: {import_stats['updated']}")
                print(f"  Errors: {import_stats['errors']}")

                # Update progress
                for i, result in enumerate(results):
                    video_path = batch_videos[i]
                    if "error" not in result:
                        progress["completed"].append(video_path)
                    else:
                        progress["errors"].append({
                            "path": video_path,
                            "error": result["error"],
                        })

                progress["last_batch"] = batch_idx
                self.save_progress(progress)

                print(f"\nProgress: {len(progress['completed'])}/{len(all_videos)} complete")

            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Progress has been saved.")
                print(f"Resume with the same command to continue from batch {batch_idx}")
                self.save_progress(progress)
                sys.exit(0)

            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {e}")
                print("Saving progress and continuing...")
                self.save_progress(progress)
                continue

        # Final summary
        print(f"\n{'='*60}")
        print("Library Analysis Complete!")
        print(f"{'='*60}\n")
        self.show_summary()

    def show_summary(self):
        """Show library analysis summary."""
        stats = self.db.get_stats()

        print("\n=== Video Library Summary ===\n")
        print(f"Total videos indexed: {stats['total_videos']}")
        print(f"Total size: {stats['total_size_mb']:.2f} MB ({stats['total_size_mb']/1024:.2f} GB)")
        print(f"Average duration: {stats['avg_duration_sec']:.1f} seconds")
        print(f"Average quality: {stats['avg_quality']:.1f}/100")

        print(f"\nAvailable tags by category:")
        for category, tags in sorted(stats['all_tags'].items()):
            print(f"  {category}: {len(tags)} unique tags")

        print(f"\nDatabase: {self.db_path}")
        print(f"Search with: python search_videos.py --help")

    def close(self):
        """Clean up resources."""
        self.db.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze entire video library in batches"
    )
    parser.add_argument(
        "video_dir",
        help="Directory containing videos"
    )
    parser.add_argument(
        "--db",
        default="video_library.db",
        help="Database file path"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Videos per batch (default: 100)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Max frames per video (default: 30)"
    )
    parser.add_argument(
        "--resume-file",
        default="analysis_progress.json",
        help="Progress tracking file"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh (ignore previous progress)"
    )

    args = parser.parse_args()

    # Verify directory exists
    if not os.path.isdir(args.video_dir):
        print(f"Error: Directory not found: {args.video_dir}")
        sys.exit(1)

    # Run analysis
    analyzer = LibraryAnalyzer(
        video_dir=args.video_dir,
        db_path=args.db,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        resume_file=args.resume_file,
    )

    try:
        analyzer.analyze_library(resume=not args.no_resume)
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
