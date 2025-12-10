"""Embed generated tags into image EXIF/IPTC metadata."""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from processors.tagger import MetadataTagger


def embed_tags_to_images(
    analysis_file: str,
    dry_run: bool = True,
    test_limit: int = None,
    backup_suffix: str = ".original"
):
    """
    Embed tags from analysis into image EXIF/IPTC metadata.

    Args:
        analysis_file: Path to analysis JSON with tags
        dry_run: If True, only show what would be done without modifying files
        test_limit: Only process first N images (for testing)
        backup_suffix: Suffix for backup files (None = no backup)
    """
    print("=" * 70)
    print("EXIF/IPTC TAG EMBEDDING")
    print("=" * 70)
    print()

    # Load analysis with tags
    with open(analysis_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    total = len(results)
    print(f"Loaded {total} images from analysis")

    if test_limit:
        results = results[:test_limit]
        print(f"TESTING MODE: Processing only first {test_limit} images")

    if dry_run:
        print("DRY RUN MODE: No files will be modified")
    elif backup_suffix:
        print(f"BACKUP ENABLED: Original files saved with '{backup_suffix}' suffix")
    else:
        print("WARNING: No backups will be created!")

    print()

    # Initialize tagger
    tagger = MetadataTagger()

    # Statistics
    stats = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "tags_written": 0,
    }

    # Process each image
    for i, img_data in enumerate(results):
        file_path = img_data.get("file_path")
        if not file_path:
            print(f"[{i+1}/{len(results)}] SKIP: No file_path in data")
            stats["skipped"] += 1
            continue

        # Check if file exists
        if not Path(file_path).exists():
            print(f"[{i+1}/{len(results)}] SKIP: File not found: {file_path}")
            stats["skipped"] += 1
            continue

        # Get tags
        flat_tags = img_data.get("flat_tags", [])
        categorized_tags = img_data.get("tags", {})

        if not flat_tags:
            print(f"[{i+1}/{len(results)}] SKIP: No tags for {Path(file_path).name}")
            stats["skipped"] += 1
            continue

        # Prepare tag data for embedding
        tag_data = {
            "keywords": flat_tags,  # IPTC Keywords
            "title": img_data.get("primary_scene", ""),
            "description": f"Quality: {img_data.get('quality_score', 0):.0f}/100",
        }

        # Add category-specific tags to description
        category_summary = []
        for category, tags in categorized_tags.items():
            if tags:
                category_summary.append(f"{category}: {', '.join(tags[:3])}")

        if category_summary:
            tag_data["description"] += " | " + " | ".join(category_summary[:3])

        if dry_run:
            print(f"[{i+1}/{len(results)}] WOULD EMBED: {Path(file_path).name}")
            print(f"  Keywords: {', '.join(flat_tags[:5])}{'...' if len(flat_tags) > 5 else ''}")
            print(f"  Title: {tag_data['title']}")
            stats["processed"] += 1
            stats["tags_written"] += len(flat_tags)
        else:
            try:
                # Create backup if requested
                if backup_suffix:
                    backup_path = str(file_path) + backup_suffix
                    if not Path(backup_path).exists():
                        import shutil
                        shutil.copy2(file_path, backup_path)

                # Embed tags
                success = tagger.embed_tags(
                    image_path=file_path,
                    keywords=tag_data["keywords"],
                    title=tag_data.get("title"),
                    description=tag_data.get("description"),
                )

                if success:
                    print(f"[{i+1}/{len(results)}] OK: {Path(file_path).name}")
                    stats["processed"] += 1
                    stats["tags_written"] += len(flat_tags)
                else:
                    print(f"[{i+1}/{len(results)}] FAIL: {Path(file_path).name}")
                    stats["errors"] += 1

            except Exception as e:
                print(f"[{i+1}/{len(results)}] ERROR: {Path(file_path).name}: {e}")
                stats["errors"] += 1

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images:      {total}")
    print(f"Processed:         {stats['processed']}")
    print(f"Skipped:           {stats['skipped']}")
    print(f"Errors:            {stats['errors']}")
    print(f"Tags written:      {stats['tags_written']}")
    print(f"Avg tags/image:    {stats['tags_written'] / max(stats['processed'], 1):.1f}")
    print()

    if dry_run:
        print("DRY RUN completed. Run with --execute to actually modify files.")
    else:
        print("Tag embedding complete!")
        if backup_suffix:
            print(f"Original files saved with '{backup_suffix}' suffix.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed tags into image metadata")
    parser.add_argument(
        "--input",
        "-i",
        default="analysis_with_tags.json",
        help="Input analysis JSON file",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually modify files (default is dry-run)",
    )
    parser.add_argument(
        "--test",
        "-t",
        type=int,
        help="Test mode: only process first N images",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files (DANGEROUS)",
    )

    args = parser.parse_args()

    embed_tags_to_images(
        analysis_file=args.input,
        dry_run=not args.execute,
        test_limit=args.test,
        backup_suffix=None if args.no_backup else ".original",
    )
