"""Find perceptual duplicates in Batch_1 analysis results."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.processors.deduplicator import Deduplicator


def main():
    """Find similar images in Batch_1."""
    # Load analysis results
    with open("batch1_analysis.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    # Extract file paths
    file_paths = [r["file_path"] for r in results if "file_path" in r]

    print(f"Analyzing {len(file_paths)} images for perceptual duplicates...")
    print("This may take a few minutes...\n")

    # Initialize deduplicator
    dedup = Deduplicator()

    # Find similar images with different thresholds
    thresholds = [4, 8, 12]

    for threshold in thresholds:
        print(f"=== Threshold {threshold} (Hamming distance) ===")
        similar_groups = dedup.find_similar(
            file_paths=file_paths,
            threshold=threshold,
            hash_type="phash",
            show_progress=True
        )

        if similar_groups:
            print(f"Found {len(similar_groups)} groups of similar images:")
            for group_id, group_files in list(similar_groups.items())[:5]:  # Show first 5 groups
                print(f"\n  Group {group_id} ({len(group_files)} images):")
                for f in group_files:
                    print(f"    - {Path(f).name}")

            if len(similar_groups) > 5:
                print(f"\n  ... and {len(similar_groups) - 5} more groups")

            # Save results
            output_file = f"batch1_similar_t{threshold}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(similar_groups, f, indent=2)
            print(f"\n✓ Full results saved to: {output_file}")
        else:
            print("  No similar image groups found at this threshold")

        print()


if __name__ == "__main__":
    main()
