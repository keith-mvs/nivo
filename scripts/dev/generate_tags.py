"""Generate comprehensive tags from analysis results using TagGenerator."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from processors.tag_generator import TagGenerator

# Load analysis
input_file = "analysis.json"
output_file = "analysis_with_tags.json"

print("=" * 70)
print(f"LOADING ANALYSIS FROM: {input_file}")
print("=" * 70)

with open(input_file, 'r', encoding='utf-8') as f:
    results = json.load(f)

print(f"Loaded {len(results)} images")
print()

# Initialize tag generator
generator = TagGenerator(max_tags=30)

print("=" * 70)
print("GENERATING TAGS (10 CATEGORIES)")
print("=" * 70)
print("Categories: scene, objects, quality, color, temporal,")
print("            technical, format, people, location, mood")
print()

# Process each image
for i, img in enumerate(results):
    tags = generator.generate_tags(img)
    img["tags"] = tags
    img["flat_tags"] = generator.get_flat_tags(tags)

    if (i + 1) % 500 == 0:
        print(f"  Processed {i + 1}/{len(results)} images...")

print(f"  Processed {len(results)}/{len(results)} images")
print()

# Save updated analysis with tags
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"[OK] Saved to: {output_file}")
print()

# Generate summary
print("=" * 70)
print("TAG SUMMARY")
print("=" * 70)

summary = generator.get_tag_summary(results)
print(f"Total images: {summary['total_images']}")
print(f"Unique tags:  {summary['unique_tags']}")
print()

print("TOP 20 TAGS:")
print("-" * 40)
for tag, count in summary["top_tags"]:
    pct = (count / len(results)) * 100
    bar = "#" * min(int(pct / 2), 30)
    print(f"  {tag:25s} {count:4d} ({pct:5.1f}%) {bar}")
print()

print("CATEGORY BREAKDOWN:")
print("-" * 40)
for category, tags in summary["category_breakdown"].items():
    if tags:
        print(f"\n  {category.upper()}:")
        for tag, count in tags[:5]:
            pct = (count / len(results)) * 100
            print(f"    {tag:23s} {count:4d} ({pct:5.1f}%)")

print()
print("=" * 70)
print("EXAMPLE QUERIES")
print("=" * 70)

# High quality outdoor photos
outdoor_quality = [img for img in results
                   if "outdoor" in img.get("flat_tags", [])
                   and "excellent_quality" in img.get("flat_tags", [])]
print(f"High-quality outdoor photos: {len(outdoor_quality)}")

# Photos with people
with_people = [img for img in results
               if "has_people" in img.get("flat_tags", [])
               or "single_person" in img.get("flat_tags", [])
               or "couple" in img.get("flat_tags", [])]
print(f"Photos with people: {len(with_people)}")

# Vehicle photos
vehicles = [img for img in results
            if "vehicle" in img.get("flat_tags", [])
            or "has_vehicles" in img.get("flat_tags", [])]
print(f"Vehicle photos: {len(vehicles)}")

# Sharp photos
sharp = [img for img in results if "sharp" in img.get("flat_tags", [])]
print(f"Sharp photos: {len(sharp)}")
