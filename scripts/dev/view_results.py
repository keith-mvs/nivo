"""Quick viewer for analysis results."""

import json
from collections import Counter
from pathlib import Path

# Load results
with open('pictures_analysis_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

print("=" * 70)
print(f"IMAGE ANALYSIS RESULTS - {len(results)} images")
print("=" * 70)
print()

# Overall quality stats
qualities = [r['quality_score'] for r in results if 'quality_score' in r]
avg_quality = sum(qualities) / len(qualities) if qualities else 0

print(f"Average Quality: {avg_quality:.1f}/100")
print(f"Best Quality: {max(qualities):.1f}/100")
print(f"Worst Quality: {min(qualities):.1f}/100")
print()

# Scene distribution
scenes = [r['primary_scene'] for r in results if 'primary_scene' in r]
scene_counts = Counter(scenes)

print("=" * 70)
print("TOP SCENES")
print("=" * 70)
for scene, count in scene_counts.most_common(10):
    percentage = (count / len(results)) * 100
    bar = "#" * int(percentage / 2)
    print(f"{scene:15s} {count:4d} ({percentage:5.1f}%) {bar}")
print()

# Quality distribution
quality_ranges = {
    'Excellent (90+)': len([q for q in qualities if q >= 90]),
    'Good (75-89)': len([q for q in qualities if 75 <= q < 90]),
    'Average (60-74)': len([q for q in qualities if 60 <= q < 75]),
    'Below Average (<60)': len([q for q in qualities if q < 60])
}

print("=" * 70)
print("QUALITY DISTRIBUTION")
print("=" * 70)
for range_name, count in quality_ranges.items():
    percentage = (count / len(qualities)) * 100
    bar = "#" * int(percentage / 2)
    print(f"{range_name:20s} {count:4d} ({percentage:5.1f}%) {bar}")
print()

# Sharpness
sharpness = [r['sharpness_level'] for r in results if 'sharpness_level' in r]
sharpness_counts = Counter(sharpness)

print("=" * 70)
print("SHARPNESS LEVELS")
print("=" * 70)
for level, count in sharpness_counts.most_common():
    percentage = (count / len(results)) * 100
    bar = "#" * int(percentage / 2)
    print(f"{level:15s} {count:4d} ({percentage:5.1f}%) {bar}")
print()

# Show sample high-quality images by scene
print("=" * 70)
print("SAMPLE HIGH-QUALITY IMAGES (Quality > 85)")
print("=" * 70)

# Get top 3 images per top scene
for scene, _ in scene_counts.most_common(5):
    scene_images = [r for r in results if r.get('primary_scene') == scene and r.get('quality_score', 0) > 85]
    scene_images.sort(key=lambda x: x.get('quality_score', 0), reverse=True)

    if scene_images:
        print(f"\n{scene.upper()}:")
        for img in scene_images[:3]:
            print(f"  - {Path(img['file_name']).name[:60]}")
            print(f"    Quality: {img['quality_score']:.1f}, Sharpness: {img['sharpness_level']}, {img['width']}x{img['height']}px")

print()
print("=" * 70)
print(f"Full results in: pictures_analysis_results.json")
print("=" * 70)
