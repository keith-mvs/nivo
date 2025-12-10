"""Quick analysis of YOLO results."""
import json
from collections import Counter

with open('camera_roll_yolo.json') as f:
    results = json.load(f)

total = len(results)
detected = sum(1 for x in results if x.get('object_count', 0) > 0)
scenes = sum(1 for x in results if x.get('primary_scene') and x['primary_scene'] != 'unknown')

# Get top scenes
scene_counts = Counter(x['primary_scene'] for x in results if x.get('primary_scene'))
top_scenes = scene_counts.most_common(5)

# Get top objects
all_objects = []
for r in results:
    if r.get('objects'):
        for obj in r['objects']:
            all_objects.append(obj['object'])
object_counts = Counter(all_objects)
top_objects = object_counts.most_common(10)

print(f"=== YOLO Analysis Results ===")
print(f"Total images analyzed: {total:,}")
print(f"Images with objects detected: {detected:,} ({100*detected/total:.1f}%)")
print(f"Images with scene classification: {scenes:,} ({100*scenes/total:.1f}%)")
print()
print(f"Top 5 scenes:")
for scene, count in top_scenes:
    print(f"  {scene}: {count:,} ({100*count/total:.1f}%)")
print()
print(f"Top 10 detected objects:")
for obj, count in top_objects:
    print(f"  {obj}: {count:,}")
