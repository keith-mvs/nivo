"""Show face detection examples to verify it's working."""
import json
from pathlib import Path

# Load results
with open("batch1_faces.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# Filter images with faces
with_faces = [r for r in results if r.get("has_faces", False)]

print("=== VERIFICATION: Face Detection Examples ===\n")

# Show 5 examples with different face counts
print("Examples with 1 face:")
one_face = [r for r in with_faces if r.get("face_count") == 1][:3]
for r in one_face:
    filename = Path(r["file_path"]).name
    bbox = r["face_locations"][0]
    print(f"  {filename}")
    print(f"    Position: left={bbox['left']}, top={bbox['top']}, right={bbox['right']}, bottom={bbox['bottom']}")

print("\nExamples with 2 faces:")
two_faces = [r for r in with_faces if r.get("face_count") == 2][:2]
for r in two_faces:
    filename = Path(r["file_path"]).name
    print(f"  {filename}")
    for i, bbox in enumerate(r["face_locations"], 1):
        print(f"    Face {i}: left={bbox['left']}, top={bbox['top']}, right={bbox['right']}, bottom={bbox['bottom']}")

print("\nExamples with 3+ faces:")
multi_faces = [r for r in with_faces if r.get("face_count") >= 3][:2]
for r in multi_faces:
    filename = Path(r["file_path"]).name
    count = r.get("face_count")
    print(f"  {filename} - {count} faces detected")

print("\n=== VERIFICATION SUMMARY ===")
print(f"Total images analyzed: {len(results)}")
print(f"Images with faces: {len(with_faces)} ({len(with_faces)/len(results)*100:.1f}%)")
print(f"Images without faces: {len(results) - len(with_faces)} ({(len(results)-len(with_faces))/len(results)*100:.1f}%)")
print("\nFace detection is WORKING if:")
print("  1. Bounding boxes have reasonable coordinates (not all zeros)")
print("  2. Face counts vary (not all 0 or all same number)")
print("  3. Results make sense for your image collection")
