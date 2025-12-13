"""Detect faces in Batch_1 images."""

import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.analyzers import FaceDetector, is_face_detection_available
from src.core.utils.image_io import load_image


def main():
    """Detect faces in Batch_1 images."""
    if not is_face_detection_available():
        print("Error: face-recognition not available")
        print("Install with: pip install face-recognition")
        return

    # Load analysis results
    with open("batch1_analysis.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    # Extract file paths
    file_paths = [r["file_path"] for r in results if "file_path" in r]

    print(f"Detecting faces in {len(file_paths)} images...")
    print("Model: InsightFace buffalo_sc (GPU-accelerated)")
    print("Format support: HEIC/HEIF via pillow-heif")
    print()

    # Initialize face detector
    detector = FaceDetector(model="buffalo_sc", compute_encodings=False, use_gpu=True)

    # Detect faces with HEIC support
    face_results = []
    for i, path in enumerate(file_paths, 1):
        if i % 10 == 0:
            print(f"Processing: {i}/{len(file_paths)}")

        # Load image (supports HEIC via pillow-heif)
        pil_img = load_image(path)
        if pil_img is None:
            face_results.append({
                "file_path": path,
                "face_count": 0,
                "has_faces": False,
                "error": "Failed to load image"
            })
            continue

        # Convert PIL to RGB numpy array for face_recognition
        img_array = np.array(pil_img.convert("RGB"))

        # Detect faces from array
        result = detector.detect_faces_from_array(img_array)
        result["file_path"] = path
        face_results.append(result)

    # Analyze results
    images_with_faces = [r for r in face_results if r.get("has_faces", False)]
    total_faces = sum(r.get("face_count", 0) for r in face_results)
    face_counts = Counter(r.get("face_count", 0) for r in face_results)

    print()
    print("=== Face Detection Results ===")
    print(f"Total images: {len(file_paths)}")
    print(f"Images with faces: {len(images_with_faces)} ({len(images_with_faces)/len(file_paths)*100:.1f}%)")
    print(f"Total faces detected: {total_faces}")
    print(f"Average faces per image (with faces): {total_faces/len(images_with_faces) if images_with_faces else 0:.2f}")
    print()
    print("Face count distribution:")
    for count in sorted(face_counts.keys()):
        print(f"  {count} faces: {face_counts[count]} images")
    print()

    # Show sample images with faces
    print("Sample images with faces:")
    for i, result in enumerate(images_with_faces[:10], 1):
        filename = Path(result["file_path"]).name
        face_count = result.get("face_count", 0)
        print(f"  {i}. {filename} - {face_count} face(s)")

    if len(images_with_faces) > 10:
        print(f"  ... and {len(images_with_faces) - 10} more")

    # Save results
    output_file = "batch1_faces.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(face_results, f, indent=2)
    print()
    print(f"Full results saved to: {output_file}")


if __name__ == "__main__":
    main()
