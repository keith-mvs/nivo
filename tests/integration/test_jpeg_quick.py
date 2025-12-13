"""Quick test of YOLO analyzer on JPEG files from D:\Pictures\jpeg."""

from pathlib import Path
from src.core.analyzers.ml_vision_yolo import YOLOVisionAnalyzer

def main():
    # Get 5 JPEG files
    jpegs = list(Path("D:/Pictures/jpeg").glob("*.jpg"))[:5]
    if not jpegs:
        jpegs = list(Path("D:/Pictures/jpeg").glob("*.jpeg"))[:5]

    print(f"Testing {len(jpegs)} JPEG files")

    analyzer = YOLOVisionAnalyzer(use_gpu=True, batch_size=8)
    results = analyzer.analyze_batch([str(p) for p in jpegs], show_progress=True)

    print("\nResults:")
    for i, r in enumerate(results[:5]):
        scene = r.get("primary_scene", "unknown")
        obj_count = r.get("object_count", 0)
        print(f"Image {i+1}: scene={scene}, objects={obj_count}")
        if r.get("objects"):
            objs = [o["object"] for o in r["objects"][:5]]
            print(f"  Objects: {objs}")

    analyzer.clear_cache()
    print("\nTest complete.")

if __name__ == "__main__":
    main()
