# CLAUDE.md

GPU-accelerated photo management system with comprehensive tagging and metadata embedding.

## Image Library Location

**Primary:** `C:\Users\kjfle\Pictures`

## Quick Start

```bash
# System info (GPU status)
python -m src.ui.cli info

# RECOMMENDED: Full analysis with YOLO + tag generation
python -m src.ui.cli analyze "C:\Users\kjfle\Pictures" --config config/yolo_config.yaml -o analysis.json
python scripts/dev/generate_tags.py

# Test tag embedding (dry-run, first 10 images)
python scripts/dev/embed_tags.py --test 10

# Execute tag embedding with backups
python scripts/dev/embed_tags.py --execute

# View results summary
python scripts/dev/view_results.py

# Fast GPU/ML test (10 images, ~1-2 min)
python tests/integration/test_yolo_quick.py
```

## Repository Structure

```
nivo/
├── src/                    # Production source code
│   ├── core/               # Core domain logic
│   │   ├── analyzers/      # Image/video analyzers (metadata, content, ML)
│   │   ├── processors/     # Image processors (dedupe, rename, tag, format)
│   │   ├── database/       # Database models and access
│   │   ├── utils/          # Shared utilities (config, image_io, gpu_monitor)
│   │   └── engine.py       # Main orchestration engine
│   ├── adapters/           # External service integrations
│   │   └── nvidia_build/   # NVIDIA Build API client
│   ├── api/                # Public API interfaces (future)
│   └── ui/                 # User interfaces
│       └── cli.py          # Command-line interface
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests (GPU, API, benchmarks)
├── docs/                   # Documentation
├── scripts/                # Automation utilities
│   ├── dev/                # Development scripts (benchmarks, analysis)
│   └── ci/                 # CI/CD scripts
├── config/                 # Configuration files
├── infra/                  # Infrastructure (terraform, k8s)
└── README.md               # Project overview
```

## Architecture

### Three-Phase Analysis Pipeline

**Phase 1: Metadata** (`src/core/analyzers/metadata.py`)
- CPU-bound, ~400-900 images/sec
- EXIF, GPS, camera info via piexif
- Single-threaded sequential

**Phase 2: Content** (`src/core/analyzers/content.py`)
- CPU-bound, ~50-100 images/sec
- Quality scoring, blur detection, color analysis with OpenCV
- ThreadPoolExecutor parallelization
- **CRITICAL**: Uses `cv2.imdecode()` for Windows Unicode filenames

**Phase 3: ML Vision** (GPU-accelerated)

**YOLO Analyzer** (`src/core/analyzers/ml_vision_yolo.py`) - **RECOMMENDED**:
- YOLOv8-nano: 3-5x faster object detection vs DETR
- CLIP for scene classification
- Batch size 16 (vs 8 baseline) for better GPU utilization
- FP16 precision via AMP for additional speedup
- Enable: `use_yolo: true` in config or use `config/yolo_config.yaml`

Standard Analyzer (`src/core/analyzers/ml_vision.py`):
- DETR for object detection (slower)
- CLIP for scene classification
- Batch size 8
- Graceful degradation on failure

### Model Loading Pattern (CRITICAL)

Lazy loading with sentinel value for failure handling:
```python
# Models are None initially
self._clip_model = None

# Load on first use
if self._clip_model is None:
    self._load_clip_model()  # Loads once per session

if self._clip_model == "FAILED":  # Sentinel value
    return {"primary_scene": "unknown"}  # Graceful degradation
```

**Why**: PyTorch 2.5.1 blocks `pytorch_model.bin` (CVE-2025-32434). Uses safetensors first, sentinel prevents reload attempts.

## Advanced Features

### SQLite Analysis Cache (✓ Implemented)

**File:** `src/core/database/analysis_cache.py`

Incremental analysis with automatic cache invalidation:
```python
from src.core.database import get_cache

cache = get_cache()  # Global singleton
cached = cache.get_cached_batch(file_paths)  # {path: result}
cache.cache_batch(analysis_results)  # Store new results
stats = cache.get_stats()  # entries, size, hit rate
```

**Configuration:**
```yaml
analysis:
  use_cache: true
  cache_path: ".nivo_cache.db"  # or null for default
```

**Features:**
- Fingerprint-based change detection (MD5 of path+size+mtime)
- Automatic invalidation when files modified
- Batch operations for efficiency
- Typical speedup: 2-3x on subsequent runs with 60-90% hit rate

### Face Detection & Recognition (✓ Implemented)

**File:** `src/core/analyzers/face_detection.py`

**Dependencies:** `dlib`, `face-recognition` (optional, installed)

```python
from src.core.analyzers import FaceDetector, is_face_detection_available

if is_face_detection_available():
    detector = FaceDetector(model="hog")  # or "cnn" for GPU
    result = detector.detect_faces(image_path)
    # Returns: {face_count, face_locations, has_faces, landmarks, encodings}

    # Batch processing
    results = detector.detect_batch(image_paths)

    # Face comparison & clustering
    is_match, distance = detector.compare_faces(enc1, enc2, tolerance=0.6)
    labels = detector.cluster_faces(encodings, tolerance=0.6)
```

**Models:**
- `hog`: CPU-based, ~100-200ms/image, good accuracy
- `cnn`: GPU-based, ~50-100ms/image, excellent accuracy (requires CUDA)

**Use Cases:**
- People counting
- Portrait detection for auto-cropping
- Face clustering (group photos by person)
- Privacy detection (blur faces)

### Perceptual Duplicate Detection (✓ Implemented)

**File:** `src/core/processors/deduplicator.py`

Find visually similar images (not just byte-identical):
```python
from src.core.processors.deduplicator import Deduplicator

dedup = Deduplicator()

# Perceptual similarity
similar_groups = dedup.find_similar(
    file_paths=image_list,
    threshold=8,           # Hamming distance (0=identical, 64=different)
    hash_type="phash",     # phash, dhash, average_hash, whash
    show_progress=True
)
# Returns: {"group_0": ["img1.jpg", "img2.jpg"], "group_1": [...]}

# Exact duplicates (SHA256)
exact_dupes = dedup.find_duplicates(file_paths)
```

**Threshold Guide:**
- 0-4: Nearly identical (same shot, different compression)
- 4-8: Very similar (burst mode, minor edits)
- 8-12: Visually similar (same scene, different angle/time)

**Algorithm:** Union-find clustering with parallel perceptual hashing

## Critical Technical Constraints

### Windows Unicode Filenames
OpenCV's `cv2.imread()` fails on Unicode paths. Solution:
```python
with open(image_path, 'rb') as f:
    img_array = np.frombuffer(f.read(), dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
```

### PyTorch CUDA Setup
- Hardware: RTX 2080 Ti
- Runtime: CUDA 12.1 required
- Install: PyTorch 2.5.1+cu121 via `--index-url https://download.pytorch.org/whl/cu121`

### Configuration Format
`config/default_config.yaml` uses **dict format**, not lists:
```yaml
tagging:
  categories:
    scene: true      # NOT a list
    objects: true
```

## Performance

| Config | Speed (5,000 images) | GPU Memory | Object Detection |
|--------|---------------------|------------|------------------|
| YOLO (recommended) | ~10-15 min | ~1-1.5GB | 3-5x faster |
| Baseline (DETR) | ~20-30 min | ~0.8-1GB | Standard |
| No ML | ~5-10 min | N/A | None |

**Metadata**: ~400-900 img/sec (CPU)
**Content**: ~50-100 img/sec (CPU multi-thread)
**ML YOLO**: ~1-3 sec/batch (16 images)
**ML DETR**: ~4-10 sec/batch (8 images)

## Important Files

**Core:**
- `src/ui/cli.py` - Entry point
- `src/core/engine.py` - Analyzer selection (YOLO > TensorRT > Baseline)
- `src/core/analyzers/ml_vision_yolo.py` - YOLO analyzer (RECOMMENDED)
- `src/core/analyzers/ml_vision.py` - Baseline DETR analyzer

**Adapters:**
- `src/adapters/nvidia_build/` - NVIDIA Build API integration

**Config:**
- `config/yolo_config.yaml` - YOLO-optimized (RECOMMENDED)
- `config/default_config.yaml` - Baseline system config

**Testing:**
- `tests/integration/test_yolo_quick.py` - Fast verification
- `tests/integration/test_yolo_vs_detr.py` - Performance benchmark

**Utilities:**
- `src/core/utils/config.py` - Configuration management
- `src/core/utils/gpu_monitor.py` - Background GPU monitoring

## Common Issues

**CLIP model loading repeatedly**
Check sentinel value in `ml_vision.py:_classify_scene()`. Model should load once.

**Unicode print error**
Windows console may not support Unicode. Use ASCII: `print("[OK]")` not `[check mark]`

**GPU not being used**
1. Verify: `python -m src.ui.cli info` shows CUDA available
2. Check nvidia-smi shows Python process
3. Confirm batch processing in Phase 3

**Process won't die**
```bash
tasklist | findstr python
taskkill /F /PID <process_id>
```

## Self-Improvement Loop

Structured logging to `logs/`:
- `YYYYMMDD_HHMMSS.log` - Human-readable
- `YYYYMMDD_HHMMSS.json` - Machine-parseable
- `errors.jsonl` - Persistent error database

Feedback loop (`src/core/utils/feedback_loop.py`) auto-fixes:
- Unicode encoding errors -> Add UTF-8
- OpenCV imread failures -> Switch to cv2.imdecode
- GPU OOM -> Reduce batch size
- Division by zero -> Add checks
