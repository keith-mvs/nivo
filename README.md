# Image Engine

An intelligent photo management system that analyzes, renames, deduplicates, and reformats images with GPU-accelerated ML-powered insights.

## Features

### 1. Multi-Level Analysis
- **Basic**: EXIF metadata extraction (date, camera, GPS, settings)
- **Advanced**: Perceptual hashing, quality scoring, blur/sharpness detection
- **ML-Powered**: Scene detection, object recognition, automatic tagging

### 2. Intelligent Renaming
- Date-based naming from EXIF data (YYYY-MM-DD_HHMMSS format)
- Fallback to file modification date if EXIF unavailable
- Collision handling with sequential suffixes
- Preview mode before renaming

### 3. Deduplication
- Hash-based exact duplicate detection (MD5/SHA256)
- Preserves highest quality original
- Safe mode with duplicate review before deletion

### 4. Format Conversion
- Convert to most compatible formats (JPEG for photos, PNG for graphics)
- Preserve EXIF data during conversion
- Quality optimization while maintaining visual fidelity
- Support for RAW, HEIC, WebP, and other formats

### 5. Tag Embedding
- Write detected tags directly to image EXIF/IPTC metadata
- Tags include: scenes, objects, quality metrics, location
- Searchable via Windows Explorer, macOS Finder, and photo apps

## Repository Structure

```
nivo/
├── src/                        # Production source code
│   ├── core/                   # Core domain logic
│   │   ├── analyzers/          # Image analyzers
│   │   │   ├── metadata.py     # EXIF/IPTC extraction
│   │   │   ├── content.py      # Perceptual hashing, quality analysis
│   │   │   ├── ml_vision.py    # ML-based scene/object detection (DETR)
│   │   │   └── ml_vision_yolo.py  # YOLO-optimized analyzer (RECOMMENDED)
│   │   ├── processors/         # Image processors
│   │   │   ├── deduplicator.py # Hash-based duplicate detection
│   │   │   ├── renamer.py      # Intelligent date-based renaming
│   │   │   ├── formatter.py    # Format conversion
│   │   │   └── tagger.py       # Metadata embedding
│   │   ├── database/           # Database models
│   │   ├── utils/              # Shared utilities
│   │   │   ├── image_io.py     # Image reading/writing utilities
│   │   │   ├── hash_utils.py   # Hashing functions
│   │   │   ├── config.py       # Configuration management
│   │   │   └── gpu_monitor.py  # GPU monitoring
│   │   └── engine.py           # Main orchestration engine
│   ├── adapters/               # External service integrations
│   │   └── nvidia_build/       # NVIDIA Build API client
│   ├── api/                    # Public API interfaces
│   └── ui/                     # User interfaces
│       └── cli.py              # Command-line interface
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests (GPU, API, benchmarks)
├── docs/                       # Documentation
├── scripts/                    # Automation utilities
│   ├── dev/                    # Development scripts
│   └── ci/                     # CI/CD scripts
├── config/                     # Configuration files
│   ├── default_config.yaml     # Default configuration
│   └── yolo_config.yaml        # YOLO-optimized config (RECOMMENDED)
├── infra/                      # Infrastructure (terraform, k8s)
├── requirements.txt            # Python dependencies
└── setup.py                    # Package setup
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Process a directory of photos
```bash
python -m src.ui.cli process ./photos --analyze --dedupe --rename --format
```

### Analyze photos only
```bash
python -m src.ui.cli analyze ./photos --output report.json
```

### Deduplicate
```bash
python -m src.ui.cli dedupe ./photos --dry-run
```

### Rename with date-based pattern
```bash
python -m src.ui.cli rename ./photos --pattern "{date}_{seq}" --preview
```

### Convert formats
```bash
python -m src.ui.cli convert ./photos --target-format jpg --quality 95
```

### System info (GPU status)
```bash
python -m src.ui.cli info
```

## Configuration

Edit `config/default_config.yaml` to customize:
- ML model preferences (YOLO vs DETR)
- Naming patterns and formats
- Quality thresholds for analysis
- Tag categories to embed
- Hash algorithms for deduplication

For best performance, use `config/yolo_config.yaml` which enables the YOLO-optimized analyzer.

## Dependencies

- **Pillow**: Image processing and basic EXIF
- **piexif**: Advanced EXIF manipulation
- **imagehash**: Perceptual hashing for similarity detection
- **opencv-python**: Image quality analysis
- **transformers** + **torch**: ML-based scene/object detection
- **ultralytics**: YOLOv8 object detection
- **iptcinfo3**: IPTC metadata embedding
- **click**: CLI framework
- **PyYAML**: Configuration management
- **tqdm**: Progress bars

## ML Models

The system uses:
- **CLIP** (OpenAI): Scene understanding and semantic tagging
- **YOLOv8-nano** (Ultralytics): Fast object detection (RECOMMENDED)
- **DETR** (Facebook): Transformer-based object detection
- **Image quality models**: Blur detection, aesthetic scoring

Models download automatically on first run (~500MB-2GB).

## Performance

| Config | Speed (5,000 images) | GPU Memory | Object Detection |
|--------|---------------------|------------|------------------|
| YOLO (recommended) | ~10-15 min | ~1-1.5GB | 3-5x faster |
| Baseline (DETR) | ~20-30 min | ~0.8-1GB | Standard |
| No ML | ~5-10 min | N/A | None |

## Development

Run tests:
```bash
# Integration tests
python tests/integration/test_yolo_quick.py
python tests/integration/test_gpu.py

# Benchmarks
python scripts/dev/benchmark_ml_performance.py
```

## License

See LICENSE file.
