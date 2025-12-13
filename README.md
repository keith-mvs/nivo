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

## Architecture

The system follows a 6-phase modular architecture:

### Phase 1: Domain Models & Configuration
- Pydantic-validated configuration models (`src/core/models/`)
- Type-safe settings with validation and defaults
- Hierarchical config loading with overrides

### Phase 2: Dependency Injection & Interfaces
- Abstract interfaces for analyzers and monitors (`src/core/interfaces/`)
- Enables testability and component swapping
- GPU monitor abstraction for resource tracking

### Phase 3: ML Analyzer Base Class
- Template Method pattern for analyzer pipeline
- Common batch processing, caching, error handling
- GPU memory management with automatic cleanup

### Phase 4: Factory & Pipeline Decomposition
- `AnalyzerFactory`: Creates analyzers based on config (YOLO > TensorRT > DETR)
- `AnalysisPipeline`: Orchestrates 3-phase analysis (metadata, content, ML)
- Clean separation of concerns and configurable behavior

### Phase 5: Comprehensive Test Suite
- 167+ unit tests with 80%+ coverage
- Integration tests for GPU, API, and real files
- Mock-based testing for ML components

### Phase 6: Performance Optimization
- LRU image cache for repeated access
- Performance metrics tracking (`PerformanceTracker`)
- GPU utilization monitoring and batch optimization

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
│   │   ├── models/             # Domain models (Pydantic)
│   │   │   ├── config_models.py # Configuration schemas
│   │   │   ├── image_data.py   # Image metadata models
│   │   │   └── processor_results.py # Processing result types
│   │   ├── factories/          # Object factories
│   │   │   └── analyzer_factory.py # Analyzer creation
│   │   ├── pipeline/           # Processing pipelines
│   │   │   └── analysis_pipeline.py # 3-phase analysis
│   │   ├── interfaces/         # Abstract interfaces
│   │   ├── database/           # Database models
│   │   ├── utils/              # Shared utilities
│   │   │   ├── config.py       # Configuration management
│   │   │   ├── gpu_monitor.py  # GPU monitoring
│   │   │   ├── filename_generator.py # Unique filename generation
│   │   │   ├── workflow_manager.py # Multi-source file processing
│   │   │   ├── performance_metrics.py # Performance tracking
│   │   │   └── image_cache.py  # LRU image caching
│   │   └── engine.py           # Main orchestration engine
│   ├── adapters/               # External service integrations
│   │   └── nvidia_build/       # NVIDIA Build API client
│   ├── api/                    # Public API interfaces
│   └── ui/                     # User interfaces
│       └── cli.py              # Command-line interface
├── tests/                      # Test suite (167+ tests)
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

## Utilities

### Performance Tracking
```python
from src.core.utils import PerformanceTracker, track_time

# Track inference time
tracker = PerformanceTracker()
with tracker.time("yolo_inference", batch_size=16):
    results = model.predict(batch)

# Get metrics
metrics = tracker.get_metrics("yolo_inference")
print(f"Throughput: {metrics.throughput_per_sec:.1f}/s")

tracker.print_summary()
```

### Multi-Source Workflow
```python
from src.core.utils import create_library_workflow

# Process images from multiple folders
workflow = create_library_workflow(
    source_directories=["D:/Pictures/jpeg", "D:/Pictures/heic"],
    end_directory="D:/Processed",
    backup_directory="D:/Backups",
    retention_days=30,
    recursive=True,
)

# Preview actions
print(workflow.preview())

# Execute
result = workflow.execute(dry_run=False)
print(f"Processed: {result.files_moved} files")
```

### Filename Generation
```python
from src.core.utils import FilenameGenerator, validate_filename

# Generate unique filename
gen = FilenameGenerator()
filename = gen.generate(extension="png")
# -> "img_20251213_143052_a1b2c3d4.png"

# Validate before save
FilenameGenerator.validate_before_save(filepath)  # Raises if invalid
```

## Development

Run tests:
```bash
# Unit tests (167+ tests)
python -m pytest tests/unit/ -v

# Integration tests
python tests/integration/test_yolo_quick.py
python tests/integration/test_phase4_components.py

# Benchmarks
python scripts/dev/benchmark_ml_performance.py
```

## License

See LICENSE file.
