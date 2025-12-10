# Image Engine - Usage Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Quick Start

### 1. Analyze Your Photos

The first step is to analyze your photo collection. This extracts metadata, assesses quality, and uses ML to detect scenes and objects.

```bash
# Basic analysis
python -m src.cli analyze "D:\Pictures\Camera Roll"

# Analysis without ML (faster, but less detailed)
python -m src.cli analyze "D:\Pictures\Camera Roll" --no-ml

# Save report to custom location
python -m src.cli analyze "D:\Pictures\Camera Roll" -o my_analysis.json
```

This will create an analysis report with:
- EXIF metadata (date, camera, GPS)
- Quality scores (sharpness, exposure, noise)
- ML tags (scenes, objects detected)
- Perceptual hashes for similarity detection

### 2. Find and Remove Duplicates

```bash
# Preview duplicates (dry run)
python -m src.cli dedupe "D:\Pictures\Camera Roll"

# Remove duplicates, keeping highest quality version
python -m src.cli dedupe "D:\Pictures\Camera Roll" --execute --strategy highest_quality

# Other strategies: oldest, newest, largest
python -m src.cli dedupe "D:\Pictures\Camera Roll" --execute --strategy oldest
```

### 3. Rename Photos with Dates

```bash
# Preview renaming with date-time pattern (default)
python -m src.cli rename "D:\Pictures\Camera Roll"

# Execute rename with custom pattern
python -m src.cli rename "D:\Pictures\Camera Roll" --pattern "{date}_{time}" --execute

# Rename and copy to new directory
python -m src.cli rename "D:\Pictures\Camera Roll" -o "D:\Pictures\Organized" --execute

# Include camera model in filename
python -m src.cli rename "D:\Pictures\Camera Roll" --pattern "{datetime}_{camera}" --execute
```

Available pattern variables:
- `{date}` - Date (YYYY-MM-DD)
- `{time}` - Time (HHMMSS)
- `{datetime}` - Date and time (YYYY-MM-DD_HHMMSS)
- `{camera}` - Camera model
- `{tags}` - ML-detected tags
- `{year}`, `{month}`, `{day}` - Individual date components

### 4. Convert to Compatible Formats

```bash
# Convert to optimal formats (auto-detected)
python -m src.cli convert "D:\Pictures\Camera Roll" -o "D:\Pictures\Converted"

# Force all to JPEG
python -m src.cli convert "D:\Pictures\Camera Roll" --format jpg --quality 95

# Convert to PNG (for graphics/screenshots)
python -m src.cli convert "D:\Pictures\Camera Roll" --format png
```

The system automatically:
- Converts HEIC → JPEG
- Converts RAW formats → JPEG
- Preserves PNG for images with transparency
- Optimizes JPEG quality while preserving visual quality

### 5. Complete Pipeline (All Steps)

Run everything in one command:

```bash
# Preview complete pipeline
python -m src.cli process "D:\Pictures\Camera Roll" \
  --analyze \
  --dedupe \
  --rename \
  --embed-tags \
  --convert \
  --dry-run

# Execute complete pipeline with output directory
python -m src.cli process "D:\Pictures\Camera Roll" \
  -o "D:\Pictures\Processed" \
  --analyze \
  --dedupe \
  --rename \
  --embed-tags \
  --convert \
  --execute
```

## GPU Acceleration

### Check GPU Status

```bash
python -m src.cli info
```

This shows:
- Whether GPU is detected and being used
- GPU memory usage
- Available ML models

### GPU Memory Management

The system automatically:
- Uses batch processing to maximize GPU efficiency
- Clears GPU cache between operations
- Falls back to CPU if GPU memory is insufficient

For systems with limited GPU memory, reduce batch size in `config/default_config.yaml`:

```yaml
analysis:
  ml_models:
    batch_size: 4  # Reduce from 8 if getting OOM errors
```

## Embedding Tags in Photos

Tags and metadata are embedded directly into image files, making them searchable in:
- Windows Explorer
- macOS Finder
- Google Photos
- Adobe Lightroom
- Any photo app that reads EXIF/IPTC

Tags include:
- Detected scenes (landscape, portrait, indoor, outdoor)
- Detected objects (person, dog, car, etc.)
- Quality scores
- Location (from GPS if available)
- Date/time

To search embedded tags:
- **Windows**: Use search like `tags:landscape` in File Explorer
- **macOS**: Use Spotlight search with tag names
- **Photo apps**: Tags appear in keywords/metadata fields

## Configuration

Edit `config/default_config.yaml` to customize:

```yaml
# Analysis settings
analysis:
  extract_metadata: true
  content_analysis: true
  ml_analysis: true
  ml_models:
    use_gpu: true
    batch_size: 8

# Renaming pattern
renaming:
  pattern: "{datetime}"
  datetime_format: "%Y-%m-%d_%H%M%S"

# Format settings
formatting:
  photo_format: "jpg"
  jpeg_quality: 95

# Tagging
tagging:
  embed_tags: true
  min_confidence: 0.6
```

## Training on Your Photos

The ML models (CLIP and DETR) are **pre-trained** and work out-of-the-box without training. They use:
- **CLIP** (OpenAI): Zero-shot scene understanding - works on any image without training
- **DETR** (Facebook): Pre-trained object detection - recognizes 80+ object categories

These models will automatically analyze photos in `D:\Pictures\Camera Roll` when you run analysis.

**No training required!** The models are ready to use and will:
1. Automatically download on first run (~500MB-2GB)
2. Use GPU acceleration if available
3. Process photos in batches for efficiency

## Advanced Examples

### Organize Photos by Date into Folders

```python
# Custom script using the engine
from src.engine import ImageEngine
from pathlib import Path
import shutil

engine = ImageEngine()
image_paths = engine.scan_directory("D:\Pictures\Camera Roll")
analysis = engine.analyze_images(image_paths)

for data in analysis:
    if 'datetime_original' in data:
        dt = data['datetime_original']
        # Create year/month folder structure
        dest_dir = Path(f"D:/Pictures/Organized/{dt.year}/{dt.month:02d}")
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy file
        src = data['file_path']
        dest = dest_dir / Path(src).name
        shutil.copy2(src, dest)
```

### Find All Photos with Specific Tags

```python
from src.engine import ImageEngine

engine = ImageEngine()
image_paths = engine.scan_directory("D:\Pictures\Camera Roll")
analysis = engine.analyze_images(image_paths)

# Find all landscape photos
landscapes = [
    data['file_path'] for data in analysis
    if 'tags' in data and 'landscape' in data['tags']
]

print(f"Found {len(landscapes)} landscape photos")
```

### Batch Process Multiple Directories

```bash
# Process multiple directories
for dir in "D:\Pictures\2023" "D:\Pictures\2024" "D:\Pictures\Camera Roll"; do
  python -m src.cli process "$dir" \
    --analyze \
    --dedupe \
    --rename \
    --embed-tags \
    --convert \
    -o "D:\Pictures\Organized" \
    --execute
done
```

## Performance Tips

1. **Use GPU**: Ensure PyTorch can access your GPU for 5-10x faster ML analysis
2. **Batch Processing**: Larger batch sizes use GPU more efficiently (if you have enough memory)
3. **Skip ML for Speed**: Use `--no-ml` flag if you only need metadata and quality analysis
4. **Parallel CPU**: The system uses multiple CPU cores automatically for metadata extraction
5. **Quick Hash**: Deduplication uses quick hash screening first, then full hash only for candidates

## Troubleshooting

### Out of Memory (GPU)

Reduce batch size in config:
```yaml
ml_models:
  batch_size: 4  # Or even 2 for very limited memory
```

### HEIC/RAW Files Not Loading

Install optional dependencies:
```bash
pip install pillow-heif rawpy
```

### Slow Performance

1. Check GPU is being used: `python -m src.cli info`
2. Disable ML for faster processing: `--no-ml`
3. Process smaller batches of files at a time

### Tags Not Embedded

Ensure iptcinfo3 is installed:
```bash
pip install iptcinfo3
```

On Windows, you may need to install `python-xmp-toolkit` separately.
