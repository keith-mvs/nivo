# Quick Start Guide

## Installation (5 minutes)

```bash
# Install dependencies
pip install -r requirements.txt
```

On first run, ML models will download automatically (~1-2 GB). This happens only once.

## Your First Commands

### 1. Check System (verify GPU)

```bash
python -m src.cli info
```

Expected output:
```
=== System Information ===

Config: C:\Users\...\config\default_config.yaml

Enabled Analyzers:
  Metadata: True
  Content: True
  ML/AI: True

ML Configuration:
  Device: cuda  # or 'cpu' if no GPU
  Batch size: 8
```

### 2. Analyze Your Photos

```bash
python -m src.cli analyze "D:\Pictures\Camera Roll"
```

This will:
- Extract EXIF data (dates, camera, GPS)
- Assess quality (sharpness, exposure)
- Detect scenes and objects using ML
- Generate `analysis_report.json`

### 3. Preview Full Pipeline (Safe)

```bash
python -m src.cli process "D:\Pictures\Camera Roll" \
  --analyze \
  --dedupe \
  --rename \
  --embed-tags \
  --convert \
  --dry-run
```

This shows what **would** happen without making changes.

### 4. Execute Full Pipeline

When ready:

```bash
python -m src.cli process "D:\Pictures\Camera Roll" \
  -o "D:\Pictures\Organized" \
  --analyze \
  --dedupe \
  --rename \
  --embed-tags \
  --convert \
  --execute
```

This will:
1. ✓ Analyze all photos (metadata + ML)
2. ✓ Remove exact duplicates (keeping highest quality)
3. ✓ Rename with date-based names (YYYY-MM-DD_HHMMSS.jpg)
4. ✓ Embed detected tags into EXIF/IPTC metadata
5. ✓ Convert to most compatible formats (JPEG/PNG)
6. ✓ Save to organized output directory

## GPU Acceleration

The system automatically uses your GPU if available:

- **NVIDIA GPU**: PyTorch with CUDA
- **AMD/Intel**: CPU processing (still fast with multi-threading)
- **Apple Silicon**: MPS support (if PyTorch configured)

Batch processing maximizes GPU utilization for ML analysis.

## Processing Your Photos

Based on your directory: `D:\Pictures\Camera Roll`

```bash
# Quick analysis (no ML)
python -m src.cli analyze "D:\Pictures\Camera Roll" --no-ml

# Full analysis with ML (recommended)
python -m src.cli analyze "D:\Pictures\Camera Roll"

# Process and organize everything
python -m src.cli process "D:\Pictures\Camera Roll" \
  -o "D:\Pictures\Organized" \
  --analyze --dedupe --rename --embed-tags --convert \
  --execute
```

## Embedded Tags = Easy Search

After processing, your photos will have tags embedded in their metadata. You can search them in:

**Windows Explorer:**
```
tags:landscape
tags:outdoor
tags:person
```

**File Explorer search bar:**
- Just type the tag name in the search box
- Photos with that tag will appear

**Photo applications:**
- Tags appear in keywords/metadata fields
- Adobe Lightroom, Google Photos, etc. can all read them

## Performance

On typical hardware:
- **With GPU**: ~2-5 images/second for full ML analysis
- **CPU only**: ~0.5-1 images/second for ML analysis
- **Metadata only**: ~50-100 images/second

For 1000 photos:
- Full ML analysis: ~5-10 minutes (GPU) or ~20-30 minutes (CPU)
- Metadata + quality: ~1-2 minutes
- Deduplication: ~30 seconds
- Renaming: ~10 seconds

## Next Steps

1. Read [USAGE.md](USAGE.md) for detailed examples
2. Customize [config/default_config.yaml](config/default_config.yaml)
3. Try different naming patterns
4. Explore individual commands (analyze, dedupe, rename, convert)

## Troubleshooting

**Models not downloading?**
- Check internet connection
- Models download from HuggingFace on first run

**Out of GPU memory?**
- Reduce `batch_size` in config to 4 or 2
- Or use `--no-ml` for faster processing without ML

**Need help?**
- Check [USAGE.md](USAGE.md) for detailed guide
- Review config in `config/default_config.yaml`

## Important Notes

1. **Pre-trained Models**: No training needed! Models work out-of-the-box
2. **Always use --dry-run first**: Preview changes before executing
3. **Backups**: Keep originals until you verify results
4. **GPU**: Use GPU for 5-10x faster ML processing
5. **Tags**: Embedded tags are searchable in Windows/macOS/photo apps
