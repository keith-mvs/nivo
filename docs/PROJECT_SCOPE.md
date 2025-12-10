# Image Engine - Project Scope

## Mission Statement

Build a GPU-accelerated photo management system capable of:
1. Analyzing image content using ML models (YOLO, CLIP)
2. Generating comprehensive hierarchical tags across 10 categories
3. Embedding metadata into EXIF/IPTC for universal compatibility
4. Operating with high performance (3-5x faster than baseline with YOLO)

---

## Scope Parameters

### Analysis Pipeline (3 Phases)

| Phase | Focus | Performance | Parallelization |
|-------|-------|-------------|-----------------|
| Metadata | EXIF, GPS, camera info | 400-900 img/sec | Single-threaded |
| Content | Quality, blur, color | 50-100 img/sec | ThreadPoolExecutor |
| ML Vision | Objects, scenes (YOLO/CLIP) | 1-3 sec/batch (16 imgs) | GPU-accelerated |

### ML Models

| Model | Purpose | Speed | Accuracy |
|-------|---------|-------|----------|
| YOLOv8-nano | Object detection | 3-5x faster than DETR | High |
| DETR | Object detection (baseline) | Standard | High |
| CLIP | Scene classification | Fast | High |
| OpenCV | Quality/blur/color | Very fast | Good |

### Tag Categories (10 Hierarchical)

1. **Scene** - Primary scene + hierarchy (outdoor, indoor, event, subject, travel)
2. **Objects** - Detected objects + semantic groups (people, vehicles, animals, etc.)
3. **Quality** - Score, sharpness, blur levels
4. **Color** - Dominant colors, temperature (warm/cool), brightness, contrast
5. **Temporal** - Year, season, time of day, weekend detection
6. **Technical** - ISO, aperture, shutter, flash, camera model
7. **Format** - Resolution, aspect ratio, orientation, 4K+
8. **People** - Count detection (single, couple, small_group, crowd, no_people)
9. **Location** - GPS, hemisphere, indoor/outdoor, urban indicators
10. **Mood** - Inferred atmosphere (cheerful, moody, romantic, peaceful, energetic)

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- Additional formats via OpenCV (BMP, TIFF, WebP)

### Performance Targets

| Configuration | Speed (5,000 images) | GPU Memory | Notes |
|---------------|---------------------|------------|-------|
| YOLO (recommended) | 10-15 min | ~1-1.5GB | 3-5x faster |
| Baseline (DETR) | 20-30 min | ~0.8-1GB | Standard |
| No ML | 5-10 min | N/A | Metadata + content only |

---

## Key Technical Decisions

### Windows Unicode Handling
**Issue:** OpenCV's `cv2.imread()` fails on Unicode paths
**Solution:** Use `cv2.imdecode()` with file buffer
```python
with open(image_path, 'rb') as f:
    img_array = np.frombuffer(f.read(), dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
```

### PyTorch Model Loading
**Issue:** PyTorch 2.5.1 blocks `pytorch_model.bin` (CVE-2025-32434)
**Solution:** Lazy loading with sentinel value for failure handling
```python
self._clip_model = None  # Lazy load
if self._clip_model is None:
    self._load_clip_model()
if self._clip_model == "FAILED":  # Sentinel
    return {"primary_scene": "unknown"}
```

### GPU Acceleration
- **Hardware:** NVIDIA RTX 2080 Ti (11GB VRAM)
- **CUDA:** 12.1
- **PyTorch:** 2.5.1+cu121
- **Batch Processing:** 16 images/batch (YOLO) vs 8 (DETR)
- **FP16 Precision:** AMP for additional speedup

---

## Architecture Components

### Core Engine
- `src/engine.py` - Analyzer selection (YOLO > TensorRT > Baseline)
- `src/cli.py` - Command-line interface
- `config/yolo_config.yaml` - YOLO-optimized config
- `config/default_config.yaml` - Baseline config

### Analyzers
- `src/analyzers/metadata.py` - EXIF/GPS extraction
- `src/analyzers/content.py` - Quality/blur/color analysis
- `src/analyzers/ml_vision_yolo.py` - YOLO analyzer (recommended)
- `src/analyzers/ml_vision.py` - Baseline DETR analyzer

### Processing
- `src/processors/tag_generator.py` - 10-category tag generation
- `src/processors/tagger.py` - EXIF/IPTC embedding

### Utilities
- `src/utils/logger.py` - Structured logging
- `src/utils/feedback_loop.py` - Autonomous improvement
- `src/utils/gpu_monitor.py` - Background GPU monitoring
- `src/utils/config.py` - Configuration management

---

## Image Location

**Primary Path:** `C:\Users\kjfle\Pictures`

**Structure:**
- `/jpeg` - JPEG images
- `/png` - PNG images
- `/Camera Roll` - Mobile photos
- Various subdirectories

---

## Tag Embedding Strategy

### Metadata Format
- **IPTC Keywords** - Flat tags (all categories merged)
- **IPTC Title** - Primary scene
- **IPTC Description** - Quality score + category summary

### Safety Features
- **Dry-run mode** - Preview without modification (default)
- **Test mode** - Process only first N images
- **Automatic backups** - Creates `.original` files
- **Rollback support** - Git tags for safe rollback

### Usage
```bash
# Dry-run (preview)
python embed_tags.py --test 10

# Execute with backups
python embed_tags.py --execute

# Execute without backups (dangerous)
python embed_tags.py --execute --no-backup
```

---

## Dependencies

### Core
- Python 3.11+
- OpenCV (cv2)
- NumPy
- piexif (EXIF reading/writing)
- iptcinfo3 (IPTC metadata)

### ML (Optional)
- PyTorch 2.5.1+cu121
- transformers (Hugging Face)
- ultralytics (YOLO)

### GPU
- CUDA 12.1
- NVIDIA GPU with 4GB+ VRAM

---


---

## Constraints & Considerations

### Technical
- Windows Unicode filename support
- GPU memory limitations (11GB RTX 2080 Ti)
- Batch size optimization for GPU utilization
- Model loading performance (lazy loading)

### Data
- EXIF data may be incomplete or missing
- GPS coordinates not always available
- Image quality varies significantly
- Unicode characters in filenames/paths

### Performance
- Metadata phase: CPU-bound, sequential
- Content phase: CPU-bound, parallelized
- ML phase: GPU-bound, batched
- Overall bottleneck: ML Vision phase

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Analysis Speed (YOLO) | <15 min for 5K images | ✅ 10-15 min |
| Tag Categories | 10 comprehensive | ✅ 10 implemented |
| Tag Accuracy | >80% relevant | ✅ High quality |
| GPU Utilization | >80% during ML phase | ✅ Good |
| Tag Embedding | EXIF/IPTC compatible | ✅ Ready |

---

## Open Questions for Refinement

1. **Deduplication:** Integrate perceptual hashing for duplicate detection?
2. **Face Recognition:** Add face detection/recognition capabilities?
3. **Cloud Storage:** Support for cloud photo libraries (Google Photos, iCloud)?
4. **Organization:** Auto-organize by tags/scenes/dates?
5. **Web Interface:** Build web UI for browsing/searching tagged photos?

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2024-XX-XX | Initial MVP with baseline DETR |
| 0.2 | 2025-11-XX | Added YOLO analyzer (3-5x speedup) |
| 0.3 | 2025-12-01 | Added 10-category tag generator |
| 0.4 | 2025-12-01 | Separated video-engine, added tag embedding |
