# Getting Started with Video Engine

## Quick Setup (5 minutes)

### 1. Install Dependencies

```powershell
# Activate your GPU-enabled virtual environment
.\.venv\Scripts\Activate.ps1

# Install video processing libraries
python -m pip install -r requirements-video.txt
```

### 2. Verify Installation

```powershell
# Run quick test
python test_video_quick.py
```

Expected output:
```
[SUCCESS] All tests passed!
```

### 3. Analyze Your First Video

```powershell
# Analyze a single video
python -m src.cli video-analyze "path\to\your\video.mp4" -o analysis.json

# View the results
cat analysis.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

## Common Use Cases

### Use Case 1: Organize Video Library

**Goal**: Tag and classify all videos for easy searching

```powershell
# Analyze entire video directory
python -m src.cli video-analyze "D:\Videos\MyLibrary" -o library_analysis.json

# Review content tags
python -c "import json; data=json.load(open('library_analysis.json')); print([v['content_tags'] for v in data])"
```

**Output**: JSON file with comprehensive tags for each video including:
- Scene types (indoor, outdoor, landscape)
- Objects detected (person, mat, equipment)
- Activities (fitness, cooking, social, product)
- Quality metrics (high-quality, good-quality)
- Technical specs (4K, 1080p, duration)

### Use Case 2: Find Specific Content

**Goal**: Quickly find all fitness/yoga videos

```powershell
# Analyze videos
python -m src.cli video-analyze "D:\Videos" -o videos.json

# Filter for fitness content (PowerShell)
$videos = Get-Content videos.json | ConvertFrom-Json
$fitness = $videos | Where-Object { $_.content_tags.activities -contains "fitness" }
$fitness | Select-Object file_name, @{N='Tags';E={$_.content_tags}} | Format-Table
```

### Use Case 3: Extract Key Moments

**Goal**: Extract representative frames from videos for thumbnails/previews

```powershell
# Extract keyframes from all videos in a folder
Get-ChildItem "D:\Videos\*.mp4" | ForEach-Object {
    $name = $_.BaseName
    python -m src.cli video-extract $_.FullName -o ".\keyframes\$name" --keyframes
}
```

**Result**: Folders with keyframes extracted at scene changes

### Use Case 4: Quality Assessment

**Goal**: Find low-quality videos that need reprocessing

```powershell
# Analyze with quality metrics
python -m src.cli video-analyze "D:\Videos" -o quality_check.json

# Find low-quality videos (PowerShell)
$videos = Get-Content quality_check.json | ConvertFrom-Json
$low_quality = $videos | Where-Object { $_.temporal_analysis.quality_stats.average -lt 60 }
$low_quality | Select-Object file_name, @{N='AvgQuality';E={$_.temporal_analysis.quality_stats.average}} | Format-Table
```

## Understanding the Output

### Sample Analysis Result

```json
{
  "file_name": "yoga_morning.mp4",
  "file_size_mb": 145.2,
  "duration_sec": 1245.5,
  "duration_formatted": "20:45",
  "resolution": "1920x1080",
  "fps": 30.0,
  "codec": "avc1",

  "content_tags": {
    "scenes": ["indoor", "room"],
    "objects": ["person", "mat", "bottle"],
    "activities": ["fitness", "social"],
    "quality": ["high-quality"],
    "technical": ["1080p", "medium-length"]
  },

  "temporal_analysis": {
    "dominant_scenes": [
      {"scene": "indoor", "count": 45, "percentage": 90.0}
    ],
    "frequent_objects": [
      {"object": "person", "count": 125},
      {"object": "mat", "count": 98}
    ],
    "quality_stats": {
      "average": 82.5,
      "min": 75.0,
      "max": 92.0
    },
    "scene_count": 50,
    "scene_changes": 12
  }
}
```

### Content Tag Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **scenes** | Visual setting/environment | indoor, outdoor, landscape, room, beach |
| **objects** | Detected physical objects | person, mat, bottle, equipment, product |
| **activities** | Inferred activities | fitness, cooking, outdoor, social, product |
| **quality** | Quality assessment | high-quality, good-quality, low-quality |
| **technical** | Video specifications | 4K, 1080p, 720p, short-clip, long-video |

### Activity Inference Logic

The system infers activities based on scene and object combinations:

- **fitness**: gym, yoga, mat, exercise, workout
- **cooking**: kitchen, food, cooking, dining
- **outdoor**: outdoor, nature, landscape, park, beach
- **social**: multiple people detected
- **product**: product, bottle, package, box

## Configuration Tips

### Adjust Scene Detection Sensitivity

Edit `config/default_config.yaml`:

```yaml
video:
  keyframe_threshold: 30.0    # Default
  # Lower (20.0) = more keyframes (more scenes detected)
  # Higher (40.0) = fewer keyframes (only major scene changes)
```

### Limit Analysis for Performance

```yaml
video:
  max_frames_per_video: 50    # Default
  # Reduce to 30 for faster analysis
  # Increase to 100 for more detailed analysis
```

### Skip ML Analysis (Faster)

```yaml
analysis:
  ml_analysis: false           # Skip CLIP/DETR models
  content_analysis: true       # Keep quality/blur analysis
```

## Performance Tuning

### For Long Videos (>30 min)

```powershell
# Reduce max frames
python -m src.cli video-analyze "long_video.mp4" --max-frames 30 -o analysis.json
```

### For Batch Processing

```powershell
# Process multiple videos efficiently
$videos = Get-ChildItem "D:\Videos\*.mp4"
$videos | ForEach-Object {
    python -m src.cli video-analyze $_.FullName -o "$($_.BaseName)_analysis.json"
}
```

### GPU Memory Issues

Edit `config/default_config.yaml`:

```yaml
analysis:
  ml_models:
    batch_size: 4              # Reduce from 8 (default)
```

## Next Steps

### Phase 2: NVIDIA Build API

The next phase will add cloud-based models for enhanced detection:

1. **Retail Object Detection**: Identify products, brands, packaging
2. **Vision-Language Models**: Natural language descriptions of video content
3. **Advanced Activity Recognition**: More precise activity classification

To prepare:
1. Sign up for NVIDIA Build API key
2. Set environment variable: `$env:NVIDIA_BUILD_API_KEY = "your-key"`
3. Enable in config: `nvidia_build.enabled: true`

### Phase 3: Video Database

Build a searchable database of your video library:

```powershell
# Coming soon
python -m src.cli video-index "D:\Videos" --database videos.db
python -m src.cli video-search --activity fitness --quality high
```

## Troubleshooting

### "Cannot open video" Error

**Cause**: Unsupported codec or corrupted file

**Solution**:
```powershell
# Check video info
python -m src.cli video-info "problem_video.mp4"

# Re-encode if needed (requires ffmpeg)
ffmpeg -i problem_video.mp4 -c:v libx264 -c:a aac fixed_video.mp4
```

### Slow Analysis

**Solutions**:
1. Reduce max frames: `--max-frames 20`
2. Use uniform sampling: `--uniform`
3. Skip ML: Set `analysis.ml_analysis: false` in config

### GPU Not Being Used

**Check**:
```powershell
# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Monitor GPU during analysis
nvidia-smi -l 1
```

**Fix**: Ensure `analysis.ml_models.use_gpu: true` in config

## Support

- Full documentation: [VIDEO_ENGINE.md](VIDEO_ENGINE.md)
- Architecture details: See `src/analyzers/video_analyzer.py`
- Issue tracker: [GitHub Issues](https://github.com/anthropics/claude-code/issues)

## Examples Gallery

### Example 1: Family Video Library

```powershell
# Analyze all family videos
python -m src.cli video-analyze "D:\Videos\Family" -o family_videos.json

# Find videos with people (social activities)
$videos = Get-Content family_videos.json | ConvertFrom-Json
$social = $videos | Where-Object { $_.content_tags.activities -contains "social" }
```

### Example 2: Yoga Session Archive

```powershell
# Analyze yoga sessions
python -m src.cli video-analyze "D:\Videos\Yoga" -o yoga_analysis.json

# Extract keyframes for thumbnails
Get-ChildItem "D:\Videos\Yoga\*.mp4" | ForEach-Object {
    python -m src.cli video-extract $_.FullName -o ".\thumbnails\$($_.BaseName)" --keyframes
}
```

### Example 3: Product Video Catalog

```powershell
# Analyze product videos
python -m src.cli video-analyze "D:\Videos\Products" -o products.json

# Filter high-quality product videos
$videos = Get-Content products.json | ConvertFrom-Json
$quality_products = $videos | Where-Object {
    ($_.content_tags.activities -contains "product") -and
    ($_.content_tags.quality -contains "high-quality")
}
```

Happy video organizing!
