# Video Library Organization Guide

## Quick Reference

### Your Video Library

- **Location**: `C:\Users\kjfle\Videos`
- **Total Videos**: 1,817 videos
- **Database**: `video_library.db`
- **Status**: Ready for full analysis

### Quick Commands

```powershell
# Analyze full library (resumable, runs in batches)
python analyze_full_library.py "C:\Users\kjfle\Videos"

# Search for high-quality videos
python search_videos.py --quality high-quality --limit 20

# Find fitness/yoga videos
python search_videos.py --activity fitness

# Find videos by duration
python search_videos.py --min-duration 60 --max-duration 300

# Show library statistics
python search_videos.py --stats

# List all available tags
python search_videos.py --list-tags
```

## Full Library Analysis

### Running the Analysis

The batch analyzer processes all 1,817 videos in chunks with automatic progress tracking:

```powershell
# Start analysis (default: 100 videos per batch)
python analyze_full_library.py "C:\Users\kjfle\Videos"

# Custom batch size (larger batches = faster, more memory)
python analyze_full_library.py "C:\Users\kjfle\Videos" --batch-size 50

# Resume after interruption (automatic)
python analyze_full_library.py "C:\Users\kjfle\Videos"
```

### Progress Tracking

- **Auto-saves**: Progress saved after each batch
- **Resumable**: Interrupt anytime (Ctrl+C), resume with same command
- **Progress file**: `analysis_progress.json`
- **Database**: Updates incrementally

### Estimated Time

| Batch Size | Est. Time per Batch | Total Time (1817 videos) |
|------------|---------------------|--------------------------|
| 50 videos  | ~15-20 min         | ~9-12 hours              |
| 100 videos | ~30-40 min         | ~9-12 hours              |
| 200 videos | ~60-80 min         | ~9-12 hours              |

**Note**: Time varies based on video length and resolution. 4K videos take longer.

### Batch Processing Recommendations

**For overnight processing**:
```powershell
# Run with default settings, let it process overnight
python analyze_full_library.py "C:\Users\kjfle\Videos"
```

**For daytime processing with breaks**:
```powershell
# Smaller batches, easier to pause/resume
python analyze_full_library.py "C:\Users\kjfle\Videos" --batch-size 50
```

## Search Examples

### By Content Type

```powershell
# Product videos
python search_videos.py --activity product --limit 20

# Social/family videos
python search_videos.py --activity social

# Outdoor videos
python search_videos.py --activity outdoor --scene "outdoor scene"
```

### By Quality

```powershell
# High quality only
python search_videos.py --quality high-quality --min-quality-score 85

# Find low quality videos for deletion/reprocessing
python search_videos.py --quality low-quality --max-duration 10
```

### By Technical Specs

```powershell
# 4K videos
python search_videos.py --resolution 4k

# 1080p videos
python search_videos.py --resolution 1080p

# Short clips (<30 seconds)
python search_videos.py --max-duration 30

# Long videos (>5 minutes)
python search_videos.py --min-duration 300
```

### Combined Searches

```powershell
# High-quality fitness videos
python search_videos.py --activity fitness --quality high-quality

# 4K outdoor scenes
python search_videos.py --scene "outdoor scene" --resolution 4k

# Short product demos
python search_videos.py --activity product --max-duration 60
```

### Save Results

```powershell
# Export search results to JSON
python search_videos.py --activity fitness -o fitness_videos.json

# Export high-quality 4K videos
python search_videos.py --resolution 4k --quality high-quality -o 4k_videos.json
```

## Database Schema

### Videos Table

Stores core video metadata:
- File path, name, size
- Duration, resolution, FPS, codec
- Quality statistics (avg, min, max)
- Analysis date and full analysis JSON

### Tags Table

Searchable tags categorized by:
- **scenes**: screenshot, indoor scene, outdoor scene, etc.
- **objects**: person, mat, bottle, etc.
- **activities**: fitness, cooking, outdoor, social, product
- **quality**: high-quality, good-quality, low-quality
- **technical**: 4k, 1080p, 720p, short-clip, medium-length, long-video

## Activity Inference

Videos are automatically tagged with activities based on visual content:

| Activity | Detection Criteria |
|----------|-------------------|
| **fitness** | gym, yoga, mat, exercise, workout |
| **cooking** | kitchen, food, cooking, dining |
| **outdoor** | outdoor, nature, landscape, park, beach |
| **social** | multiple people detected |
| **product** | product, bottle, package, box |

## Performance Optimization

### GPU Usage

- Models load once at startup (saves ~10 seconds per video)
- Batch processing on GPU (8 frames/batch)
- GPU memory: ~0.8-1GB during analysis

### Monitor GPU

```powershell
# Watch GPU in real-time
nvidia-smi -l 1
```

### Reduce Memory Usage

If you encounter GPU memory issues:

```powershell
# Reduce frames analyzed per video
python analyze_full_library.py "C:\Users\kjfle\Videos" --max-frames 20

# Smaller batch size
python analyze_full_library.py "C:\Users\kjfle\Videos" --batch-size 50
```

## Troubleshooting

### Resume Not Working

Delete progress file and restart:
```powershell
rm analysis_progress.json
python analyze_full_library.py "C:\Users\kjfle\Videos"
```

### Database Locked

Close any programs accessing the database:
```powershell
# Check what's using the database
tasklist | findstr python
```

### Out of Memory

1. Reduce batch size: `--batch-size 50`
2. Reduce frames: `--max-frames 20`
3. Close other GPU applications

### Slow Processing

- Expected: ~2-4 seconds per video on average
- 4K videos take longer (~5-10 seconds)
- First video in batch is slower (model loading)

## Next Steps After Analysis

### 1. Explore Your Library

```powershell
# See what you have
python search_videos.py --stats
python search_videos.py --list-tags
```

### 2. Organize by Content

```powershell
# Export fitness videos for workout playlist
python search_videos.py --activity fitness -o fitness.json

# Find family moments
python search_videos.py --activity social -o family.json
```

### 3. Quality Audit

```powershell
# Find low-quality videos
python search_videos.py --quality low-quality -o low_quality.json

# Find duplicates or similar content (manual review)
python search_videos.py --scene screenshot -o screenshots.json
```

### 4. Platform-Specific Exports

```powershell
# Short vertical videos for Instagram/TikTok
python search_videos.py --max-duration 60 --resolution "1080x1920" -o vertical_shorts.json

# High-quality 4K for YouTube
python search_videos.py --resolution 4k --quality high-quality -o youtube_4k.json
```

## Maintenance

### Re-analyze Videos

```powershell
# Re-analyze with fresh start
python analyze_full_library.py "C:\Users\kjfle\Videos" --no-resume
```

### Backup Database

```powershell
# Copy database for backup
cp video_library.db video_library_backup_$(Get-Date -Format 'yyyyMMdd').db
```

### Update Database

When you add new videos:
```powershell
# Run analysis again - will skip already processed videos
python analyze_full_library.py "C:\Users\kjfle\Videos"
```

## Advanced Usage

### Custom Search Queries

For advanced searches, use Python directly:

```python
from src.database.video_db import VideoDatabase

with VideoDatabase("video_library.db") as db:
    # Complex multi-criteria search
    results = db.search(
        categories={
            "activities": ["fitness", "outdoor"],
            "quality": ["high-quality"]
        },
        min_duration=30,
        max_duration=300,
        min_quality=80,
        resolution="4k",
        limit=50
    )

    for video in results:
        print(f"{video['file_name']}: {video['duration_formatted']}")
```

### Export for External Tools

```python
import json
from src.database.video_db import VideoDatabase

with VideoDatabase("video_library.db") as db:
    results = db.search(activity="fitness")

    # Export paths for video editing software
    paths = [r['file_path'] for r in results]
    with open('fitness_videos_paths.txt', 'w') as f:
        f.write('\n'.join(paths))
```

## Support

For issues or questions:
1. Check `VIDEO_ENGINE.md` for technical details
2. Review `GETTING_STARTED_VIDEO.md` for basics
3. See `test_video_quick.py` for system verification

---

**Happy organizing!**
