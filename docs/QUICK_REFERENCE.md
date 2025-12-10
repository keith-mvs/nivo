# Video Library - Quick Reference Card

## Essential Commands

### Analysis
```bash
# Full library analysis (resumable)
python analyze_full_library.py "C:\Users\kjfle\Videos"

# Custom batch size
python analyze_full_library.py "C:\Users\kjfle\Videos" --batch-size 50

# Restart from scratch
python analyze_full_library.py "C:\Users\kjfle\Videos" --no-resume
```

### Search & Discovery
```bash
# Library statistics
python search_videos.py --stats

# List all available tags
python search_videos.py --list-tags

# Basic searches
python search_videos.py --activity fitness
python search_videos.py --quality high-quality
python search_videos.py --resolution 4k
python search_videos.py --scene "outdoor scene"

# Combined filters
python search_videos.py --activity fitness --quality high-quality --min-duration 30
python search_videos.py --resolution 4k --scene "outdoor scene" --limit 20

# Duration filters
python search_videos.py --min-duration 60 --max-duration 300
python search_videos.py --max-duration 30  # Short clips only

# Export results
python search_videos.py --activity fitness -o fitness_videos.json
```

## Common Workflows

### Fitness Content Organization
```bash
# All fitness videos
python search_videos.py --activity fitness -o fitness_all.json

# High-quality workout sessions
python search_videos.py --activity fitness --quality high-quality --min-duration 30 -o workouts.json

# Short fitness clips for social media
python search_videos.py --activity fitness --max-duration 60 -o fitness_clips.json
```

### Quality Audit
```bash
# Find best quality videos
python search_videos.py --quality high-quality --resolution 4k -o premium.json

# Find low-quality videos for review
python search_videos.py --quality low-quality -o review_needed.json

# Quality by resolution
python search_videos.py --resolution 1080p --min-quality-score 80 -o hq_1080p.json
```

### Content Type Searches
```bash
# Product/brand content
python search_videos.py --activity product --quality high-quality -o product_videos.json

# Outdoor b-roll
python search_videos.py --activity outdoor --resolution 4k -o outdoor_broll.json

# Social/family moments
python search_videos.py --activity social --min-duration 10 -o family_videos.json
```

### Platform-Specific Exports
```bash
# Instagram/TikTok vertical shorts
python search_videos.py --max-duration 60 --resolution "1080x1920" -o vertical_shorts.json

# YouTube 4K content
python search_videos.py --resolution 4k --quality high-quality --min-duration 60 -o youtube_4k.json

# Quick clips for stories
python search_videos.py --max-duration 15 -o story_clips.json
```

## Tag Categories

### Activities
- `fitness` - Exercise, yoga, gym, workout content
- `cooking` - Kitchen, food preparation
- `outdoor` - Nature, landscapes, parks, beaches
- `social` - Family, groups, people interactions
- `product` - Product demonstrations, packages

### Quality
- `high-quality` - Quality score 90+/100
- `good-quality` - Quality score 70-89/100
- `low-quality` - Quality score <70/100

### Scenes
- `screenshot` - Digital screen captures
- `document` - Papers, text documents
- `indoor scene` / `outdoor scene`
- `landscape` - Natural scenery
- `portrait` / `group photo`
- `sunset` - Sunset/sunrise scenes
- `product photo` - Product shots

### Technical
- Resolution: `4k`, `1080p`, `720p`
- Duration: `short-clip` (<30s), `medium-length` (30s-5m), `long-video` (>5m)

## Database Maintenance

### Backup
```bash
# Create timestamped backup
cp video_library.db "video_library_backup_$(Get-Date -Format 'yyyyMMdd').db"
```

### Re-analyze
```bash
# Add new videos (skips existing)
python analyze_full_library.py "C:\Users\kjfle\Videos"

# Force re-analysis
rm video_library.db analysis_progress.json
python analyze_full_library.py "C:\Users\kjfle\Videos"
```

### Monitor Progress
```bash
# Check database stats
python search_videos.py --stats

# View progress file
cat analysis_progress.json

# GPU monitoring (separate terminal)
nvidia-smi -l 1
```

## Performance Tips

### Speed Optimization
- Smaller batches = more frequent saves, slightly slower overall
- Larger batches = faster processing, less frequent saves
- Optimal batch size: 100 videos (default)

### GPU Memory
```bash
# Reduce memory usage if needed
python analyze_full_library.py "C:\Users\kjfle\Videos" --max-frames 20
python analyze_full_library.py "C:\Users\kjfle\Videos" --batch-size 50
```

### Resume After Interruption
- Progress auto-saves after each batch
- Press Ctrl+C to stop gracefully
- Run same command to resume: `python analyze_full_library.py "C:\Users\kjfle\Videos"`

## Search Examples by Use Case

### Finding Specific Content
```bash
# Yoga sessions
python search_videos.py --activity fitness --scene "indoor scene" --min-duration 20

# Outdoor adventures
python search_videos.py --activity outdoor --quality high-quality

# Product demos
python search_videos.py --activity product --max-duration 120
```

### Creating Collections
```bash
# Best of 4K
python search_videos.py --resolution 4k --min-quality-score 85 -o best_4k.json

# Social media ready
python search_videos.py --max-duration 60 --quality good-quality -o social_ready.json

# Long-form content
python search_videos.py --min-duration 300 --quality high-quality -o longform.json
```

## Troubleshooting

### Analysis Stuck
```bash
# Check if process is running
tasklist | findstr python

# Force kill if needed
taskkill /F /PID <process_id>

# Restart analysis (will resume)
python analyze_full_library.py "C:\Users\kjfle\Videos"
```

### Database Issues
```bash
# Check database file exists
dir video_library.db

# Check database stats
python search_videos.py --stats

# Restart from scratch if corrupted
rm video_library.db analysis_progress.json
```

### No Results Found
- Check available tags: `python search_videos.py --list-tags`
- Verify database has videos: `python search_videos.py --stats`
- Try broader search criteria
- Check spelling of activity/scene names

## Advanced Python Usage

### Custom Searches
```python
from src.database.video_db import VideoDatabase

with VideoDatabase("video_library.db") as db:
    # Complex multi-criteria search
    results = db.search(
        categories={"activities": ["fitness", "outdoor"]},
        min_duration=30,
        min_quality=80,
        limit=50
    )

    for video in results:
        print(f"{video['file_name']}: {video['duration_formatted']}")
```

### Export File Paths
```python
import json
from src.database.video_db import VideoDatabase

with VideoDatabase("video_library.db") as db:
    results = db.search(activity="fitness")

    # Export paths for video editing software
    paths = [r['file_path'] for r in results]
    with open('fitness_paths.txt', 'w') as f:
        f.write('\n'.join(paths))
```

## File Locations

| Item | Path |
|------|------|
| Database | `video_library.db` |
| Progress tracker | `analysis_progress.json` |
| Search tool | `search_videos.py` |
| Analysis tool | `analyze_full_library.py` |
| Documentation | `VIDEO_LIBRARY_GUIDE.md` |
| This reference | `QUICK_REFERENCE.md` |

## Support

For detailed information, see:
- **Full Guide**: `VIDEO_LIBRARY_GUIDE.md`
- **Getting Started**: `GETTING_STARTED_VIDEO.md`
- **Technical Details**: `VIDEO_ENGINE.md`
- **Summary**: `VIDEO_LIBRARY_SUMMARY.md`
