# Rollback Instructions

## Pre-Tag-Embed Backup (2025-12-01)

### Backup Points

1. **Git Tag**: `pre-tag-embed-backup`
   - Commit: 854058a (Remove video functionality)
   - Date: 2025-12-01

2. **Analysis File Backup**: `image_engine_analysis_with_tags.backup.json`

### To Rollback

#### Rollback Code Changes:
```bash
git checkout pre-tag-embed-backup
# Or to create new branch from backup:
git checkout -b rollback-branch pre-tag-embed-backup
```

#### Rollback Analysis Data:
```bash
# Restore analysis JSON
cp image_engine_analysis_with_tags.backup.json image_engine_analysis_with_tags.json
```

#### Rollback EXIF/IPTC Changes:
If tag embedding modifies image files, restore from your original image backup.

**IMPORTANT**: Tag embedding writes directly to image files. Ensure you have
backups of your original images before proceeding.

### Test Images Directory
For safe testing, use a copy of a small subset of images:
```bash
# Create test directory with copies
mkdir test_images
cp /path/to/originals/*.jpg test_images/
```

### Verification After Rollback
```bash
# Check git status
git log -1

# Verify analysis file
ls -lh image_engine_analysis*.json
```
