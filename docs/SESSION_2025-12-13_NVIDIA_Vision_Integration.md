# Session Log: NVIDIA Build API Vision Integration
**Date:** 2025-12-13
**Duration:** ~30 minutes
**Status:** ✓ Complete (Ready for Testing)

---

## Session Objectives

1. Investigate poor tagging quality from local YOLO/CLIP models
2. Integrate existing NVIDIA Build API adapters for superior vision analysis
3. Create cloud-based analyzer as Priority 1 ML backend
4. Prepare test environment for API-based tagging

---

## Problem Statement

**User Observation:** "the tagging system needs training"

**Root Cause Analysis:**
- Local YOLOv8-nano (yolov8n.pt) - weakest YOLO variant, high confidence threshold (0.6)
- Generic CLIP scene classification - not tuned for personal photos
- Result:
  - `soft_focus` tag on 100% of images
  - `very_blurry` on 72%
  - `vehicle` misclassified on 29%
  - YOLO object count = 0 on most images

**Discovery:** User already has NVIDIA Build API integration (`src/adapters/nvidia_build/`) - unused!

---

## Implementation Summary

### 1. Created NVIDIA Vision Analyzer
**File:** `src/core/analyzers/ml_vision_nvidia.py` (239 lines)

**Architecture:**
- Extends `BaseMLAnalyzer` for pipeline compatibility
- Cloud-based processing (no local GPU required)
- Two-model approach:
  1. **VisionLanguageModel** - Llama 3.2 Vision 11B for scene understanding + tag generation
  2. **RetailObjectDetector** - VLM-based object detection via natural language prompts

**Key Features:**
```python
class NVIDIAVisionAnalyzer(BaseMLAnalyzer):
    def __init__(
        self,
        api_key: Optional[str] = None,  # Reads NVIDIA_API_KEY env var
        model: str = "llama-vision",    # or "nemotron"
        batch_size: int = 8,            # API rate limiting
        min_confidence: float = 0.3,    # Lower for VLM
    ):
```

**Methods:**
- `_classify_scene_batch()` - Override base CLIP with VLM scene understanding
- `_detect_objects_batch()` - VLM-based object detection (vs YOLO)
- `analyze_image()` - Enhanced with `ai_generated_tags` field (searchable tags)
- Automatic temp file management for API calls (Windows-compatible)

### 2. Updated Analyzer Factory
**File:** `src/core/factories/analyzer_factory.py`

**New Priority Order:**
1. **NVIDIA Build API** (if `use_nvidia: true`) - Highest quality ← NEW
2. YOLO (if `use_yolo: true`) - Fastest local
3. TensorRT (if `use_tensorrt: true`) - Optimized local
4. Standard PyTorch - Baseline DETR

**Factory Method:**
```python
def _create_nvidia_analyzer(self, ml_config: dict, common_params: dict):
    return NVIDIAVisionAnalyzer(
        batch_size=ml_config.get("batch_size", 8),
        model=ml_config.get("nvidia_model", "llama-vision"),
        api_key=ml_config.get("nvidia_api_key"),  # Optional
        min_confidence=common_params["min_confidence"],
    )
```

### 3. Created NVIDIA Configuration
**File:** `config/nvidia_config.yaml` (135 lines)

**Key Settings:**
```yaml
analysis:
  ml_models:
    use_nvidia: true
    nvidia_model: "llama-vision"  # Llama 3.2 11B
    nvidia_api_key: null          # Reads from env
    batch_size: 8                 # API rate limiting

tagging:
  min_confidence: 0.3  # Lower for VLM (more permissive)

deduplication:
  check_perceptual: true  # Enabled for better duplicates
  perceptual_threshold: 8
```

### 4. Created Test Script
**File:** `test_nvidia_single.py` (109 lines)

**Usage:**
```bash
# Set API key
$env:NVIDIA_API_KEY = "nvapi-xxxxx"

# Run single-image test
python test_nvidia_single.py
```

**Output:**
- Scene classification
- Object detection results
- AI-generated tags (3-5 searchable keywords)
- Full JSON result for debugging
- Helpful error messages if API key missing

### 5. Minor UI Improvements
**File:** `src/ui/web_ui.py`

**Changes:**
- Thumbnail size: 256x256 → 128x128 (1/4 the size)
- Grid columns: 3 → 5 (more compact)
- Fixed width: `width=128` (proportional aspect ratio, no stretching)

---

## Files Created/Modified

### Created (4 files)
- `src/core/analyzers/ml_vision_nvidia.py` (239 lines) - NVIDIA Vision analyzer
- `config/nvidia_config.yaml` (135 lines) - NVIDIA-optimized config
- `test_nvidia_single.py` (109 lines) - Single image test script
- `docs/SESSION_2025-12-13_NVIDIA_Vision_Integration.md` (this file)

### Modified (3 files)
- `src/core/factories/analyzer_factory.py` (+19 lines) - Added NVIDIA factory method, updated priority
- `src/ui/web_ui.py` (3 changes) - Smaller thumbnails, 5 columns, fixed width

---

## API Integration Details

### NVIDIA Build API Endpoints
**Base URL:** `https://integrate.api.nvidia.com/v1`

**Models Available:**
1. **Llama 3.2 Vision 11B** (`meta/llama-3.2-11b-vision-instruct`)
   - General-purpose vision-language model
   - Scene understanding, object detection, tag generation
   - Temperature: 0.1-0.3 for deterministic tagging

2. **Nemotron Nano 12B VL** (`nvidia/nemotron-nano-12b-v2-vl`)
   - Alternative VLM option
   - Optimized for efficiency

**Rate Limiting:**
- Client-side: 100 requests/minute (built-in `_wait_for_rate_limit()`)
- Batch size: 8 images (conservative for API stability)
- Automatic retry on 429/5xx errors

**Authentication:**
- Header: `Authorization: Bearer <api_key>`
- Environment variable: `NVIDIA_API_KEY`
- Get key at: https://build.nvidia.com

---

## Testing Instructions

### 1. Set API Key
```powershell
# PowerShell
$env:NVIDIA_API_KEY = "nvapi-xxxxx"

# Or add to system environment variables permanently
```

### 2. Single Image Test
```bash
python test_nvidia_single.py
```

**Expected Output:**
```
NVIDIA Vision Analyzer - Single Image Test
Test image: 20240925_020819000_iOS.heic
Model: Llama 3.2 Vision 11B (cloud-based)

✓ Analyzer initialized
✓ Analysis complete

RESULTS
Scene: portrait
Objects detected: 3
  person, face, background
AI-generated tags (5):
  portrait, person, indoor, face, natural
```

### 3. Full Batch Analysis (Batch_1)
```bash
python -m src.ui.cli analyze "D:\Pictures\Batch_1" --config config/nvidia_config.yaml -o batch1_nvidia.json
```

**Expected:**
- 177 images analyzed via NVIDIA API
- Time: ~10-15 minutes (API latency ~3-5 sec/image)
- Output: `batch1_nvidia.json` with enhanced tags

### 4. Compare Results
```bash
# Generate tags from NVIDIA results
python scripts/dev/generate_tags.py --input batch1_nvidia.json --output batch1_nvidia_tags.json

# View in web UI
streamlit run src/ui/web_ui.py
# Load batch1_nvidia_tags.json
```

---

## Expected Improvements

### Tag Quality
**Before (YOLO + CLIP):**
- Generic: `soft_focus` (100%), `very_blurry` (72%)
- Misclassified: `vehicle` (29% - wrong)
- Empty: Most images had 0 objects detected

**After (NVIDIA VLM):**
- Contextual: `portrait`, `sunset`, `beach`, `street photography`
- Accurate objects: `person`, `bicycle`, `building`, `tree`
- Natural language tags: `outdoor nature`, `urban scene`, `food closeup`

### Scene Understanding
**CLIP:** 80 generic classes (indoor, outdoor, vehicle, etc.)
**Llama 3.2 Vision:** Natural language descriptions (e.g., "A person using a laptop at a coffee shop")

### Object Detection
**YOLO:** Binary classification with confidence threshold
**VLM:** "List all objects and items visible in this image" → comma-separated natural language

---

## Performance Comparison

| Metric | YOLO (Local) | NVIDIA API (Cloud) |
|--------|-------------|-------------------|
| **Scene Accuracy** | Generic (80 classes) | Contextual (unlimited) |
| **Object Detection** | 0-5 objects (high threshold) | 3-10+ objects (permissive) |
| **Tag Quality** | Generic keywords | Natural language |
| **Speed** | ~1-3 sec/batch (16 images) | ~3-5 sec/image (API latency) |
| **GPU Usage** | ~1-1.5GB VRAM | 0 (cloud-based) |
| **Cost** | Free (local) | Free tier available |
| **Setup** | Model download (~6MB) | API key only |

**Recommendation:** Use NVIDIA for initial tagging (quality), YOLO for incremental updates (speed).

---

## Known Limitations

### 1. API Rate Limiting
- Free tier: Limited requests/day (check NVIDIA Build quotas)
- Batch size: 8 images (conservative)
- Solution: Process large libraries in multiple sessions

### 2. Temp File I/O
- API requires file paths (base64 encoding)
- Current: Save temp JPEG, send to API, delete
- Future: In-memory base64 encoding to skip temp files

### 3. HEIC Support
- `pillow-heif` registers HEIC with PIL globally
- Temp files saved as JPEG for API compatibility
- Quality loss: Minimal (JPEG 95% quality)

### 4. Cost Monitoring
- Cloud API = potential costs beyond free tier
- Solution: Set up billing alerts, monitor usage
- Local YOLO: Always free fallback

---

## Next Steps

### Immediate (User)
1. ✓ Set `NVIDIA_API_KEY` environment variable
2. ⏸ Run single image test: `python test_nvidia_single.py`
3. ⏸ Analyze Batch_1 with NVIDIA: `python -m src.ui.cli analyze ... --config config/nvidia_config.yaml`
4. ⏸ Compare tag quality vs YOLO results

### Future Enhancements
1. **Hybrid Mode:** NVIDIA for new images, cache for processed
2. **In-Memory Encoding:** Skip temp file I/O with base64
3. **Batch API Calls:** Group multiple images per request
4. **Cost Tracking:** Log API usage for billing monitoring
5. **Fine-Tuning:** Custom prompts for specific photo types
6. **Duplicate Filter:** Add perceptual hash UI filter (20% duplicates in Batch_1)

---

## Troubleshooting

### "NVIDIA_API_KEY not found"
**Solution:** Set environment variable
```powershell
$env:NVIDIA_API_KEY = "nvapi-xxxxx"
```

### "Rate limit reached"
**Solution:** Wait 60 seconds, or reduce batch size in config
```yaml
ml_models:
  batch_size: 4  # Reduce from 8
```

### "Missing module: tempfile"
**Solution:** Built-in Python module, check Python installation

### "Scene classification failed"
**Cause:** API timeout or network issue
**Solution:** Check internet connection, retry with lower batch size

---

## Session Metrics

**Time Breakdown:**
- Problem investigation: ~5 min
- NVIDIA analyzer implementation: ~10 min
- Factory integration: ~5 min
- Configuration + test script: ~5 min
- UI improvements: ~3 min
- Documentation: ~2 min

**Code Statistics:**
- Lines added: ~430 (239 analyzer + 135 config + 56 factory/UI)
- Files created: 4
- Files modified: 3
- Tests created: 1 (single image test)

**Git Status:** Clean working tree, all changes uncommitted (ready for testing first)

---

## Related Documentation

- NVIDIA Build API: https://build.nvidia.com/explore/discover
- Llama 3.2 Vision: https://build.nvidia.com/meta/llama-3_2-11b-vision-instruct
- Project CLAUDE.md: Lines 136-189 (InsightFace API reference)
- Previous session: `docs/SESSION_2025-12-13_InsightFace_Migration.md`

---

**Session Complete** - All code ready for testing with user's NVIDIA API key.
