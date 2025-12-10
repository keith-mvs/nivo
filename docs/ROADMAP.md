# Video Engine - Development Roadmap

## Current Status: MVP Complete ✓

**Phase 1: Core Infrastructure** (COMPLETE)
- Video analysis pipeline with GPU acceleration
- SQLite database with multi-dimensional search
- Batch processing with resume capability
- Content tagging across 5 categories
- CLI search interface

**Library Status**: 1,817 videos → Processing in progress

---

## Phase 2: NVIDIA Build API Integration (NEXT)

### Objectives
Enhance video analysis with NVIDIA's production-ready AI models for superior accuracy and specialized capabilities.

### Implementation Tasks

#### 2.1 Retail Object Detection
**Model**: `nvidia/retail-object-detection`
**Purpose**: Identify products, packages, brands in product videos

```python
# src/analyzers/nvidia_retail.py
class NVIDIARetailAnalyzer:
    """Analyze videos for retail/product content using NVIDIA Build API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://ai.api.nvidia.com/v1/cv/nvidia/retail-object-detection"

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect retail objects in video frame.

        Returns:
            {
                "objects": [
                    {"class": "bottle", "confidence": 0.95, "bbox": [x, y, w, h]},
                    {"class": "package", "confidence": 0.88, "bbox": [x, y, w, h]}
                ],
                "product_count": 2,
                "dominant_products": ["bottle", "package"]
            }
        """
        # Convert frame to base64
        # POST to NVIDIA API
        # Parse retail detection results
        # Return structured data
```

**Integration Points**:
- Add to `VideoAnalyzer.analyze()` pipeline
- Create new tag category: `retail_objects`
- Update search filters for product content
- Store product metadata in database

**Benefits**:
- More accurate product identification
- Brand/package recognition
- Product placement analysis
- Commercial content classification

#### 2.2 Vision-Language Model
**Model**: `nvidia/nemotron-nano-12b-v2-vl`
**Purpose**: Natural language descriptions of video content

```python
# src/analyzers/nvidia_vision_language.py
class NVIDIAVisionLanguageAnalyzer:
    """Generate natural language descriptions of video frames."""

    def generate_description(self, frame: np.ndarray, prompt: str = None) -> str:
        """
        Generate text description of video frame.

        Args:
            frame: Video frame
            prompt: Optional prompt (e.g., "Describe the activity in this video")

        Returns:
            "A person performing a yoga pose on a mat in a bright indoor space"
        """
        # Default prompt if not provided
        if not prompt:
            prompt = "Describe what is happening in this video frame in one sentence."

        # Send frame + prompt to NVIDIA API
        # Return natural language description
```

**Use Cases**:
- Searchable text descriptions
- Accessibility (screen readers)
- Context-aware search
- Automatic video titling
- Content summarization

**Database Schema Update**:
```sql
ALTER TABLE videos ADD COLUMN description TEXT;
ALTER TABLE videos ADD COLUMN description_vector BLOB;  -- For semantic search
```

#### 2.3 API Management

```python
# src/utils/nvidia_api.py
class NVIDIAAPIManager:
    """Manage NVIDIA Build API requests with rate limiting and caching."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiter = RateLimiter(max_requests=100, per_seconds=60)
        self.cache = APICache(ttl_hours=24)

    async def request(self, endpoint: str, payload: Dict) -> Dict:
        """
        Make API request with rate limiting and error handling.

        Features:
        - Automatic retry with exponential backoff
        - Response caching
        - Rate limit compliance
        - Error logging
        """
```

**Configuration**:
```yaml
# config/default_config.yaml
nvidia_build:
  enabled: true
  api_key: ${NVIDIA_API_KEY}  # From environment variable
  models:
    retail_detection:
      endpoint: "nvidia/retail-object-detection"
      enabled: true
      confidence_threshold: 0.7
    vision_language:
      endpoint: "nvidia/nemotron-nano-12b-v2-vl"
      enabled: true
      max_tokens: 100
  rate_limit:
    max_requests: 100
    per_seconds: 60
  cache:
    enabled: true
    ttl_hours: 24
```

**Timeline**: 2-3 weeks
**Dependencies**: NVIDIA API key, async HTTP client (httpx)

---

## Phase 3: Performance Optimization

### 3.1 TensorRT Model Optimization
**Goal**: 3-5x faster inference for real-time 4K video processing

**Implementation**:
```python
# src/optimizers/tensorrt_optimizer.py
class TensorRTOptimizer:
    """Convert PyTorch models to TensorRT for faster inference."""

    def optimize_model(self, model_name: str, precision: str = "fp16"):
        """
        Convert CLIP/DETR to TensorRT engine.

        Args:
            model_name: "clip" or "detr"
            precision: "fp32", "fp16", or "int8"

        Returns:
            Optimized TensorRT engine
        """
        # Export to ONNX
        # Convert ONNX to TensorRT
        # Benchmark performance
        # Save optimized engine
```

**Expected Performance**:
- Current: ~2-4 seconds/frame (CLIP + DETR)
- TensorRT FP16: ~0.5-1 second/frame
- TensorRT INT8: ~0.3-0.7 seconds/frame

**Benefits**:
- Real-time 4K video analysis
- Lower GPU memory usage
- Process 1,817 videos in ~2-3 hours instead of 9-12

**Timeline**: 1-2 weeks

### 3.2 RAPIDS GPU Acceleration
**Goal**: GPU-accelerated data processing and filtering

```python
# src/utils/rapids_processor.py
import cudf  # GPU DataFrame
import cuml  # GPU ML algorithms

class RAPIDSProcessor:
    """GPU-accelerated video metadata processing."""

    def parallel_search(self, filters: Dict) -> cudf.DataFrame:
        """
        Search database using GPU DataFrames.

        100x faster than pandas for large datasets
        """
        # Load database to GPU memory (cudf)
        # Apply filters with GPU acceleration
        # Return results as GPU DataFrame
```

**Use Cases**:
- Similarity search across thousands of videos
- Real-time filtering and aggregation
- Duplicate detection
- Clustering videos by content

**Timeline**: 1 week

---

## Phase 4: Advanced Features

### 4.1 Web UI
**Framework**: FastAPI + React or Streamlit
**Features**:
- Video library browser with thumbnails
- Interactive search with real-time filtering
- Video player with metadata overlay
- Drag-and-drop video upload
- Export playlists

**Architecture**:
```
frontend/
├── src/
│   ├── components/
│   │   ├── VideoGrid.tsx
│   │   ├── SearchBar.tsx
│   │   ├── VideoPlayer.tsx
│   │   └── FilterPanel.tsx
│   └── App.tsx
backend/
├── api/
│   ├── videos.py       # Video CRUD
│   ├── search.py       # Search endpoints
│   └── analysis.py     # Trigger analysis
└── main.py
```

**Timeline**: 2-3 weeks

### 4.2 Video Scene Segmentation
**Goal**: Automatically split videos into scenes

```python
# src/analyzers/scene_segmenter.py
class SceneSegmenter:
    """Detect scene boundaries and extract clips."""

    def segment_video(self, video_path: str) -> List[Dict]:
        """
        Split video into semantic scenes.

        Returns:
            [
                {"start": 0.0, "end": 5.2, "description": "yoga pose"},
                {"start": 5.2, "end": 12.8, "description": "transition"},
                {"start": 12.8, "end": 20.5, "description": "different pose"}
            ]
        """
```

**Use Cases**:
- Automatic highlight extraction
- Scene-level tagging
- Clip generation for social media
- Video summarization

**Timeline**: 1-2 weeks

### 4.3 Audio Analysis
**Models**: Whisper (speech-to-text), Audio classification

```python
# src/analyzers/audio_analyzer.py
class AudioAnalyzer:
    """Analyze audio tracks in videos."""

    def analyze(self, video_path: str) -> Dict:
        """
        Extract audio features.

        Returns:
            {
                "has_speech": True,
                "transcript": "...",
                "music_detected": False,
                "dominant_sounds": ["voice", "ambient"],
                "audio_quality": 0.85
            }
        """
```

**Benefits**:
- Search by spoken content
- Music/silence detection
- Audio quality assessment
- Accessibility (captions)

**Timeline**: 2 weeks

### 4.4 Duplicate & Similar Video Detection
**Approach**: Perceptual hashing + embedding similarity

```python
# src/analyzers/similarity.py
class VideoSimilarity:
    """Find duplicate and similar videos."""

    def compute_hash(self, video_path: str) -> str:
        """Generate perceptual hash for duplicate detection."""

    def find_similar(self, video_id: int, threshold: float = 0.8) -> List[Dict]:
        """
        Find videos similar to given video.

        Uses CLIP embeddings for semantic similarity
        """
```

**Timeline**: 1 week

---

## Phase 5: Production Deployment

### 5.1 Distributed Processing
**Framework**: Celery + Redis or Ray

```python
# tasks/video_analysis.py
from celery import Celery

app = Celery('video_analysis', broker='redis://localhost:6379')

@app.task
def analyze_video_task(video_path: str):
    """Distributed video analysis task."""
    # Analyze video
    # Store results
    # Update progress
```

**Benefits**:
- Parallel processing across multiple GPUs
- Horizontal scaling
- Process thousands of videos simultaneously

**Timeline**: 1-2 weeks

### 5.2 API Server
**Framework**: FastAPI

```python
# api/main.py
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/videos/analyze")
async def analyze_video(file: UploadFile):
    """Upload and analyze video."""

@app.get("/videos/search")
async def search_videos(
    activity: str = None,
    quality: str = None,
    min_duration: float = None
):
    """Search video library."""
```

**Timeline**: 1 week

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 2: NVIDIA Integration | 2-3 weeks | Retail detection, vision-language descriptions |
| Phase 3: Optimization | 2-3 weeks | TensorRT (3-5x speedup), RAPIDS acceleration |
| Phase 4: Advanced Features | 6-8 weeks | Web UI, scene segmentation, audio analysis, similarity search |
| Phase 5: Production | 2-3 weeks | Distributed processing, API server |

**Total**: ~12-17 weeks for complete system

---

## Priority Recommendations

### High Priority (Next 4 weeks)
1. **NVIDIA Build API Integration** - Significant accuracy improvement
2. **TensorRT Optimization** - 3-5x performance boost
3. **Web UI (Basic)** - Much better user experience than CLI

### Medium Priority (4-8 weeks)
4. **Scene Segmentation** - Enables new use cases
5. **Audio Analysis** - Searchable transcripts
6. **RAPIDS Acceleration** - Fast search at scale

### Low Priority (8+ weeks)
7. **Distributed Processing** - Only needed for massive scale
8. **Advanced Web UI Features** - Nice to have
9. **API Server** - Only if external integrations needed

---

## Resource Requirements

### Development
- **GPU**: RTX 2080 Ti (current) - sufficient for all phases
- **RAM**: 16GB+ recommended for RAPIDS
- **Storage**: 50GB+ for models and cache

### Production (If Deployed)
- **GPU Server**: Single RTX 3090 or A4000 (handles 100+ videos/hour)
- **Database**: PostgreSQL or continue with SQLite (<10K videos)
- **Cache**: Redis (for distributed processing)
- **Storage**: Network storage for video library

---

## Risk Mitigation

### Technical Risks
1. **NVIDIA API Rate Limits**: Implement caching, batch requests
2. **TensorRT Compatibility**: Test on target GPU before full migration
3. **Database Performance**: Add indexes, consider PostgreSQL if >10K videos

### Timeline Risks
1. **Phase 2 API Integration**: May encounter API changes → Allow 1 week buffer
2. **Phase 4 Web UI**: Frontend complexity → Start with Streamlit MVP

---

## Success Metrics

### Phase 2 Success Criteria
- Retail detection accuracy >85% on product videos
- Natural language descriptions >90% relevant
- API response time <2 seconds per frame

### Phase 3 Success Criteria
- TensorRT inference 3x faster than PyTorch
- GPU memory usage reduced by 40%
- Full library analysis <4 hours

### Phase 4 Success Criteria
- Web UI loads 1000+ videos smoothly
- Search response time <500ms
- Scene segmentation accuracy >80%

---

## Next Steps

1. **Immediate** (This Week):
   - Monitor current batch analysis completion
   - Test search functionality on processed videos
   - Verify database performance at scale

2. **Week 1-2** (Phase 2 Start):
   - Obtain NVIDIA API key
   - Implement retail detection analyzer
   - Test on sample product videos

3. **Week 3-4** (Phase 2 Continued):
   - Implement vision-language integration
   - Add description search to database
   - Create comprehensive test suite

4. **Month 2** (Phase 3):
   - TensorRT optimization
   - Benchmark performance improvements
   - RAPIDS integration for search

---

**Status**: Ready to begin Phase 2 upon Phase 1 completion
**Last Updated**: 2025-11-30
