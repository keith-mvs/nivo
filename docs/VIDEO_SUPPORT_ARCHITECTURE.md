# Video Support Architecture for Image Engine

**Version**: 1.0
**Date**: 2025-11-29
**Status**: Design Phase - Ready for Implementation

---

## Executive Summary

Expanding Image Engine to support video analysis leverages existing GPU infrastructure and NVIDIA's video AI models to provide comprehensive media management. This document outlines the architecture, models, and implementation strategy.

### Key Capabilities

1. **Video Metadata Extraction**: Duration, resolution, codecs, fps, bitrate
2. **Temporal Scene Analysis**: Scene changes, key moments, activity detection
3. **Action Recognition**: Human activities, sports, gestures
4. **Object Tracking**: Track people, vehicles, animals across frames
5. **Highlight Detection**: Auto-identify best moments (peaks, faces, motion)
6. **Video Summarization**: Generate thumbnails, preview clips, summaries
7. **Audio Analysis**: Speech-to-text, music detection, sound classification

---

## 1. NVIDIA Video Models & Technologies

### 1.1 DeepStream SDK

**What**: NVIDIA's video analytics framework (GPU-accelerated)
**Use**: Real-time video processing, multi-stream support
**Models**: 3D Action Recognition, object tracking, multi-object tracking

**Key Features**:
- Hardware-accelerated decode (NVDEC)
- Temporal batching (process multiple frames together)
- Multi-ROI support (track multiple regions)
- Optimized for RTX GPUs

**Integration**: Can be wrapped in Python via GStreamer bindings

### 1.2 TAO Action Recognition Models

**Available Models**:
1. **ActionRecognitionNet**: Pre-trained on Kinetics-400 dataset
   - 400 action classes (walking, running, cooking, sports, etc.)
   - RGB + Flow input modalities
   - 3D ConvNet architecture

2. **Temporal models**: C3D, I3D, SlowFast
   - C3D: 3D convolutional networks
   - I3D: Inflated 3D conv (spatial + temporal)
   - SlowFast: Dual-pathway (slow for spatial, fast for temporal)

3. **2D+T models**: TSM, TSN
   - Temporal Segment Networks
   - Temporal Shift Modules
   - Lightweight, efficient

**Performance**: Real-time on RTX 2080 Ti (~30 FPS for action recognition)

### 1.3 Vision Language Models (VLMs) for Video

**NVIDIA's Video VLMs** (TAO 5.5):
- **LITA** (Localized Interpretable Temporal Attention)
  - Understands "when" and "where" events occur
  - Temporal reasoning over long videos
  - Can answer: "When did the person start running?"

- **Video Search & Summarization (VSS)**
  - Generate text descriptions of video content
  - Track objects across time
  - Create searchable video databases

**Use Cases**:
- Natural language video search: "Find videos with dogs playing"
- Auto-captioning: Generate descriptions
- Question answering: "How many cars appeared?"

### 1.4 Video Transformers

**TimeSformer**: Space-time attention transformers
**VideoMAE**: Masked autoencoder for video representation
**ViViT**: Video Vision Transformer

**Benefits**:
- Better long-range temporal modeling
- Higher accuracy than 3D CNNs
- Can be fine-tuned on custom data

---

## 2. Architecture Design

### 2.1 High-Level Pipeline

```
Video File
    │
    ├─> Metadata Extraction (ffprobe)
    │   └─> Duration, resolution, fps, codec, bitrate
    │
    ├─> Frame Extraction Strategy
    │   ├─> Key frames (I-frames)
    │   ├─> Uniform sampling (every N seconds)
    │   ├─> Smart sampling (scene changes)
    │   └─> Motion-based sampling
    │
    ├─> Frame Analysis (GPU Batched)
    │   ├─> Scene Classification (CLIP/NVCLIP)
    │   ├─> Object Detection (YOLOv4/DETR)
    │   ├─> Quality Assessment (per-frame)
    │   └─> Face Detection (optional)
    │
    ├─> Temporal Analysis (GPU Batched)
    │   ├─> Action Recognition (ActionRecognitionNet)
    │   ├─> Scene Change Detection
    │   ├─> Object Tracking (Multi-Object Tracker)
    │   └─> Motion Analysis
    │
    ├─> Audio Analysis (CPU/GPU)
    │   ├─> Speech-to-Text (Whisper)
    │   ├─> Music Detection
    │   └─> Sound Classification
    │
    └─> Aggregation & Summarization
        ├─> Best frame selection (thumbnail)
        ├─> Highlight detection
        ├─> Activity timeline
        ├─> Tags & metadata
        └─> Video summary JSON
```

### 2.2 Frame Extraction Strategies

#### Strategy 1: Uniform Sampling (Simple)
```python
# Extract 1 frame per second
fps = 1  # 1 FPS
total_frames = video_duration * fps
frames = extract_frames_uniform(video_path, fps)
```

**Pros**: Predictable, simple
**Cons**: May miss important moments
**Use**: Quick preview, low-detail analysis

#### Strategy 2: Key Frame Extraction (I-frames)
```python
# Extract only I-frames (keyframes from video codec)
keyframes = extract_keyframes(video_path)
```

**Pros**: Fast, no re-encoding
**Cons**: Irregular intervals
**Use**: Scene change detection, thumbnails

#### Strategy 3: Adaptive Sampling (Smart)
```python
# Extract frames based on scene changes
scene_changes = detect_scene_changes(video_path)
frames = extract_frames_at_timestamps(video_path, scene_changes)
```

**Pros**: Captures important moments
**Cons**: More computation
**Use**: Highlight detection, summarization

#### Strategy 4: Motion-Based Sampling
```python
# Extract frames with significant motion
motion_frames = extract_high_motion_frames(video_path, threshold=0.7)
```

**Pros**: Focus on action
**Cons**: Miss static important scenes
**Use**: Sports, action videos

### 2.3 Batch Processing Optimization

**GPU Memory Management**:
```python
# For RTX 2080 Ti (11GB VRAM)
batch_size_frames = 32  # Process 32 frames at once
batch_size_clips = 8    # Process 8 video clips (4 frames each)

# Temporal batching for action recognition
clip_length = 16  # 16 frames per clip
clip_stride = 8   # Overlapping clips
```

**Multi-Stream Processing**:
```python
# Process multiple videos in parallel (if sufficient VRAM)
num_concurrent_videos = 2  # 2 videos simultaneously
```

---

## 3. Component Design

### 3.1 VideoAnalyzer Class

```python
class VideoAnalyzer:
    """Main video analysis engine."""

    def __init__(self, use_gpu=True, batch_size=32, extract_fps=1):
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.batch_size = batch_size
        self.extract_fps = extract_fps

        # Initialize analyzers
        self.metadata_extractor = VideoMetadataExtractor()
        self.frame_analyzer = FrameAnalyzer(use_gpu=use_gpu)
        self.action_recognizer = ActionRecognizer(use_gpu=use_gpu)
        self.audio_analyzer = AudioAnalyzer()

    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Full video analysis pipeline."""

        # 1. Extract metadata
        metadata = self.metadata_extractor.extract(video_path)

        # 2. Extract frames (adaptive strategy)
        frames = self.extract_frames_smart(video_path, metadata)

        # 3. Analyze frames (batched)
        frame_results = self.frame_analyzer.analyze_batch(frames)

        # 4. Temporal analysis (action recognition)
        action_results = self.action_recognizer.analyze_video(video_path)

        # 5. Audio analysis
        audio_results = self.audio_analyzer.analyze(video_path)

        # 6. Aggregate results
        summary = self.aggregate_results(
            metadata, frame_results, action_results, audio_results
        )

        return summary
```

### 3.2 Frame Extraction Module

```python
import cv2
import numpy as np

class FrameExtractor:
    """Extract frames from video efficiently."""

    def __init__(self, strategy="adaptive", fps=1):
        self.strategy = strategy
        self.fps = fps

    def extract(self, video_path: str) -> List[np.ndarray]:
        """Extract frames based on strategy."""

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.strategy == "uniform":
            return self._extract_uniform(cap, video_fps, total_frames)
        elif self.strategy == "keyframes":
            return self._extract_keyframes(video_path)
        elif self.strategy == "adaptive":
            return self._extract_adaptive(cap, video_fps, total_frames)

    def _extract_uniform(self, cap, video_fps, total_frames):
        """Extract frames at uniform intervals."""
        frames = []
        frame_interval = int(video_fps / self.fps)

        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        return frames
```

### 3.3 Action Recognition Module

```python
class ActionRecognizer:
    """Recognize actions in videos using NVIDIA TAO models."""

    def __init__(self, model_path="nvidia/tao/actionrecognitionnet", use_gpu=True):
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.model = self._load_model(model_path)
        self.clip_length = 16  # frames per clip

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze video for actions."""

        # Extract clips (temporal segments)
        clips = self._extract_clips(video_path, self.clip_length)

        # Batch process clips
        actions = []
        for clip_batch in batch(clips, batch_size=8):
            results = self._recognize_batch(clip_batch)
            actions.extend(results)

        # Aggregate actions (temporal smoothing)
        action_timeline = self._aggregate_actions(actions)

        return {
            "actions": action_timeline,
            "primary_action": self._get_primary_action(action_timeline),
            "action_count": len(action_timeline),
        }

    def _recognize_batch(self, clips: List[torch.Tensor]) -> List[Dict]:
        """Batch action recognition."""

        # Stack clips into batch
        clip_tensor = torch.stack(clips).to(self.device)  # [B, T, C, H, W]

        with torch.no_grad():
            outputs = self.model(clip_tensor)
            probs = torch.softmax(outputs, dim=1)

        # Parse results
        results = []
        for prob in probs:
            top_k = torch.topk(prob, k=3)
            actions = [{
                "action": self.class_labels[idx],
                "confidence": float(conf)
            } for conf, idx in zip(top_k.values, top_k.indices)]
            results.append(actions)

        return results
```

### 3.4 Video Metadata Extractor

```python
import subprocess
import json

class VideoMetadataExtractor:
    """Extract video metadata using ffprobe."""

    def extract(self, video_path: str) -> Dict[str, Any]:
        """Extract comprehensive video metadata."""

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)

        # Parse metadata
        format_data = data.get("format", {})
        video_stream = self._get_video_stream(data.get("streams", []))
        audio_stream = self._get_audio_stream(data.get("streams", []))

        return {
            "file_path": video_path,
            "file_size_mb": float(format_data.get("size", 0)) / 1_000_000,
            "duration_seconds": float(format_data.get("duration", 0)),
            "bitrate_kbps": int(format_data.get("bit_rate", 0)) / 1000,
            "container_format": format_data.get("format_name"),

            # Video stream
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "fps": eval(video_stream.get("r_frame_rate", "0/1")),
            "codec": video_stream.get("codec_name"),
            "video_bitrate_kbps": int(video_stream.get("bit_rate", 0)) / 1000,

            # Audio stream
            "has_audio": audio_stream is not None,
            "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
            "sample_rate": audio_stream.get("sample_rate") if audio_stream else None,
        }

    def _get_video_stream(self, streams):
        for stream in streams:
            if stream.get("codec_type") == "video":
                return stream
        return {}
```

---

## 4. Storage & Database Schema

### 4.1 Video Metadata Schema

```json
{
  "video_id": "uuid",
  "file_path": "/path/to/video.mp4",
  "file_size_mb": 150.5,
  "duration_seconds": 120.5,
  "fps": 30,
  "resolution": "1920x1080",
  "codec": "h264",
  "bitrate_kbps": 5000,

  "analysis": {
    "analyzed_at": "2025-11-29T12:00:00",
    "analysis_version": "1.0",
    "frames_analyzed": 120,
    "strategy": "adaptive",

    "primary_action": "playing basketball",
    "actions": [
      {"timestamp": 0.0, "action": "dribbling", "confidence": 0.9},
      {"timestamp": 5.2, "action": "shooting", "confidence": 0.85}
    ],

    "objects_detected": ["person", "basketball", "hoop"],
    "scene_types": ["outdoor", "sports court"],

    "highlights": [
      {"timestamp": 15.5, "description": "Best moment", "score": 0.95}
    ],

    "tags": ["sports", "basketball", "outdoor", "action"],

    "thumbnail_frame": 42,  # Frame number for best thumbnail
    "best_frames": [10, 42, 87, 105],

    "has_people": true,
    "people_count_avg": 3,
    "motion_intensity": "high"
  },

  "audio": {
    "has_speech": true,
    "has_music": false,
    "transcription": "Let's play! Nice shot!",
    "dominant_sound": "crowd noise"
  }
}
```

---

## 5. Performance Considerations

### 5.1 Processing Time Estimates (RTX 2080 Ti)

**Video: 60 seconds, 1080p, 30 FPS**

| Task | Time | Notes |
|------|------|-------|
| Metadata extraction | <1 sec | CPU, ffprobe |
| Frame extraction (1 FPS) | ~2-3 sec | 60 frames |
| Frame analysis (batched) | ~5-10 sec | Scene + objects, batch 32 |
| Action recognition | ~3-5 sec | 8 clips, batch 8 |
| Audio transcription | ~3-5 sec | Whisper small |
| **Total** | **15-25 sec** | ~4x video playback speed |

**Optimized (TensorRT + larger batches)**: ~8-12 seconds (~2x playback speed)

### 5.2 Storage Requirements

**Per video** (60 sec, 1080p):
- Original video: ~50-150 MB (depends on compression)
- Metadata JSON: ~10-50 KB
- Extracted frames (thumbnails): ~1-5 MB (if saved)
- **Total**: +1-5 MB overhead per video

**For 1,000 videos** (1 hour each):
- Metadata: ~50 MB
- Thumbnails: ~5 GB
- **Total overhead**: ~5-6 GB

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Tasks**:
1. ✅ Research NVIDIA video models
2. ✅ Design architecture
3. Implement `VideoMetadataExtractor`
4. Implement `FrameExtractor` (uniform + keyframe strategies)
5. Test frame extraction on sample videos

**Deliverables**:
- Working metadata extraction
- Frame extraction module
- Unit tests

### Phase 2: Frame Analysis (Week 2-3)

**Tasks**:
1. Integrate existing `MLVisionAnalyzer` for frames
2. Implement batched frame processing
3. Add scene change detection
4. Implement thumbnail selection algorithm

**Deliverables**:
- Frame-based video analysis working
- Best frame selection
- Tests on various video types

### Phase 3: Temporal Analysis (Week 3-4)

**Tasks**:
1. Install NVIDIA TAO Toolkit
2. Download ActionRecognitionNet model
3. Implement `ActionRecognizer` class
4. Add temporal smoothing/aggregation
5. Test on action videos

**Deliverables**:
- Working action recognition
- Temporal analysis pipeline
- Action timeline generation

### Phase 4: Audio Analysis (Week 4-5)

**Tasks**:
1. Integrate Whisper for speech-to-text
2. Add music/speech classification
3. Implement sound event detection (optional)
4. Test on various audio types

**Deliverables**:
- Audio transcription working
- Music detection
- Integrated audio analysis

### Phase 5: Integration & Optimization (Week 5-6)

**Tasks**:
1. Integrate all components into unified pipeline
2. Add TensorRT optimization for action models
3. Implement highlight detection
4. Add video summarization
5. Create CLI commands for video analysis

**Deliverables**:
- Full video pipeline working
- CLI: `python -m src.cli analyze-video ./video.mp4`
- Performance optimizations applied

### Phase 6: Advanced Features (Week 6-8)

**Tasks**:
1. Add object tracking (MOT)
2. Implement scene-based video splitting
3. Add VLM integration for video Q&A
4. Create video search functionality
5. Build video gallery/viewer

**Deliverables**:
- Advanced video features
- Searchable video database
- Video management UI (optional)

---

## 7. API Design

### 7.1 CLI Commands

```bash
# Analyze single video
python -m src.cli analyze-video ./video.mp4 -o analysis.json

# Analyze video directory
python -m src.cli analyze-video ./videos/ --recursive -o video_results.json

# Extract highlights
python -m src.cli extract-highlights ./video.mp4 -o highlights/ --duration 10

# Generate thumbnail
python -m src.cli generate-thumbnail ./video.mp4 -o thumbnail.jpg

# Search videos
python -m src.cli search-videos "basketball game" --database ./video_db.json
```

### 7.2 Python API

```python
from src.video import VideoAnalyzer

# Initialize analyzer
analyzer = VideoAnalyzer(use_gpu=True, extract_fps=1)

# Analyze video
results = analyzer.analyze("./video.mp4")

# Get highlights
highlights = analyzer.extract_highlights("./video.mp4", min_score=0.8)

# Search videos
matches = analyzer.search("playing basketball", video_database)
```

---

## 8. Dependencies

### Core Dependencies

```txt
# Video processing
opencv-python>=4.8.0
ffmpeg-python>=0.2.0

# NVIDIA DeepStream (optional - advanced)
# Install separately via NVIDIA NGC

# Audio analysis
openai-whisper>=20231117  # Speech-to-text
librosa>=0.10.0           # Audio processing

# ML models
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0

# Existing dependencies
# (same as image engine)
```

### Optional Dependencies

```txt
# Video transformers
pytorchvideo>=0.1.5       # Meta's video library
timm>=0.9.0               # Vision models

# Object tracking
motpy>=0.0.10             # Multi-object tracking
```

---

## 9. Testing Strategy

### 9.1 Test Videos

Create test video collection:
- Short clip (10 sec)
- Medium video (60 sec)
- Long video (5 min)
- Different resolutions (480p, 720p, 1080p, 4K)
- Different codecs (H.264, H.265, VP9)
- With/without audio
- Action video (sports)
- Static video (lecture)
- Multiple people
- Animals

### 9.2 Unit Tests

```python
def test_video_metadata_extraction():
    extractor = VideoMetadataExtractor()
    metadata = extractor.extract("test_video.mp4")
    assert metadata["fps"] > 0
    assert metadata["duration_seconds"] > 0

def test_frame_extraction_uniform():
    extractor = FrameExtractor(strategy="uniform", fps=1)
    frames = extractor.extract("test_video.mp4")
    # 10 sec video -> ~10 frames at 1 FPS
    assert len(frames) >= 9 and len(frames) <= 11

def test_action_recognition():
    recognizer = ActionRecognizer()
    results = recognizer.analyze_video("basketball.mp4")
    # Should detect basketball-related actions
    actions = [a["action"] for a in results["actions"]]
    assert any("basketball" in a.lower() or "dribbling" in a.lower() for a in actions)
```

---

## 10. Known Challenges & Solutions

### Challenge 1: Large Video Files

**Problem**: Processing 4K videos with hours of content
**Solution**:
- Progressive processing (chunks)
- Adaptive frame sampling (only important frames)
- Store results incrementally
- Option to analyze only first N minutes

### Challenge 2: Variable Frame Rates

**Problem**: Videos with variable FPS (VFR)
**Solution**:
- Use ffprobe to detect VFR
- Extract based on timestamps, not frame numbers
- Convert to CFR if needed

### Challenge 3: GPU Memory for Long Videos

**Problem**: Cannot load all frames in VRAM
**Solution**:
- Process in temporal batches
- Clear cache between batches
- Use frame subsampling

### Challenge 4: Audio Sync

**Problem**: Keep audio analysis aligned with video
**Solution**:
- Use timestamps, not frame numbers
- Extract audio separately (ffmpeg)
- Align transcription with video events

---

## 11. Future Enhancements

### Phase 7+: Advanced Features

1. **Live Video Analysis**
   - Real-time webcam analysis
   - Stream processing (RTSP, HLS)
   - DeepStream integration

2. **Video Editing Automation**
   - Auto-trim boring parts
   - Auto-create montages
   - Smart video compression

3. **Multi-Camera Sync**
   - Synchronize footage from multiple cameras
   - 360° video support

4. **Advanced VLM Features**
   - Video question answering
   - Story generation from videos
   - Automatic video chapters

5. **Cloud Processing**
   - Distributed video processing
   - S3/cloud storage integration
   - Batch job management

---

## Sources

- [Spatio-Temporal Context Prompting for Zero-Shot Action Detection | NVIDIA Research](https://research.nvidia.com/publication/2025-02_spatio-temporal-context-prompting-zero-shot-action-detection)
- [DeepStream 3D Action Recognition App — DeepStream documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_3D_Action.html)
- [Advance Video Analytics Using NVIDIA AI Blueprint](https://developer.nvidia.com/blog/advance-video-analytics-ai-agents-using-the-nvidia-ai-blueprint-for-video-search-and-summarization/)
- [Developing Custom Action Recognition with NVIDIA TAO](https://developer.nvidia.com/blog/developing-and-deploying-your-custom-action-recognition-application-without-any-ai-expertise-using-tao-and-deepstream/)
- [NVIDIA TAO ActionRecognitionNet | NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet)
- [New Foundational Models with NVIDIA TAO 5.5](https://developer.nvidia.com/blog/new-foundational-models-and-training-capabilities-with-nvidia-tao-5-5/)
- [TAO Toolkit Overview](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html)

---

## Conclusion

Video support transforms Image Engine from a photo manager into a comprehensive media management system. By leveraging NVIDIA's video AI models and GPU acceleration, we can process videos at 2-4x playback speed on the RTX 2080 Ti.

**Key Benefits**:
- Unified photo + video management
- GPU-accelerated processing
- Advanced action recognition
- Searchable video database
- Auto-highlight generation

**Timeline**: 6-8 weeks for full implementation
**Priority**: High - complements existing image capabilities

**Next Steps**:
1. Begin Phase 1 (foundation)
2. Test with sample videos
3. Integrate with existing Image Engine infrastructure
